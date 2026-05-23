from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Annotated, Any

import typer

from . import __version__
from .config import (
    McpServerConfig,
    get_config_path,
    get_config_value,
    init_config_file,
    load_config,
    mask_config,
    read_config_file,
    set_config_value,
    set_mcp_server_config,
)
from .core.agent import Agent, load_system_prompt
from .core.checkpoint_store import CheckpointStore
from .core.session import ChatSession
from .core.session_store import SessionStore
from .core.shell_mode import InteractiveShellRunner
from .llm.factory import create_llm_client
from .mcp import StdioMcpClient
from .tools import create_default_registry
from .ui.console import AsenConsole
from .ui.input import InputReader
from .ui.json_console import AsenJsonConsole
from .utils.errors import AsenError

app = typer.Typer(
    name="asen",
    help="Teaching-friendly CLI coding agent.",
    no_args_is_help=False,
    invoke_without_command=True,
)
config_app = typer.Typer(
    help="Show and manage asen cli configuration.",
    invoke_without_command=True,
)
session_app = typer.Typer(
    help="List and resume saved interactive sessions.",
    no_args_is_help=True,
)
checkpoint_app = typer.Typer(
    help="List, diff, and restore saved checkpoints.",
    no_args_is_help=True,
)
mcp_app = typer.Typer(
    help="Configure and call lightweight MCP servers over stdio.",
    no_args_is_help=True,
)


@app.callback(invoke_without_command=True)
def callback(
    ctx: typer.Context,
    version: Annotated[
        bool,
        typer.Option("--version", help="Show version.", is_eager=True),
    ] = False,
    config: Annotated[Path | None, typer.Option("--config", "-c")] = None,
    workspace: Annotated[Path | None, typer.Option("--workspace", "-w")] = None,
    no_approval: Annotated[
        bool,
        typer.Option(help="Disable approval prompts for demo/testing."),
    ] = False,
    verbose: Annotated[
        bool,
        typer.Option("--verbose", "-v", help="Show agent internals."),
    ] = False,
    stream: Annotated[
        bool,
        typer.Option("--stream/--no-stream", help="Stream assistant output as it arrives."),
    ] = True,
) -> None:
    if version:
        typer.echo(f"asen-cli {__version__}")
        raise typer.Exit()
    if ctx.invoked_subcommand is None:
        asyncio.run(_chat(config, workspace, no_approval, verbose, stream))


@app.command()
def chat(
    config: Annotated[Path | None, typer.Option("--config", "-c")] = None,
    workspace: Annotated[Path | None, typer.Option("--workspace", "-w")] = None,
    no_approval: Annotated[
        bool,
        typer.Option(help="Disable approval prompts for demo/testing."),
    ] = False,
    verbose: Annotated[
        bool,
        typer.Option("--verbose", "-v", help="Show agent internals."),
    ] = False,
    stream: Annotated[
        bool,
        typer.Option(
            "--stream/--no-stream",
            help="Stream assistant output as it arrives.",
        ),
    ] = True,
) -> None:
    """Start an interactive coding-agent session."""
    asyncio.run(_chat(config, workspace, no_approval, verbose, stream))


@app.command()
def ask(
    task: Annotated[str, typer.Argument(help="One-shot task for asen cli.")],
    config: Annotated[Path | None, typer.Option("--config", "-c")] = None,
    workspace: Annotated[Path | None, typer.Option("--workspace", "-w")] = None,
    no_approval: Annotated[
        bool,
        typer.Option(help="Disable approval prompts for demo/testing."),
    ] = False,
    verbose: Annotated[
        bool,
        typer.Option("--verbose", "-v", help="Show agent internals."),
    ] = False,
    stream: Annotated[
        bool,
        typer.Option(
            "--stream/--no-stream",
            help="Stream assistant output as it arrives.",
        ),
    ] = True,
    json_output: Annotated[
        bool,
        typer.Option("--json", help="Emit a single machine-readable JSON object."),
    ] = False,
) -> None:
    """Run a one-shot task and print the final answer."""
    asyncio.run(
        _ask(task, config, workspace, no_approval, verbose, stream, json_output)
    )


@app.command()
def shell(
    config: Annotated[Path | None, typer.Option("--config", "-c")] = None,
    workspace: Annotated[Path | None, typer.Option("--workspace", "-w")] = None,
    no_approval: Annotated[
        bool,
        typer.Option(help="Disable approval prompts for demo/testing."),
    ] = False,
    verbose: Annotated[
        bool,
        typer.Option("--verbose", "-v", help="Show agent internals."),
    ] = False,
    stream: Annotated[
        bool,
        typer.Option(
            "--stream/--no-stream",
            help="Stream assistant output as it arrives.",
        ),
    ] = True,
) -> None:
    """Start a shell-heavy interactive session with !command support."""
    asyncio.run(_shell(config, workspace, no_approval, verbose, stream))


@config_app.callback(invoke_without_command=True)
def config_callback(
    ctx: typer.Context,
    config: Annotated[Path | None, typer.Option("--config", "-c")] = None,
    workspace: Annotated[Path | None, typer.Option("--workspace", "-w")] = None,
) -> None:
    """Print resolved configuration without revealing the API key."""
    if ctx.invoked_subcommand is not None:
        return
    console = AsenConsole()
    try:
        cfg = load_config(config, workspace=workspace)
        console.config(mask_config(cfg.model_dump(mode="json")))
    except AsenError as exc:
        console.error(str(exc))
        raise typer.Exit(code=1) from exc


@config_app.command("get")
def config_get(
    key: Annotated[str | None, typer.Argument(help="Config key to read.")] = None,
    global_config: Annotated[
        bool,
        typer.Option(
            "--global",
            help="Read from ~/.asen/config.yaml instead of resolved config.",
        ),
    ] = False,
    project_config: Annotated[
        bool,
        typer.Option(
            "--project",
            help="Read from .asen/config.yaml instead of resolved config.",
        ),
    ] = False,
) -> None:
    """Get resolved configuration or one key."""
    console = AsenConsole()
    try:
        if global_config or project_config:
            path = get_config_path(_config_scope(global_config, project_config))
            data = mask_config(read_config_file(path))
            if key is None:
                console.config(data)
                return
            console.info(str(data.get(key, "")))
            return

        cfg = load_config()
        if key is None:
            console.config(mask_config(cfg.model_dump(mode="json")))
            return
        value = get_config_value(cfg, key)
        console.info("***" if key == "api_key" and value else str(value))
    except AsenError as exc:
        console.error(str(exc))
        raise typer.Exit(code=1) from exc


@config_app.command("set")
def config_set(
    key: Annotated[str, typer.Argument(help="Config key to set.")],
    value: Annotated[str, typer.Argument(help="Config value to write.")],
    global_config: Annotated[
        bool,
        typer.Option("--global", help="Write to ~/.asen/config.yaml."),
    ] = False,
    project_config: Annotated[
        bool,
        typer.Option("--project", help="Write to .asen/config.yaml. Default."),
    ] = False,
) -> None:
    """Set a config key in project or global config."""
    console = AsenConsole()
    try:
        path = get_config_path(_config_scope(global_config, project_config))
        set_config_value(path, key, value)
        console.success(f"Set {key} in {path}")
    except (AsenError, ValueError) as exc:
        console.error(str(exc))
        raise typer.Exit(code=1) from exc


@config_app.command("init")
def config_init(
    global_config: Annotated[
        bool,
        typer.Option("--global", help="Create ~/.asen/config.yaml."),
    ] = False,
    project_config: Annotated[
        bool,
        typer.Option("--project", help="Create .asen/config.yaml. Default."),
    ] = False,
    force: Annotated[
        bool,
        typer.Option("--force", "-f", help="Overwrite existing config."),
    ] = False,
) -> None:
    """Create a starter config file."""
    console = AsenConsole()
    try:
        path = get_config_path(_config_scope(global_config, project_config))
        created = init_config_file(path, force=force)
        console.success(f"Initialized config: {created}")
    except AsenError as exc:
        console.error(str(exc))
        raise typer.Exit(code=1) from exc


@session_app.command("list")
def session_list(
    config: Annotated[Path | None, typer.Option("--config", "-c")] = None,
    workspace: Annotated[Path | None, typer.Option("--workspace", "-w")] = None,
) -> None:
    """List saved sessions in the current workspace."""
    console = AsenConsole()
    try:
        cfg = load_config(config, workspace=workspace)
        sessions = SessionStore.list(cfg.workspace)
        if not sessions:
            console.info(
                "No saved sessions yet. Start one with `asen chat` or `asen shell`."
            )
            return
        rows = [
            {
                "session_id": item.session_id,
                "alias": item.alias or "",
                "session_mode": item.session_mode,
                "updated_at": _compact_timestamp(item.updated_at),
                "turn_count": item.turn_count,
                "tool_call_count": item.tool_call_count,
                "summary_preview": item.summary_preview or item.title,
            }
            for item in sessions
        ]
        console.sessions(rows)
    except AsenError as exc:
        console.error(str(exc))
        raise typer.Exit(code=1) from exc


@session_app.command("resume")
def session_resume(
    session_id: Annotated[
        str,
        typer.Argument(help="Session id or unique prefix to resume."),
    ],
    config: Annotated[Path | None, typer.Option("--config", "-c")] = None,
    workspace: Annotated[Path | None, typer.Option("--workspace", "-w")] = None,
    no_approval: Annotated[
        bool,
        typer.Option(help="Disable approval prompts for demo/testing."),
    ] = False,
    verbose: Annotated[
        bool,
        typer.Option("--verbose", "-v", help="Show agent internals."),
    ] = False,
    stream: Annotated[
        bool,
        typer.Option(
            "--stream/--no-stream",
            help="Stream assistant output as it arrives.",
        ),
    ] = True,
) -> None:
    """Resume a saved interactive session."""
    asyncio.run(
        _resume_session(session_id, config, workspace, no_approval, verbose, stream)
    )


@checkpoint_app.command("list")
def checkpoint_list(
    config: Annotated[Path | None, typer.Option("--config", "-c")] = None,
    workspace: Annotated[Path | None, typer.Option("--workspace", "-w")] = None,
    json_output: Annotated[
        bool,
        typer.Option("--json", help="Emit machine-readable JSON."),
    ] = False,
) -> None:
    """List saved checkpoints in the current workspace."""
    console = AsenConsole()
    try:
        cfg = load_config(config, workspace=workspace)
        checkpoints = CheckpointStore.list(cfg.workspace)
        if json_output:
            _print_json(
                {
                    "ok": True,
                    "command": "checkpoint list",
                    "workspace": str(cfg.workspace),
                    "checkpoints": [item.model_dump(mode="json") for item in checkpoints],
                }
            )
            return
        if not checkpoints:
            console.info("No checkpoints yet. Modify files with asen to create one.")
            return
        rows = [
            {
                "checkpoint_id": item.checkpoint_id,
                "tool_name": item.tool_name,
                "updated_at": _compact_timestamp(item.updated_at),
                "file_count": item.file_count,
                "summary_preview": item.summary_preview or item.title,
            }
            for item in checkpoints
        ]
        console.checkpoints(rows)
    except AsenError as exc:
        if json_output:
            _print_json(
                _json_error(
                    "checkpoint list",
                    str(exc),
                    workspace=str(workspace) if workspace else None,
                )
            )
        else:
            console.error(str(exc))
        raise typer.Exit(code=1) from exc


@checkpoint_app.command("diff")
def checkpoint_diff(
    checkpoint_id: Annotated[
        str,
        typer.Argument(help="Checkpoint id or unique prefix to inspect."),
    ],
    config: Annotated[Path | None, typer.Option("--config", "-c")] = None,
    workspace: Annotated[Path | None, typer.Option("--workspace", "-w")] = None,
    json_output: Annotated[
        bool,
        typer.Option("--json", help="Emit machine-readable JSON."),
    ] = False,
) -> None:
    """Show the captured diff for a checkpoint."""
    console = AsenConsole()
    try:
        cfg = load_config(config, workspace=workspace)
        checkpoint = CheckpointStore.open(cfg.workspace, checkpoint_id)
        if json_output:
            _print_json(
                {
                    "ok": True,
                    "command": "checkpoint diff",
                    "workspace": str(cfg.workspace),
                    "checkpoint": checkpoint.metadata_payload(),
                    "changed_files": checkpoint.changed_files_payload(),
                    "diff": checkpoint.render_diff(),
                }
            )
            return
        console.checkpoint_diff(checkpoint.checkpoint_id, checkpoint.render_diff())
    except AsenError as exc:
        if json_output:
            _print_json(
                _json_error(
                    "checkpoint diff",
                    str(exc),
                    workspace=str(workspace) if workspace else None,
                )
            )
        else:
            console.error(str(exc))
        raise typer.Exit(code=1) from exc


@checkpoint_app.command("restore")
def checkpoint_restore(
    checkpoint_id: Annotated[
        str,
        typer.Argument(help="Checkpoint id or unique prefix to restore."),
    ],
    config: Annotated[Path | None, typer.Option("--config", "-c")] = None,
    workspace: Annotated[Path | None, typer.Option("--workspace", "-w")] = None,
    force: Annotated[
        bool,
        typer.Option("--force", help="Restore without interactive confirmation."),
    ] = False,
    json_output: Annotated[
        bool,
        typer.Option("--json", help="Emit machine-readable JSON."),
    ] = False,
) -> None:
    """Restore files captured by a checkpoint."""
    console = AsenConsole()
    try:
        cfg = load_config(config, workspace=workspace)
        checkpoint = CheckpointStore.open(cfg.workspace, checkpoint_id)
        if json_output and not force:
            raise AsenError(
                "checkpoint restore --json requires --force to avoid interactive prompts"
            )
        if not force:
            prompt = (
                f"Restore checkpoint {checkpoint.checkpoint_id}?\n"
                f"This will revert {checkpoint.meta.file_count} file(s).\n"
                f"{checkpoint.meta.summary_preview}"
            )
            if not console.confirm(prompt):
                console.info("Restore cancelled.")
                return
        actions = checkpoint.restore()
        if json_output:
            _print_json(
                {
                    "ok": True,
                    "command": "checkpoint restore",
                    "workspace": str(cfg.workspace),
                    "checkpoint": checkpoint.metadata_payload(),
                    "changed_files": checkpoint.changed_files_payload(),
                    "actions": actions,
                }
            )
            return
        console.success(f"Restored checkpoint {checkpoint.checkpoint_id}")
        for action in actions:
            console.info(action)
    except AsenError as exc:
        if json_output:
            _print_json(
                _json_error(
                    "checkpoint restore",
                    str(exc),
                    workspace=str(workspace) if workspace else None,
                )
            )
        else:
            console.error(str(exc))
        raise typer.Exit(code=1) from exc


@mcp_app.command("list")
def mcp_list(
    config: Annotated[Path | None, typer.Option("--config", "-c")] = None,
    workspace: Annotated[Path | None, typer.Option("--workspace", "-w")] = None,
    json_output: Annotated[
        bool,
        typer.Option("--json", help="Emit machine-readable JSON."),
    ] = False,
) -> None:
    """List configured MCP servers."""
    console = AsenConsole()
    try:
        cfg = load_config(config, workspace=workspace)
        servers = [
            _mcp_server_payload(name, server)
            for name, server in sorted(cfg.mcp_servers.items())
        ]
        if json_output:
            _print_json(
                {
                    "ok": True,
                    "command": "mcp list",
                    "workspace": str(cfg.workspace),
                    "servers": servers,
                }
            )
            return
        if not servers:
            console.info("No MCP servers configured yet. Add one with `asen mcp add ...`.")
            return
        for server in servers:
            command_text = " ".join([server["command"], *server.get("args", [])]).strip()
            status = "enabled" if server.get("enabled", True) else "disabled"
            description = f" :: {server['description']}" if server.get("description") else ""
            console.info(f"{server['name']} [{status}] -> {command_text}{description}")
            if server.get("cwd"):
                console.info(
                    f"  cwd={server['cwd']} timeout={server['timeout_seconds']}s"
                )
    except AsenError as exc:
        if json_output:
            _print_json(
                _json_error(
                    "mcp list",
                    str(exc),
                    workspace=str(workspace) if workspace else None,
                )
            )
        else:
            console.error(str(exc))
        raise typer.Exit(code=1) from exc


@mcp_app.command("add")
def mcp_add(
    server_name: Annotated[str, typer.Argument(help="Logical name for the MCP server.")],
    command: Annotated[str, typer.Argument(help="Executable used to start the MCP server.")],
    server_args: Annotated[
        list[str],
        typer.Option("--arg", help="Argument passed to the MCP server. Repeatable."),
    ] = None,
    cwd: Annotated[
        Path | None,
        typer.Option("--cwd", help="Working directory used when launching the MCP server."),
    ] = None,
    env: Annotated[
        list[str],
        typer.Option("--env", help="Environment variable in KEY=VALUE form. Repeatable."),
    ] = [],  # noqa: B006
    timeout_seconds: Annotated[
        int,
        typer.Option("--timeout", help="Timeout in seconds for MCP requests."),
    ] = 20,
    description: Annotated[
        str | None,
        typer.Option("--description", help="Optional human-readable description."),
    ] = None,
    global_config: Annotated[
        bool,
        typer.Option("--global", help="Write to ~/.asen/config.yaml."),
    ] = False,
    project_config: Annotated[
        bool,
        typer.Option("--project", help="Write to .asen/config.yaml. Default."),
    ] = False,
    disabled: Annotated[
        bool,
        typer.Option("--disabled", help="Save the server config as disabled."),
    ] = False,
) -> None:
    """Add or update an MCP server config entry."""
    console = AsenConsole()
    try:
        path = get_config_path(_config_scope(global_config, project_config))
        server = McpServerConfig(
            command=command,
            args=list(server_args),
            env=_parse_env_pairs(env),
            cwd=cwd,
            timeout_seconds=timeout_seconds,
            enabled=not disabled,
            description=description,
        )
        set_mcp_server_config(path, server_name, server)
        console.success(f"Saved MCP server `{server_name}` in {path}")
        console.info(f"Try `asen mcp tools {server_name}` to verify the connection.")
    except AsenError as exc:
        console.error(str(exc))
        raise typer.Exit(code=1) from exc


@mcp_app.command("tools")
def mcp_tools(
    server_name: Annotated[str, typer.Argument(help="Configured MCP server name.")],
    config: Annotated[Path | None, typer.Option("--config", "-c")] = None,
    workspace: Annotated[Path | None, typer.Option("--workspace", "-w")] = None,
    json_output: Annotated[
        bool,
        typer.Option("--json", help="Emit machine-readable JSON."),
    ] = False,
) -> None:
    """Start an MCP server and list its available tools."""
    console = AsenConsole()
    try:
        payload = asyncio.run(_mcp_tools(server_name, config, workspace))
        if json_output:
            _print_json(payload)
            return
        server_info = payload.get("server_info", {}).get("serverInfo", {})
        console.info(
            (
                f"Connected to {server_name} -> "
                f"{server_info.get('name', server_name)} {server_info.get('version', '')}"
            ).strip()
        )
        console.tools(payload["tools"])
    except AsenError as exc:
        if json_output:
            _print_json(
                _json_error(
                    "mcp tools",
                    str(exc),
                    workspace=str(workspace) if workspace else None,
                )
            )
        else:
            console.error(str(exc))
        raise typer.Exit(code=1) from exc


@mcp_app.command("call")
def mcp_call(
    server_name: Annotated[str, typer.Argument(help="Configured MCP server name.")],
    tool_name: Annotated[str, typer.Argument(help="Tool name exposed by the MCP server.")],
    arguments: Annotated[
        str | None,
        typer.Argument(help="Tool arguments as a JSON object. Defaults to {}."),
    ] = None,
    config: Annotated[Path | None, typer.Option("--config", "-c")] = None,
    workspace: Annotated[Path | None, typer.Option("--workspace", "-w")] = None,
    json_output: Annotated[
        bool,
        typer.Option("--json", help="Emit machine-readable JSON."),
    ] = False,
) -> None:
    """Start an MCP server and call one tool."""
    console = AsenConsole()
    try:
        payload = asyncio.run(
            _mcp_call(server_name, tool_name, arguments, config, workspace)
        )
        if json_output:
            _print_json(payload)
        else:
            rendered = str(payload.get("rendered", ""))
            if payload.get("ok", False):
                console.assistant(rendered)
            else:
                console.error(rendered)
        if not payload.get("ok", False):
            raise typer.Exit(code=1)
    except typer.Exit:
        raise
    except AsenError as exc:
        if json_output:
            _print_json(
                _json_error(
                    "mcp call",
                    str(exc),
                    workspace=str(workspace) if workspace else None,
                )
            )
        else:
            console.error(str(exc))
        raise typer.Exit(code=1) from exc


app.add_typer(config_app, name="config")
app.add_typer(session_app, name="session")
app.add_typer(checkpoint_app, name="checkpoint")
app.add_typer(mcp_app, name="mcp")


async def _chat(
    config_path: Path | None,
    workspace: Path | None,
    no_approval: bool,
    verbose: bool,
    stream: bool,
) -> None:
    await _interactive_session(
        config_path,
        workspace,
        no_approval,
        verbose,
        stream,
        session_mode="chat",
    )


async def _shell(
    config_path: Path | None,
    workspace: Path | None,
    no_approval: bool,
    verbose: bool,
    stream: bool,
) -> None:
    await _interactive_session(
        config_path,
        workspace,
        no_approval,
        verbose,
        stream,
        session_mode="shell",
    )


async def _resume_session(
    session_id: str,
    config_path: Path | None,
    workspace: Path | None,
    no_approval: bool,
    verbose: bool,
    stream: bool,
) -> None:
    await _interactive_session(
        config_path,
        workspace,
        no_approval,
        verbose,
        stream,
        session_mode=None,
        resume_session_id=session_id,
    )


async def _interactive_session(
    config_path: Path | None,
    workspace: Path | None,
    no_approval: bool,
    verbose: bool,
    stream: bool,
    *,
    session_mode: str | None,
    resume_session_id: str | None = None,
) -> None:
    console = AsenConsole(verbose=verbose)
    try:
        agent = _build_agent(config_path, workspace, no_approval, console, stream)
        resumed = resume_session_id is not None
        store = (
            SessionStore.open(agent.config.workspace, resume_session_id)
            if resume_session_id is not None
            else SessionStore.create(agent.config, session_mode=session_mode or "chat")
        )
        snapshot = store.latest_snapshot()
        if snapshot:
            agent.load_context_snapshot(snapshot)
        resolved_mode = store.meta.session_mode if resumed else (session_mode or "chat")
        session = ChatSession(
            agent=agent,
            config=agent.config,
            console=console,
            input_reader=InputReader(),
            shell_runner=InteractiveShellRunner(agent.config, confirm=console.confirm),
            session_mode=resolved_mode,
            session_store=store,
            resumed=resumed,
        )
        await session.run()
    except AsenError as exc:
        console.error(str(exc))
        raise typer.Exit(code=1) from exc


async def _ask(
    task: str,
    config_path: Path | None,
    workspace: Path | None,
    no_approval: bool,
    verbose: bool,
    stream: bool,
    json_output: bool,
) -> None:
    console: AsenConsole | AsenJsonConsole
    console = AsenJsonConsole(verbose=verbose) if json_output else AsenConsole(verbose=verbose)
    workspace_value = str(workspace) if workspace is not None else None
    try:
        agent = _build_agent(
            config_path,
            workspace,
            no_approval,
            console,
            stream=False if json_output else stream,
        )
        workspace_value = str(agent.config.workspace)
        answer = await agent.run(task)
        if json_output and isinstance(console, AsenJsonConsole):
            console.assistant(answer)
            _print_json(console.build_ask_payload(task=task, workspace=agent.config.workspace))
            return
        console.assistant(answer)
    except AsenError as exc:
        if json_output:
            _print_json(_json_error("ask", str(exc), workspace=workspace_value))
        else:
            console.error(str(exc))
        raise typer.Exit(code=1) from exc


def _build_agent(
    config_path: Path | None,
    workspace: Path | None,
    no_approval: bool,
    console: AsenConsole | AsenJsonConsole,
    stream: bool,
) -> Agent:
    overrides = {"require_approval": False} if no_approval else None
    cfg = load_config(config_path, workspace=workspace, overrides=overrides)
    registry = create_default_registry(cfg, confirm=console.confirm)
    return Agent(
        config=cfg,
        llm=create_llm_client(cfg),
        tools=registry,
        system_prompt=load_system_prompt(),
        events=console.agent_events(),
        stream=stream,
    )


def _config_scope(global_config: bool, project_config: bool) -> str:
    if global_config and project_config:
        raise AsenError("Use only one of --global or --project")
    return "global" if global_config else "project"


async def _mcp_tools(
    server_name: str,
    config_path: Path | None,
    workspace: Path | None,
) -> dict[str, Any]:
    cfg = load_config(config_path, workspace=workspace)
    server = _load_mcp_server(cfg, server_name)
    async with StdioMcpClient(server_name, server, workspace=cfg.workspace) as client:
        tools = await client.list_tools()
        return {
            "ok": True,
            "command": "mcp tools",
            "workspace": str(cfg.workspace),
            "server": _mcp_server_payload(server_name, server),
            "server_info": (
                client.initialize_result.model_dump(mode="json")
                if client.initialize_result
                else {}
            ),
            "tools": [tool.model_dump(mode="json") for tool in tools],
        }


async def _mcp_call(
    server_name: str,
    tool_name: str,
    arguments: str | None,
    config_path: Path | None,
    workspace: Path | None,
) -> dict[str, Any]:
    cfg = load_config(config_path, workspace=workspace)
    parsed_arguments = _parse_json_object(arguments)
    server = _load_mcp_server(cfg, server_name)
    async with StdioMcpClient(server_name, server, workspace=cfg.workspace) as client:
        result = await client.call_tool(tool_name, parsed_arguments)
        return {
            "ok": not result.isError,
            "command": "mcp call",
            "workspace": str(cfg.workspace),
            "server": _mcp_server_payload(server_name, server),
            "server_info": (
                client.initialize_result.model_dump(mode="json")
                if client.initialize_result
                else {}
            ),
            "tool": tool_name,
            "arguments": parsed_arguments,
            "result": result.model_dump(mode="json"),
            "rendered": result.render_text(),
        }


def _load_mcp_server(cfg: Any, server_name: str) -> McpServerConfig:
    server = cfg.mcp_servers.get(server_name)
    if server is None:
        raise AsenError(
            f"MCP server `{server_name}` not found. Add one with `asen mcp add {server_name} ...`."
        )
    return server


def _mcp_server_payload(name: str, server: McpServerConfig) -> dict[str, Any]:
    return {
        "name": name,
        **server.model_dump(mode="json", exclude_none=True),
    }


def _parse_env_pairs(values: list[str]) -> dict[str, str]:
    env: dict[str, str] = {}
    for item in values:
        if "=" not in item:
            raise AsenError(f"Invalid --env value `{item}`. Use KEY=VALUE.")
        key, value = item.split("=", 1)
        key = key.strip()
        if not key:
            raise AsenError(f"Invalid --env value `{item}`. KEY cannot be empty.")
        env[key] = value
    return env


def _parse_json_object(raw: str | None) -> dict[str, Any]:
    if raw is None or not raw.strip():
        return {}
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise AsenError(f"Invalid JSON arguments: {exc}") from exc
    if not isinstance(payload, dict):
        raise AsenError("MCP tool arguments must be a JSON object")
    return payload


def _compact_timestamp(value: str) -> str:
    return value.replace("T", " ").replace("+00:00", " UTC")[:23]


def _print_json(payload: dict[str, Any]) -> None:
    typer.echo(json.dumps(payload, ensure_ascii=False, indent=2))


def _json_error(
    command: str,
    message: str,
    *,
    workspace: str | None = None,
    code: str = "asen_error",
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "ok": False,
        "command": command,
        "error": {
            "code": code,
            "message": message,
        },
    }
    if workspace is not None:
        payload["workspace"] = workspace
    return payload
