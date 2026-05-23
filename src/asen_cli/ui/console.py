from __future__ import annotations

from typing import Any

from rich import box
from rich.align import Align
from rich.console import Console, Group
from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Confirm
from rich.table import Table
from rich.text import Text

from ..core.agent import AgentEvents
from ..core.protocol import PlanStep
from ..utils.safety import truncate_text

BANNER = r"""
    ___   _____ ______ _   __
   /   | / ___// ____// | / /
  / /| | \__ \/ __/  /  |/ /
 / ___ |___/ / /___ / /|  /
/_/  |_/____/_____//_/ |_/
""".strip("\n")

STATUS_STYLES = {
    "pending": "dim",
    "in_progress": "yellow",
    "completed": "green",
    "failed": "red",
}

STATUS_LABELS = {
    "pending": "pending",
    "in_progress": "running",
    "completed": "done",
    "failed": "failed",
}


class AsenConsole:
    def __init__(self, *, verbose: bool = False) -> None:
        self.console = Console()
        self.verbose_enabled = verbose
        self._stream_live: Live | None = None
        self._stream_buffer = ""
        self._last_streamed_message: str | None = None

    def title(self, mode: str = "chat") -> None:
        banner = Text(BANNER, style="bold cyan")
        if mode == "shell":
            subtitle = Text(
                "Shell-heavy agent session · use !command to run inside the workspace",
                style="dim white",
            )
            panel_title = "[bold cyan]asen shell[/]"
        else:
            subtitle = Text(
                "Teaching-friendly CLI coding agent · type /help to start",
                style="dim white",
            )
            panel_title = "[bold cyan]asen chat[/]"
        body = Group(Align.center(banner), Align.center(subtitle))
        self.console.print(
            Panel(
                body,
                title=panel_title,
                subtitle="[dim]!command  /tools  /config  /paste  /exit[/]",
                border_style="bright_cyan",
                box=box.DOUBLE,
                padding=(1, 2),
            )
        )

    def clear(self, mode: str = "chat") -> None:
        self.console.clear()
        self.title(mode)

    def info(self, message: str) -> None:
        self._finish_stream(remember=False)
        self.console.print(f"[cyan]{message}[/]")

    def success(self, message: str) -> None:
        self._finish_stream(remember=False)
        self.console.print(f"[green]{message}[/]")

    def error(self, message: str) -> None:
        self._finish_stream(remember=False)
        self.console.print(
            Panel(message, title="error", border_style="red", box=box.ROUNDED)
        )

    def assistant(self, message: str) -> None:
        self._finish_stream(remember=False)
        if self._last_streamed_message == message:
            self._last_streamed_message = None
            return
        self._last_streamed_message = None
        self.console.print(self._assistant_panel(message))

    def stream_delta(self, chunk: str) -> None:
        if not chunk:
            return
        self._last_streamed_message = None
        self._stream_buffer += chunk
        panel = self._assistant_panel(self._stream_buffer)
        if self._stream_live is None:
            self._stream_live = Live(
                panel,
                console=self.console,
                refresh_per_second=12,
                transient=False,
            )
            self._stream_live.start()
            return
        self._stream_live.update(panel, refresh=True)

    def stream_end(self) -> None:
        self._finish_stream(remember=True)

    def thinking(self, step: int) -> None:
        self._finish_stream(remember=False)
        self.console.print(f"[bright_cyan]╭─ thinking[/] [dim]step {step}[/]")

    def plan(self, steps: list[PlanStep]) -> None:
        self._finish_stream(remember=False)
        table = Table(
            title="Execution Plan",
            box=box.SIMPLE_HEAVY,
            header_style="bold blue",
            border_style="blue",
        )
        table.add_column("#", style="bold blue", no_wrap=True)
        table.add_column("Status", no_wrap=True)
        table.add_column("Task", style="white")
        for step in steps:
            table.add_row(
                step.id,
                _render_status(step.status),
                step.content,
            )
        self.console.print(table)

    def plan_step(self, step: PlanStep) -> None:
        self._finish_stream(remember=False)
        self.console.print(
            f"[blue]├─ plan[/] {step.id}. {step.content} [{_render_status(step.status)}]"
        )

    def tool_call(self, name: str, arguments: dict[str, Any]) -> None:
        self._finish_stream(remember=False)
        if self.verbose_enabled:
            self.console.print(
                Panel(
                    str(arguments),
                    title=f"[yellow]tool call · {name}[/]",
                    border_style="yellow",
                    box=box.ROUNDED,
                )
            )
            return
        self.console.print(f"[yellow]├─ tool[/] [bold]{name}[/]")

    def tool_result(self, name: str, rendered: str) -> None:
        self._finish_stream(remember=False)
        if self.verbose_enabled:
            self.console.print(
                Panel(
                    rendered,
                    title=f"[yellow]tool result · {name}[/]",
                    border_style="yellow",
                    box=box.ROUNDED,
                )
            )

    def shell_command(self, command: str) -> None:
        self._finish_stream(remember=False)
        self.console.print(f"[cyan]├─ shell[/] [bold]{command}[/]")

    def shell_result(self, command: str, rendered: str) -> None:
        self._finish_stream(remember=False)
        body = truncate_text(rendered or "(no output)", 4_000)
        title_command = truncate_text(command, 96).replace("\n", " ")
        self.console.print(
            Panel(
                body,
                title=f"[cyan]shell result · {title_command}[/]",
                border_style="cyan",
                box=box.ROUNDED,
            )
        )

    def raw_llm_response(self, raw_response: str) -> None:
        if self.verbose_enabled:
            self.console.print(
                Panel(
                    raw_response,
                    title="raw llm response",
                    border_style="dim",
                    box=box.ROUNDED,
                )
            )

    def tools(self, schemas: list[dict[str, Any]]) -> None:
        self._finish_stream(remember=False)
        table = Table(
            title="Agent Toolbelt",
            box=box.SIMPLE_HEAVY,
            header_style="bold cyan",
            border_style="cyan",
        )
        table.add_column("Tool", style="bold cyan", no_wrap=True)
        table.add_column("Description", style="white")
        for schema in schemas:
            table.add_row(str(schema.get("name", "")), str(schema.get("description", "")))
        self.console.print(table)

    def config(self, data: dict[str, Any]) -> None:
        self._finish_stream(remember=False)
        table = Table(
            title="Runtime Config",
            box=box.SIMPLE_HEAVY,
            header_style="bold magenta",
            border_style="magenta",
        )
        table.add_column("Key", style="bold magenta", no_wrap=True)
        table.add_column("Value", style="white")
        for key, value in data.items():
            table.add_row(str(key), str(value))
        self.console.print(table)

    def sessions(self, rows: list[dict[str, Any]]) -> None:
        self._finish_stream(remember=False)
        table = Table(
            title="Saved Sessions",
            box=box.SIMPLE_HEAVY,
            header_style="bold cyan",
            border_style="cyan",
        )
        table.add_column("Session ID", style="bold cyan", no_wrap=True)
        table.add_column("Alias", style="green", no_wrap=True)
        table.add_column("Mode", no_wrap=True)
        table.add_column("Updated", no_wrap=True)
        table.add_column("Turns", justify="right", no_wrap=True)
        table.add_column("Tools", justify="right", no_wrap=True)
        table.add_column("Summary", style="white")
        for row in rows:
            table.add_row(
                str(row.get("session_id", "")),
                str(row.get("alias", "") or ""),
                str(row.get("session_mode", "")),
                str(row.get("updated_at", "")),
                str(row.get("turn_count", "")),
                str(row.get("tool_call_count", "")),
                str(row.get("summary_preview", "")),
            )
        self.console.print(table)

    def checkpoints(self, rows: list[dict[str, Any]]) -> None:
        self._finish_stream(remember=False)
        table = Table(
            title="Saved Checkpoints",
            box=box.SIMPLE_HEAVY,
            header_style="bold magenta",
            border_style="magenta",
        )
        table.add_column("Checkpoint ID", style="bold magenta", no_wrap=True)
        table.add_column("Tool", no_wrap=True)
        table.add_column("Updated", no_wrap=True)
        table.add_column("Files", justify="right", no_wrap=True)
        table.add_column("Summary", style="white")
        for row in rows:
            table.add_row(
                str(row.get("checkpoint_id", "")),
                str(row.get("tool_name", "")),
                str(row.get("updated_at", "")),
                str(row.get("file_count", "")),
                str(row.get("summary_preview", "")),
            )
        self.console.print(table)

    def checkpoint_diff(self, checkpoint_id: str, rendered: str) -> None:
        self._finish_stream(remember=False)
        self.console.print(
            Panel(
                rendered or "(empty diff)",
                title=f"checkpoint diff · {checkpoint_id}",
                border_style="magenta",
                box=box.ROUNDED,
            )
        )

    def help(self, message: str) -> None:
        self._finish_stream(remember=False)
        self.console.print(
            Panel(
                message,
                title="slash commands",
                border_style="cyan",
                box=box.ROUNDED,
                padding=(1, 2),
            )
        )

    def paste_hint(self) -> None:
        self._finish_stream(remember=False)
        self.console.print(
            Panel(
                "Paste multiline input below. Finish with a single EOF line.",
                title="paste mode",
                border_style="blue",
                box=box.ROUNDED,
            )
        )

    def goodbye(self) -> None:
        self._finish_stream(remember=False)
        self.console.print("[dim]session closed. bye.[/]")

    def verbose(self, message: str) -> None:
        self._finish_stream(remember=False)
        if self.verbose_enabled:
            self.console.print(f"[dim]{message}[/]")

    def confirm(self, message: str) -> bool:
        self._finish_stream(remember=False)
        self.console.print(
            Panel(message, title="approval required", border_style="magenta", box=box.ROUNDED)
        )
        return Confirm.ask("Approve", default=False)

    def agent_events(self) -> AgentEvents:
        return AgentEvents(
            on_thinking=self.thinking,
            on_llm_response=self.raw_llm_response,
            on_stream_delta=self.stream_delta,
            on_stream_end=self.stream_end,
            on_plan=self.plan,
            on_plan_step=self.plan_step,
            on_tool_call=self.tool_call,
            on_tool_result=self.tool_result,
        )

    def _assistant_panel(self, message: str) -> Panel:
        return Panel(
            Markdown(message or " "),
            title="[bold green]asen[/]",
            border_style="green",
            box=box.ROUNDED,
            padding=(1, 2),
        )

    def _finish_stream(self, *, remember: bool) -> None:
        if self._stream_live is None:
            if not remember and self._stream_buffer:
                self._stream_buffer = ""
            return
        self._stream_live.update(self._assistant_panel(self._stream_buffer), refresh=True)
        self._stream_live.stop()
        self._stream_live = None
        self._last_streamed_message = (
            self._stream_buffer if remember and self._stream_buffer else None
        )
        self._stream_buffer = ""


def _render_status(status: str) -> str:
    style = STATUS_STYLES.get(status, "white")
    label = STATUS_LABELS.get(status, status)
    return f"[{style}]{label}[/]"
