from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ParsedSlashCommand:
    name: str
    args: list[str]


@dataclass(frozen=True, slots=True)
class SlashCommandSpec:
    name: str
    description: str


COMMANDS: tuple[SlashCommandSpec, ...] = (
    SlashCommandSpec("/help", "Show available slash commands."),
    SlashCommandSpec("/clear", "Clear the terminal screen."),
    SlashCommandSpec("/exit", "Exit the interactive session."),
    SlashCommandSpec("/quit", "Exit the interactive session."),
    SlashCommandSpec("/tools", "List tools available to the agent."),
    SlashCommandSpec("/config", "Show resolved runtime configuration without API key."),
    SlashCommandSpec("/rename", "Rename the current session: /rename <new title>"),
    SlashCommandSpec("/paste", "Enter multiline input. Finish with a single EOF line."),
)


def parse_slash_command(text: str) -> ParsedSlashCommand | None:
    stripped = text.strip()
    if not stripped.startswith("/"):
        return None
    parts = stripped.split()
    return ParsedSlashCommand(name=parts[0].lower(), args=parts[1:])


def help_text() -> str:
    rows = ["Slash commands:"]
    rows.extend(f"  {spec.name:<8} {spec.description}" for spec in COMMANDS)
    rows.extend(
        [
            "",
            "Bang commands:",
            "  !<command> Run a shell command in the workspace.",
            "             High-risk commands require approval; blocked commands are rejected.",
        ]
    )
    return "\n".join(rows)
