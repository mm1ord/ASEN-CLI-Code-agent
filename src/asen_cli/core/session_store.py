from __future__ import annotations

import json
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from ..config import AsenConfig
from ..tools.base import ToolExecutionLog
from ..utils.errors import AsenError
from ..utils.safety import truncate_text
from .context_manager import ContextManager, render_plan

SESSION_ROOT_DIRNAME = "sessions"
DEFAULT_SUMMARY = "No final summary yet."
SLUG_PATTERN = re.compile(r"[^a-z0-9]+")


class SessionMetadata(BaseModel):
    session_id: str
    title: str
    session_mode: str
    workspace: str
    provider: str
    model: str
    created_at: str
    updated_at: str
    message_count: int = 0
    turn_count: int = 0
    tool_call_count: int = 0
    summary_preview: str = ""
    final_summary: str = DEFAULT_SUMMARY
    last_user_message: str | None = None
    closed_at: str | None = None
    alias: str | None = None


class SessionStore:
    def __init__(self, root: Path, meta: SessionMetadata) -> None:
        self.root = root
        self.meta = meta
        self.events_path = self.root / "events.jsonl"
        self.meta_path = self.root / "meta.json"
        self.summary_path = self.root / "summary.md"

    @property
    def session_id(self) -> str:
        return self.meta.session_id

    @classmethod
    def create(
        cls,
        config: AsenConfig,
        *,
        session_mode: str,
        session_id: str | None = None,
    ) -> SessionStore:
        root_dir = session_root(config.workspace)
        root_dir.mkdir(parents=True, exist_ok=True)
        resolved_id = session_id or _next_session_id(root_dir, config.workspace.name)
        session_dir = root_dir / resolved_id
        session_dir.mkdir(parents=True, exist_ok=False)
        now = _utcnow()
        meta = SessionMetadata(
            session_id=resolved_id,
            title=f"{session_mode} · {config.workspace.name}",
            session_mode=session_mode,
            workspace=str(config.workspace),
            provider=config.provider,
            model=config.model,
            created_at=now,
            updated_at=now,
        )
        store = cls(session_dir, meta)
        store._write_meta()
        store._write_summary(DEFAULT_SUMMARY)
        store.append_event(
            "session_started",
            session_mode=session_mode,
            workspace=str(config.workspace),
            provider=config.provider,
            model=config.model,
        )
        return store

    @classmethod
    def open(cls, workspace: Path, session_id: str) -> SessionStore:
        session_dir = resolve_session_dir(workspace, session_id)
        if session_dir is None:
            raise AsenError(f"Session not found: {session_id}")
        if not session_dir.is_dir():
            raise AsenError(f"Invalid session directory: {session_dir}")
        meta_path = session_dir / "meta.json"
        if not meta_path.exists():
            raise AsenError(f"Session metadata missing: {meta_path}")
        meta = SessionMetadata.model_validate_json(meta_path.read_text(encoding="utf-8"))
        return cls(session_dir, meta)

    @staticmethod
    def list(workspace: Path) -> list[SessionMetadata]:
        root = session_root(workspace)
        if not root.exists():
            return []
        sessions: list[SessionMetadata] = []
        for meta_path in root.glob("*/meta.json"):
            try:
                sessions.append(
                    SessionMetadata.model_validate_json(meta_path.read_text(encoding="utf-8"))
                )
            except Exception:
                continue
        sessions.sort(key=lambda item: item.updated_at, reverse=True)
        return sessions

    def append_event(self, event_type: str, **data: Any) -> None:
        payload = {
            "type": event_type,
            "timestamp": _utcnow(),
            "session_id": self.session_id,
            **data,
        }
        self.events_path.parent.mkdir(parents=True, exist_ok=True)
        with self.events_path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(payload, ensure_ascii=False) + "\n")
        self.meta.updated_at = payload["timestamp"]
        self._write_meta()

    def record_resume(self) -> None:
        self.append_event("session_resumed")

    def record_user_message(self, content: str) -> None:
        stripped = content.strip()
        self.meta.turn_count += 1
        self.meta.last_user_message = truncate_text(stripped, 240) if stripped else None
        if self._should_replace_title() and stripped:
            self.meta.title = truncate_text(_single_line(stripped), 60)
        self.append_event("user_message", content=content)

    def record_assistant_message(self, content: str) -> None:
        self.append_event("assistant_message", content=content)

    def record_shell_command(self, command: str, rendered: str, *, ok: bool) -> None:
        self.append_event(
            "shell_command",
            command=command,
            ok=ok,
            rendered=truncate_text(rendered, 4_000),
        )

    def record_tool_logs(self, logs: list[ToolExecutionLog]) -> None:
        if not logs:
            return
        for log in logs:
            self.append_event("tool_call", **log.model_dump(mode="json"))
        self.meta.tool_call_count += len(logs)
        self._write_meta()

    def record_snapshot(self, context: ContextManager, *, final_text: str | None = None) -> None:
        snapshot = context.snapshot()
        self.meta.message_count = len(snapshot["messages"])
        summary = build_final_summary(context, final_text=final_text)
        self.meta.final_summary = summary
        self.meta.summary_preview = truncate_text(_single_line(summary), 120)
        self.append_event("context_snapshot", snapshot=snapshot)
        self._write_summary(summary)
        self._write_meta()

    def close(self, context: ContextManager, *, reason: str = "user_exit") -> None:
        self.meta.closed_at = _utcnow()
        self.record_snapshot(context)
        self.append_event("session_closed", reason=reason)
        self._write_meta()

    def latest_snapshot(self) -> dict[str, Any] | None:
        if not self.events_path.exists():
            return None
        latest: dict[str, Any] | None = None
        for line in self.events_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if event.get("type") == "context_snapshot" and isinstance(event.get("snapshot"), dict):
                latest = event["snapshot"]
        return latest

    def _should_replace_title(self) -> bool:
        return self.meta.title == f"{self.meta.session_mode} · {Path(self.meta.workspace).name}"

    def _write_meta(self) -> None:
        self.meta_path.parent.mkdir(parents=True, exist_ok=True)
        self.meta_path.write_text(
            json.dumps(self.meta.model_dump(mode="json"), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def _write_summary(self, summary: str) -> None:
        body = "# Final Summary\n\n" + (summary.strip() or DEFAULT_SUMMARY) + "\n"
        self.summary_path.write_text(body, encoding="utf-8")


def session_root(workspace: Path) -> Path:
    return workspace / ".asen" / SESSION_ROOT_DIRNAME


def resolve_session_dir(workspace: Path, session_id: str) -> Path | None:
    root = session_root(workspace)
    if not root.exists():
        return None
    exact = root / session_id
    if exact.exists():
        return exact
    matches = sorted(
        path
        for path in root.iterdir()
        if path.is_dir() and path.name.startswith(session_id)
    )
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        match_names = ", ".join(path.name for path in matches)
        raise AsenError(
            f"Session id '{session_id}' is ambiguous. Matches: {match_names}"
        )
    return _resolve_by_alias(root, session_id)


def _resolve_by_alias(root: Path, alias: str) -> Path | None:
    matches: list[Path] = []
    for meta_path in root.glob("*/meta.json"):
        try:
            meta = SessionMetadata.model_validate_json(meta_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if meta.alias == alias:
            matches.append(meta_path.parent)
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        match_names = ", ".join(path.name for path in sorted(matches))
        raise AsenError(f"Alias '{alias}' is ambiguous. Matches: {match_names}")
    return None


def build_final_summary(context: ContextManager, *, final_text: str | None = None) -> str:
    edits = [
        fact.removeprefix("File edit: ")
        for fact in context.facts
        if fact.startswith("File edit:")
    ]
    failures = [
        fact.removeprefix("Tool failure from ")
        for fact in context.facts
        if fact.startswith("Tool failure from ")
    ]
    latest = final_text or context.latest_assistant_message() or ""
    sections: list[str] = []
    if edits:
        sections.append("## Recent file edits\n\n" + "\n".join(f"- {item}" for item in edits[-6:]))
    if context.plan_steps:
        sections.append("## Current plan\n\n" + render_plan(context.plan_steps))
    if failures:
        sections.append(
            "## Recent tool failures\n\n" + "\n".join(f"- {item}" for item in failures[-4:])
        )
    if latest:
        sections.append("## Latest assistant answer\n\n" + truncate_text(latest, 2_000))
    return "\n\n".join(sections) or DEFAULT_SUMMARY


def _next_session_id(root: Path, workspace_name: str) -> str:
    stamp = datetime.now().strftime("%Y-%m-%d-%H%M%S")
    slug = _slugify(workspace_name) or "session"
    base = f"{stamp}-{slug}"
    candidate = base
    index = 2
    while (root / candidate).exists():
        candidate = f"{base}-{index}"
        index += 1
    return candidate


def _slugify(text: str) -> str:
    lowered = text.strip().lower()
    normalized = SLUG_PATTERN.sub("-", lowered).strip("-")
    return normalized[:32]


def _single_line(text: str) -> str:
    return " ".join(text.split())


def _utcnow() -> str:
    return datetime.now(UTC).isoformat()
