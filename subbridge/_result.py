"""Collect one turn identically for synchronous and asynchronous callers."""

from __future__ import annotations

import time
from typing import Any

from ._process import describe_turn_failure
from .errors import (
    ClaudeProtocolError,
    ClaudeTurnError,
    CodexProtocolError,
    CodexTurnError,
)
from .events import normalize_event
from .models import ProviderName, TurnResult


class TurnCollector:
    def __init__(
        self, provider: ProviderName, model: str | None, include_events: bool
    ) -> None:
        self.provider = provider
        self.include_events = include_events
        self.started = time.monotonic()
        self.completed = False
        self.error: str | None = None
        self.result = TurnResult(
            text="", thread_id=None, usage=None, provider=provider, model=model
        )

    def add(self, event: dict[str, Any]) -> None:
        normalized = normalize_event(self.provider, event)
        if self.include_events:
            self.result.events.append(event)
            if self.provider == "claude" and event.get("type") == "assistant":
                self.result.items.append(event)
            elif self.provider == "codex" and event.get("type") == "item.completed":
                item = event.get("item")
                self.result.items.append(item if isinstance(item, dict) else {})
        if normalized.kind == "message":
            if self.provider == "claude":
                self.result.text += normalized.text or ""
            else:
                self.result.text = normalized.text or ""
        elif normalized.kind == "turn_completed":
            self.completed = True
        elif normalized.kind == "turn_error":
            self.error = str(normalized.text or f"{self.provider.title()} turn failed")
        if normalized.usage is not None:
            self.result.usage = normalized.usage
        if self.provider == "claude" and event.get("type") == "result":
            self.result.text = event.get("result", self.result.text)
            self.result.structured_output = event.get("structured_output")

    def finish(self, thread_id: str | None) -> TurnResult:
        if self.provider == "claude":
            label, turn_error, protocol_error = (
                "Claude Code",
                ClaudeTurnError,
                ClaudeProtocolError,
            )
            missing = "Claude Code completed without a terminal result event."
        else:
            label, turn_error, protocol_error = (
                "Codex",
                CodexTurnError,
                CodexProtocolError,
            )
            missing = "Codex completed without a terminal turn.completed event."
        if self.error:
            raise turn_error(describe_turn_failure(label, self.error))
        if not self.completed:
            raise protocol_error(missing)
        self.result.thread_id = thread_id
        self.result.elapsed_seconds = time.monotonic() - self.started
        return self.result
