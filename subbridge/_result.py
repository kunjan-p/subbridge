"""Collect one CLI turn into a TurnResult."""

from __future__ import annotations

import time
from typing import Any

from ._errors import (
    ClaudeProtocolError,
    ClaudeTurnError,
    CodexProtocolError,
    CodexTurnError,
)
from ._events import normalize_event
from ._models import ProviderName, TurnResult
from ._process import describe_turn_failure


class TurnCollector:
    def __init__(self, provider: ProviderName, model: str | None) -> None:
        self.provider = provider
        self.started = time.monotonic()
        self.completed = False
        self.error: str | None = None
        self.result = TurnResult(
            text="", thread_id=None, usage=None, provider=provider, model=model
        )

    def add(self, event: dict[str, Any]) -> None:
        normalized = normalize_event(self.provider, event)
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
