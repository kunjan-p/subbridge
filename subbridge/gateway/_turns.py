"""Types shared by the gateway's server and its API endpoints."""

from __future__ import annotations

from collections.abc import Generator, Iterator
from dataclasses import dataclass
from typing import Any, Protocol

from .._process import describe_turn_failure
from ..errors import ClaudeTurnError, CodexTurnError
from ..models import ProviderName, StreamEvent, TurnResult, Usage
from ._errors import ErrorStyle, GatewayError

# One server-sent event: (event name or None, JSON payload or raw data string).
Frame = tuple[str | None, Any]


@dataclass
class TurnRequest:
    """One API request, reduced to what the CLI needs."""

    prompt: str
    model: str | None = None
    stream: bool = False
    output_schema: dict[str, Any] | None = None


class Endpoint(Protocol):
    """One API route: it parses its requests and renders replies in its shape."""

    provider: ProviderName
    style: ErrorStyle

    def parse(self, body: dict[str, Any]) -> TurnRequest: ...

    def respond(self, result: TurnResult) -> dict[str, Any]: ...

    def stream(self, texts: TextStream) -> Iterator[Frame]: ...

    def stream_error(self, error: GatewayError) -> list[Frame]: ...


class TextStream:
    """The text of one CLI turn as it arrives; raises the provider's turn error."""

    def __init__(
        self,
        provider: ProviderName,
        events: Generator[StreamEvent, None, None],
        *,
        only_final_message: bool = False,
    ) -> None:
        self.provider = provider
        self.events = events
        self.usage: Usage | None = None
        # A structured-output request needs one JSON document, not a preamble
        # followed by one, so its stream buffers every "message" event and
        # emits only the last -- matching the non-streamed `TurnCollector`,
        # which overwrites (rather than appends to) its text for Codex -- so
        # a streamed and non-streamed reply to the same request parse
        # identically. Anthropic's endpoint never sets `output_schema`, so it
        # never asks for this and keeps the join-everything behavior below.
        self.only_final_message = only_final_message
        self._texts = self._read()
        self._first: str | None = None

    def prime(self) -> None:
        """Wait for the first text, so an early failure still gets an HTTP status."""
        self._first = next(self._texts, None)

    def __iter__(self) -> Iterator[str]:
        if self._first is not None:
            yield self._first
        yield from self._texts

    def close(self) -> None:
        self._texts.close()
        self.events.close()

    def _read(self) -> Generator[str, None, None]:
        if self.only_final_message:
            yield from self._read_final_message()
            return
        started = False
        for event in self.events:
            if event.usage is not None:
                self.usage = event.usage
            if event.kind == "turn_error":
                raise turn_error(self.provider, event.text)
            if event.kind == "message" and event.text:
                # Each "message" event is a separate assistant message (for
                # example, one before a tool call and one after), not a
                # fragment of the same one -- join them with a blank line so
                # they don't run together, but never before the first.
                yield f"\n\n{event.text}" if started else event.text
                started = True

    def _read_final_message(self) -> Generator[str, None, None]:
        final: str | None = None
        for event in self.events:
            if event.usage is not None:
                self.usage = event.usage
            if event.kind == "turn_error":
                raise turn_error(self.provider, event.text)
            if event.kind == "message" and event.text:
                final = event.text
        if final is not None:
            yield final


def turn_error(provider: ProviderName, text: str | None) -> Exception:
    label, error = (
        ("Claude Code", ClaudeTurnError)
        if provider == "claude"
        else ("Codex", CodexTurnError)
    )
    return error(describe_turn_failure(label, text or f"{label} turn failed"))
