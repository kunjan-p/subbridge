"""POST /v1/messages in the Anthropic Messages shape, answered by Claude Code."""

from __future__ import annotations

import uuid
from collections.abc import Iterator
from typing import Any

from .._models import ProviderName, TurnResult, Usage
from ._errors import ErrorStyle, GatewayError, error_body
from ._transcript import (
    flatten,
    invalid,
    message_turns,
    model_name,
    reject_params,
    text_turn,
)
from ._turns import Frame, TextStream, TurnRequest

REJECTED = ("tools", "tool_choice", "mcp_servers", "container")
ROLES = {"user": "user", "assistant": "assistant"}


class MessagesEndpoint:
    provider: ProviderName = "claude"
    style: ErrorStyle = "anthropic"

    def __init__(self) -> None:
        self.id = f"msg_{uuid.uuid4().hex}"
        self.model = "default"

    def parse(self, body: dict[str, Any]) -> TurnRequest:
        reject_params(body, REJECTED)
        output_config = body.get("output_config")
        if isinstance(output_config, dict) and output_config.get("format"):
            raise invalid(
                "Structured output is not supported on /v1/messages.",
                "output_config.format",
            )
        model = model_name(body)
        self.model = model or self.model
        turns = text_turn("system", body.get("system"), "system")
        turns += message_turns(body.get("messages"), ROLES, "messages")
        return TurnRequest(
            prompt=flatten(turns), model=model, stream=body.get("stream") is True
        )

    def respond(self, result: TurnResult) -> dict[str, Any]:
        content = [{"type": "text", "text": result.text}]
        return self._message(content, "end_turn", result.usage)

    def stream(self, texts: TextStream) -> Iterator[Frame]:
        yield _event("message_start", message=self._message([], None, None))
        block = {"type": "text", "text": ""}
        yield _event("content_block_start", index=0, content_block=block)
        for text in texts:
            delta = {"type": "text_delta", "text": text}
            yield _event("content_block_delta", index=0, delta=delta)
        yield _event("content_block_stop", index=0)
        stop = {"stop_reason": "end_turn", "stop_sequence": None}
        yield _event("message_delta", delta=stop, usage=_usage(texts.usage))
        yield _event("message_stop")

    def stream_error(self, error: GatewayError) -> list[Frame]:
        return [("error", error_body("anthropic", error))]

    def _message(
        self,
        content: list[dict[str, Any]],
        stop_reason: str | None,
        usage: Usage | None,
    ) -> dict[str, Any]:
        return {
            "id": self.id,
            "type": "message",
            "role": "assistant",
            "model": self.model,
            "content": content,
            "stop_reason": stop_reason,
            "stop_sequence": None,
            "usage": _usage(usage),
        }


def _event(kind: str, **fields: Any) -> Frame:
    return kind, {"type": kind, **fields}


def _usage(usage: Usage | None) -> dict[str, int]:
    usage = usage or Usage()
    return {
        "input_tokens": usage.input_tokens,
        "output_tokens": usage.output_tokens,
        "cache_read_input_tokens": usage.cached_input_tokens,
        "cache_creation_input_tokens": usage.cache_write_input_tokens,
    }
