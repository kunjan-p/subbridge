"""POST /v1/messages in the Anthropic Messages shape, answered by Claude Code."""

from __future__ import annotations

import uuid
from typing import Any

from ..models import ProviderName, TurnResult, Usage
from ._errors import ErrorStyle
from ._transcript import (
    flatten,
    invalid,
    message_turns,
    model_name,
    reject_params,
    text_turn,
)
from ._turns import TurnRequest

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


def _usage(usage: Usage | None) -> dict[str, int]:
    usage = usage or Usage()
    return {
        "input_tokens": usage.input_tokens,
        "output_tokens": usage.output_tokens,
        "cache_read_input_tokens": usage.cached_input_tokens,
        "cache_creation_input_tokens": usage.cache_write_input_tokens,
    }
