"""POST /v1/chat/completions in the OpenAI Chat Completions shape, answered by Codex."""

from __future__ import annotations

import time
import uuid
from collections.abc import Iterator
from typing import Any

from ..models import ProviderName, TurnResult, Usage
from ._errors import ErrorStyle, GatewayError, error_body
from ._transcript import (
    OPENAI_ROLES,
    flatten,
    invalid,
    json_schema_format,
    message_turns,
    model_name,
    reject_params,
)
from ._turns import Frame, TextStream, TurnRequest

REJECTED = ("tools", "tool_choice", "functions", "function_call", "audio")


class ChatCompletionsEndpoint:
    provider: ProviderName = "codex"
    style: ErrorStyle = "openai"

    def __init__(self) -> None:
        self.id = f"chatcmpl-{uuid.uuid4().hex}"
        self.created = int(time.time())
        self.model = "default"

    def parse(self, body: dict[str, Any]) -> TurnRequest:
        reject_params(body, REJECTED)
        if body.get("n") not in (None, 1):
            raise invalid("`n` must be 1; the gateway returns one choice.", "n")
        model = model_name(body)
        self.model = model or self.model
        turns = message_turns(body.get("messages"), OPENAI_ROLES, "messages")
        return TurnRequest(
            prompt=flatten(turns),
            model=model,
            stream=body.get("stream") is True,
            output_schema=json_schema_format(
                body.get("response_format"), "response_format"
            ),
        )

    def respond(self, result: TurnResult) -> dict[str, Any]:
        message = {"role": "assistant", "content": result.text, "refusal": None}
        choice = {
            "index": 0,
            "message": message,
            "finish_reason": "stop",
            "logprobs": None,
        }
        return {
            **self._envelope("chat.completion", choice),
            "usage": _usage(result.usage),
        }

    def stream(self, texts: TextStream) -> Iterator[Frame]:
        yield None, self._chunk({"role": "assistant", "content": ""}, None)
        for text in texts:
            yield None, self._chunk({"content": text}, None)
        yield None, self._chunk({}, "stop")
        yield None, "[DONE]"

    def stream_error(self, error: GatewayError) -> list[Frame]:
        return [(None, error_body("openai", error))]

    def _chunk(
        self, delta: dict[str, Any], finish_reason: str | None
    ) -> dict[str, Any]:
        choice = {"index": 0, "delta": delta, "finish_reason": finish_reason}
        return self._envelope("chat.completion.chunk", choice)

    def _envelope(self, kind: str, choice: dict[str, Any]) -> dict[str, Any]:
        return {
            "id": self.id,
            "object": kind,
            "created": self.created,
            "model": self.model,
            "choices": [choice],
        }


def _usage(usage: Usage | None) -> dict[str, int]:
    usage = usage or Usage()
    return {
        "prompt_tokens": usage.input_tokens,
        "completion_tokens": usage.output_tokens,
        "total_tokens": usage.input_tokens + usage.output_tokens,
    }
