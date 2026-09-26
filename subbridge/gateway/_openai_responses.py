"""POST /v1/responses in the OpenAI Responses shape, answered by Codex."""

from __future__ import annotations

import time
import uuid
from collections.abc import Iterator
from typing import Any

from .._models import ProviderName, TurnResult, Usage
from ._errors import ErrorStyle, GatewayError, openai_error
from ._transcript import (
    OPENAI_ROLES,
    flatten,
    invalid,
    json_schema_format,
    message_turns,
    model_name,
    reject_params,
    text_turn,
)
from ._turns import Frame, TextStream, TurnRequest

REJECTED = (
    "tools",
    "tool_choice",
    "previous_response_id",
    "conversation",
    "prompt",
    "background",
)


class ResponsesEndpoint:
    provider: ProviderName = "codex"
    style: ErrorStyle = "openai"

    def __init__(self) -> None:
        self.id = f"resp_{uuid.uuid4().hex}"
        self.item_id = f"msg_{uuid.uuid4().hex}"
        self.created = int(time.time())
        self.model = "default"
        self.instructions: str | None = None
        self.sequence = 0

    def parse(self, body: dict[str, Any]) -> TurnRequest:
        reject_params(body, REJECTED)
        model = model_name(body)
        self.model = model or self.model
        instructions = body.get("instructions")
        turns = text_turn("system", instructions, "instructions")
        self.instructions = instructions
        source = body.get("input")
        if isinstance(source, str):
            turns.append(("user", source))
        else:
            turns += message_turns(source, OPENAI_ROLES, "input")
        text_config = body.get("text")
        if text_config is not None and not isinstance(text_config, dict):
            raise invalid("`text` must be an object.", "text")
        text_format = text_config.get("format") if text_config else None
        return TurnRequest(
            prompt=flatten(turns),
            model=model,
            stream=body.get("stream") is True,
            output_schema=json_schema_format(text_format, "text.format"),
        )

    def respond(self, result: TurnResult) -> dict[str, Any]:
        return self._response(
            "completed", [self._item(result.text, "completed")], result.usage
        )

    def stream(self, texts: TextStream) -> Iterator[Frame]:
        yield self._event(
            "response.created", response=self._response("in_progress", [], None)
        )
        yield self._event(
            "response.output_item.added",
            output_index=0,
            item=self._item(None, "in_progress"),
        )
        yield self._part_event("response.content_part.added", part=_text_part(""))
        parts = []
        for text in texts:
            parts.append(text)
            yield self._part_event(
                "response.output_text.delta", delta=text, logprobs=[]
            )
        text = "".join(parts)
        yield self._part_event("response.output_text.done", text=text, logprobs=[])
        yield self._part_event("response.content_part.done", part=_text_part(text))
        item = self._item(text, "completed")
        yield self._event("response.output_item.done", output_index=0, item=item)
        response = self._response("completed", [item], texts.usage)
        yield self._event("response.completed", response=response)

    def stream_error(self, error: GatewayError) -> list[Frame]:
        fields = openai_error(error)
        return [
            self._event(
                "error",
                code=fields["code"],
                message=fields["message"],
                param=fields["param"],
            )
        ]

    def _event(self, kind: str, **fields: Any) -> Frame:
        payload = {"type": kind, "sequence_number": self.sequence, **fields}
        self.sequence += 1
        return kind, payload

    def _part_event(self, kind: str, **fields: Any) -> Frame:
        return self._event(
            kind, item_id=self.item_id, output_index=0, content_index=0, **fields
        )

    def _item(self, text: str | None, status: str) -> dict[str, Any]:
        return {
            "type": "message",
            "id": self.item_id,
            "role": "assistant",
            "status": status,
            "content": [] if text is None else [_text_part(text)],
        }

    def _response(
        self, status: str, output: list[dict[str, Any]], usage: Usage | None
    ) -> dict[str, Any]:
        return {
            "id": self.id,
            "object": "response",
            "created_at": self.created,
            "status": status,
            "model": self.model,
            "output": output,
            "usage": None if usage is None else _usage(usage),
            "error": None,
            "incomplete_details": None,
            "instructions": self.instructions,
            "metadata": {},
            "parallel_tool_calls": False,
            "tool_choice": "none",
            "tools": [],
        }


def _text_part(text: str) -> dict[str, Any]:
    return {"type": "output_text", "text": text, "annotations": [], "logprobs": []}


def _usage(usage: Usage) -> dict[str, Any]:
    return {
        "input_tokens": usage.input_tokens,
        "input_tokens_details": {
            "cached_tokens": usage.cached_input_tokens,
            "cache_write_tokens": usage.cache_write_input_tokens,
        },
        "output_tokens": usage.output_tokens,
        "output_tokens_details": {"reasoning_tokens": usage.reasoning_output_tokens},
        "total_tokens": usage.input_tokens + usage.output_tokens,
    }
