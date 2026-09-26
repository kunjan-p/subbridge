"""Validate chat-style request bodies and flatten them into one CLI prompt."""

from __future__ import annotations

import re
from typing import Any

from ._errors import GatewayError

Turn = tuple[str, str]

TEXT_PART_TYPES = frozenset({"text", "input_text", "output_text"})
_MODEL_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:/@\[\]-]{0,199}")


def flatten(turns: list[Turn]) -> str:
    """One user turn is sent as-is; anything longer becomes a tagged transcript."""
    if len(turns) == 1 and turns[0][0] == "user":
        return turns[0][1]
    return "\n\n".join(f"<{role}>\n{text}\n</{role}>" for role, text in turns)


def model_name(body: dict[str, Any]) -> str | None:
    model = body.get("model")
    if model is None:
        return None
    if not isinstance(model, str) or not _MODEL_NAME.fullmatch(model):
        raise invalid("`model` must be a model name such as `sonnet`.", "model")
    return model


def reject_params(body: dict[str, Any], names: tuple[str, ...]) -> None:
    for name in names:
        if body.get(name):
            raise GatewayError(
                400,
                f"The SubBridge gateway does not support `{name}`; "
                "it answers with text only.",
                param=name,
                code="unsupported_parameter",
            )


def json_schema_format(value: Any, param: str) -> dict[str, Any] | None:
    """Return the schema of an OpenAI `json_schema` format, or None for text."""
    if value is None or (isinstance(value, dict) and value.get("type") == "text"):
        return None
    if not isinstance(value, dict) or value.get("type") != "json_schema":
        raise invalid(
            f"`{param}` supports only the `text` and `json_schema` types.", param
        )
    holder = value.get("json_schema", value)
    schema = holder.get("schema") if isinstance(holder, dict) else None
    if not isinstance(schema, dict):
        raise invalid(f"`{param}` needs a JSON Schema object in `schema`.", param)
    return schema


def text_turn(role: str, content: Any, param: str) -> list[Turn]:
    if content is None:
        return []
    text = content_text(content, param)
    return [(role, text)] if text else []


def message_turns(messages: Any, roles: dict[str, str], param: str) -> list[Turn]:
    if not isinstance(messages, list) or not messages:
        raise invalid(f"`{param}` must be a non-empty list of messages.", param)
    return [
        _message_turn(message, roles, f"{param}[{index}]")
        for index, message in enumerate(messages)
    ]


def content_text(content: Any, param: str) -> str:
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        raise invalid(f"`{param}` must be a string or a list of text parts.", param)
    return "\n".join(
        _part_text(part, f"{param}[{index}]") for index, part in enumerate(content)
    )


def invalid(message: str, param: str) -> GatewayError:
    return GatewayError(400, message, param=param, code="invalid_request")


def _message_turn(message: Any, roles: dict[str, str], param: str) -> Turn:
    role = message.get("role") if isinstance(message, dict) else None
    if role not in roles:
        allowed = ", ".join(sorted(roles))
        raise invalid(f"`{param}` must be a message with role {allowed}.", param)
    return roles[role], content_text(message.get("content"), f"{param}.content")


def _part_text(part: Any, param: str) -> str:
    if isinstance(part, dict) and part.get("type") in TEXT_PART_TYPES:
        text = part.get("text")
        if isinstance(text, str):
            return text
    kind = part.get("type") if isinstance(part, dict) else type(part).__name__
    raise GatewayError(
        400,
        f"`{param}` is a {kind!r} part; the SubBridge gateway accepts text only.",
        param=param,
        code="unsupported_content",
    )


# Chat Completions messages and Responses input items share these roles.
OPENAI_ROLES = {
    "system": "system",
    "developer": "system",
    "user": "user",
    "assistant": "assistant",
}
