"""Small normalization layer for the JSONL events emitted by provider CLIs."""

from __future__ import annotations

from typing import Any, Literal

from .models import ProviderName, StreamEvent, Usage


def normalize_event(
    provider: ProviderName,
    event: dict[str, Any],
    *,
    include_raw: bool = False,
    thread_id: str | None = None,
) -> StreamEvent:
    """Normalize one raw CLI event without retaining its payload by default.

    Raw events may contain prompts, file paths, or tool inputs. Set
    ``include_raw=True`` only when the caller needs provider-specific details.
    """
    if provider == "claude":
        kind, event_type, text, usage, event_thread_id = _normalize_claude(event)
    else:
        kind, event_type, text, usage, event_thread_id = _normalize_codex(event)
    return StreamEvent(
        provider=provider,
        kind=kind,
        provider_event=event_type,
        thread_id=thread_id or event_thread_id,
        text=text,
        usage=usage,
        raw=event if include_raw else None,
    )


def _normalize_claude(
    event: dict[str, Any],
) -> tuple[
    Literal["thread_started", "message", "turn_completed", "turn_error", "event"],
    str,
    str | None,
    Usage | None,
    str | None,
]:
    event_type = str(event.get("type", "unknown"))
    thread_id = event.get("session_id")
    if event_type == "assistant":
        message = _object(event.get("message"))
        content = message.get("content")
        content = content if isinstance(content, list) else []
        text = "".join(
            block.get("text", "")
            for block in content
            if isinstance(block, dict) and block.get("type") == "text"
        )
        return (
            "message" if text else "event",
            event_type,
            text or None,
            None,
            thread_id,
        )
    if event_type == "result":
        raw_usage = _object(event.get("usage"))
        usage = Usage(
            input_tokens=raw_usage.get("input_tokens", 0),
            cached_input_tokens=raw_usage.get("cache_read_input_tokens", 0),
            cache_write_input_tokens=raw_usage.get("cache_creation_input_tokens", 0),
            output_tokens=raw_usage.get("output_tokens", 0),
        )
        success = (
            event.get("subtype") in (None, "success")
            and event.get("is_error") is not True
        )
        text = event.get("result")
        return (
            "turn_completed" if success else "turn_error",
            event_type,
            text,
            usage,
            thread_id,
        )
    return ("event", event_type, None, None, thread_id)


def _normalize_codex(
    event: dict[str, Any],
) -> tuple[
    Literal["thread_started", "message", "turn_completed", "turn_error", "event"],
    str,
    str | None,
    Usage | None,
    str | None,
]:
    event_type = str(event.get("type", "unknown"))
    thread_id = event.get("thread_id")
    if event_type == "thread.started":
        return ("thread_started", event_type, None, None, thread_id)
    if event_type == "item.completed":
        item = _object(event.get("item"))
        text = item.get("text") if item.get("type") == "agent_message" else None
        return (
            "message" if text is not None else "event",
            event_type,
            text,
            None,
            thread_id,
        )
    if event_type == "turn.completed":
        raw_usage = _object(event.get("usage"))
        usage = Usage(
            input_tokens=raw_usage.get("input_tokens", 0),
            cached_input_tokens=raw_usage.get("cached_input_tokens", 0),
            cache_write_input_tokens=raw_usage.get("cache_write_input_tokens", 0),
            output_tokens=raw_usage.get("output_tokens", 0),
            reasoning_output_tokens=raw_usage.get("reasoning_output_tokens", 0),
        )
        return ("turn_completed", event_type, None, usage, thread_id)
    if event_type in {"turn.failed", "error"}:
        error = event.get("error") or {}
        message = (
            error.get("message") if isinstance(error, dict) else event.get("message")
        )
        return (
            "turn_error",
            event_type,
            message or event.get("message"),
            None,
            thread_id,
        )
    return ("event", event_type, None, None, thread_id)


def _object(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}
