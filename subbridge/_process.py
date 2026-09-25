"""Cross-platform subprocess lifecycle helpers shared by provider clients."""

from __future__ import annotations

import asyncio
import os
import signal
import subprocess
import time
from contextlib import suppress
from queue import Empty, Full, Queue
from threading import Event, Thread
from typing import Any, Self

ASYNC_STREAM_LINE_LIMIT = 16 * 1024 * 1024
SYNC_STREAM_LINE_LIMIT = 16 * 1024 * 1024


class StreamLineLimitExceeded(ValueError):
    """Raised when a provider emits one JSONL event above the safety limit."""


class SyncProcessIO:
    """Pump stdin and bounded stdout concurrently under one request deadline."""

    def __init__(
        self,
        process: subprocess.Popen[bytes],
        prompt: str,
        timeout: float | None,
        line_limit: int,
    ) -> None:
        self.process = process
        self.prompt = prompt
        self.timeout = timeout
        self.line_limit = line_limit
        self.deadline = None if timeout is None else time.monotonic() + timeout
        self.lines: Queue[bytes | Exception | None] = Queue(maxsize=2)
        self.stopped = Event()
        self.reader = Thread(target=self._read, name="subbridge-stdout", daemon=True)
        self.writer = Thread(target=self._write, name="subbridge-stdin", daemon=True)

    def __enter__(self) -> Self:
        self.reader.start()
        self.writer.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self.stopped.set()
        terminate_process_tree(self.process)
        for thread, pipe in (
            (self.reader, self.process.stdout),
            (self.writer, self.process.stdin),
        ):
            thread.join(timeout=5)
            if not thread.is_alive() and pipe is not None:
                pipe.close()

    def _put(self, value: bytes | Exception | None) -> None:
        while not self.stopped.is_set():
            try:
                self.lines.put(value, timeout=0.05)
                return
            except Full:
                continue

    def _read(self) -> None:
        assert self.process.stdout is not None
        try:
            while not self.stopped.is_set():
                line = self.process.stdout.readline(self.line_limit + 1)
                if not line:
                    break
                if len(line) > self.line_limit:
                    raise StreamLineLimitExceeded()
                self._put(line)
        except (OSError, StreamLineLimitExceeded) as exc:
            self._put(exc)
        finally:
            self._put(None)

    def _write(self) -> None:
        assert self.process.stdin is not None
        try:
            self.process.stdin.write(self.prompt.encode("utf-8"))
            self.process.stdin.close()
        except (OSError, UnicodeError) as exc:
            self._put(exc)

    def remaining(self) -> float | None:
        return (
            None
            if self.deadline is None
            else max(0.0, self.deadline - time.monotonic())
        )

    def readline(self) -> str:
        remaining = self.remaining()
        if remaining == 0:
            raise subprocess.TimeoutExpired(self.process.args, self.timeout)
        try:
            line = self.lines.get(timeout=remaining)
        except Empty as exc:
            raise subprocess.TimeoutExpired(self.process.args, self.timeout) from exc
        if isinstance(line, Exception):
            raise line
        return "" if line is None else line.decode("utf-8")


_FAILURE_TERMS = (
    (
        "plan",
        (
            "not available on your plan",
            "not included in your plan",
            "plan does not include",
            "doesn't include access",
            "not entitled",
            "not supported when using codex with a chatgpt account",
            "not available for your account",
        ),
    ),
    (
        "model",
        (
            "model_not_found",
            "unknown model",
            "invalid model",
            "model not found",
            "unrecognized model",
        ),
    ),
    (
        "rate_limit",
        (
            "usage limit",
            "rate limit",
            "quota exceeded",
            "capacity limit",
            "limit reached",
        ),
    ),
)

_FAILURE_HINTS = {
    "plan": "{provider} rejected the requested model because it appears unavailable on this account plan. Check the provider's model picker or run `subbridge doctor`; plan entitlements cannot always be queried by the CLI.",
    "model": "{provider} rejected the requested model name. It may be misspelled, unavailable in this CLI version, or unavailable to this account; run `subbridge doctor` to inspect CLI-known models.",
    "rate_limit": "{provider} reported a usage or rate limit. Check the provider account's usage page; detailed quota data is not exposed consistently by the CLI.",
}


def failure_kind(details: str) -> str | None:
    """Return "plan", "model", or "rate_limit" when CLI output names that failure."""
    lowered = details.lower()
    for kind, terms in _FAILURE_TERMS:
        if any(term in lowered for term in terms):
            return kind
    return None


def classify_provider_failure(provider: str, details: str) -> str | None:
    kind = failure_kind(details)
    return None if kind is None else _FAILURE_HINTS[kind].format(provider=provider)


def describe_turn_failure(provider: str, error: str, context: str = "") -> str:
    """Pair SubBridge's hint with the CLI's own message, which carries details
    such as when a usage limit resets. ``context`` (stderr) only aids
    classification; it is never shown, keeping raw diagnostics opt-in."""
    hint = classify_provider_failure(provider, f"{error} {context}")
    return f"{hint}\n\n{provider} said: {error}" if hint else error


def validate_thread_id(thread_id: str) -> str:
    """Reject IDs the CLI would parse as flags (for example ``--last``)."""
    if not isinstance(thread_id, str) or not thread_id or thread_id.startswith("-"):
        raise ValueError(f"Invalid thread ID: {thread_id!r}")
    return thread_id


def process_group_options() -> dict[str, Any]:
    """Start a separate POSIX process group so cancellation can stop children."""
    if os.name == "posix":
        return {"start_new_session": True}
    if os.name == "nt":
        return {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP}
    return {}


def terminate_process_tree(process: subprocess.Popen[Any]) -> None:
    # On POSIX, signal the group even if the CLI already exited: a background
    # child it spawned can outlive it and keep stdout open.
    if os.name == "posix":
        with suppress(ProcessLookupError, PermissionError):
            os.killpg(process.pid, signal.SIGKILL)
    elif process.poll() is not None:
        return
    elif os.name == "nt":
        with suppress(OSError, subprocess.SubprocessError):
            subprocess.run(
                ["taskkill", "/PID", str(process.pid), "/T", "/F"],
                capture_output=True,
                check=False,
                timeout=5,
            )
    if process.poll() is None:
        with suppress(OSError):
            process.kill()
    with suppress(OSError):
        process.wait(timeout=5)


async def terminate_async_process_tree(process: asyncio.subprocess.Process) -> None:
    if os.name == "posix":
        with suppress(ProcessLookupError, PermissionError):
            os.killpg(process.pid, signal.SIGKILL)
    elif process.returncode is not None:
        return
    elif os.name == "nt":
        with suppress(OSError, subprocess.SubprocessError):
            await asyncio.to_thread(
                subprocess.run,
                ["taskkill", "/PID", str(process.pid), "/T", "/F"],
                capture_output=True,
                check=False,
                timeout=5,
            )
    if process.returncode is None:
        with suppress(ProcessLookupError):
            process.kill()
    with suppress(ProcessLookupError):
        await process.wait()


def process_error_message(
    provider: str,
    return_code: int,
    stderr: str,
    *,
    include_raw_diagnostics: bool,
) -> str:
    details = stderr.strip()
    message = (
        classify_provider_failure(provider, details)
        or f"{provider} CLI exited with code {return_code}."
    )
    if include_raw_diagnostics and details:
        message += f"\n\nCLI diagnostics (may contain sensitive paths or account data):\n{details}"
    return message
