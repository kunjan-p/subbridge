"""JSONL transport shared by the synchronous and asynchronous provider threads."""

from __future__ import annotations

import asyncio
import json
import subprocess
import tempfile
import threading
import time
from collections.abc import AsyncGenerator, Generator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from ._process import (
    ASYNC_STREAM_LINE_LIMIT,
    SYNC_STREAM_LINE_LIMIT,
    StreamLineLimitExceeded,
    SyncProcessIO,
    describe_turn_failure,
    process_error_message,
    process_group_options,
    terminate_async_process_tree,
)
from .errors import (
    ClaudeProcessError,
    ClaudeProtocolError,
    ClaudeTurnError,
    CodexProcessError,
    CodexProtocolError,
    CodexTurnError,
)
from .models import ProviderName


class EventStream:
    """Own one CLI invocation, including its pipes, deadline, and diagnostics."""

    def __init__(
        self,
        provider: ProviderName,
        command: list[str],
        *,
        env: dict[str, str],
        cwd: str | Path | None,
        timeout: float | None,
        include_raw_diagnostics: bool,
    ) -> None:
        self.provider = provider
        self.command = command
        self.env = env
        self.cwd = str(Path(cwd).expanduser()) if cwd else None
        self.timeout = timeout
        self.include_raw_diagnostics = include_raw_diagnostics
        self.terminal = False
        self.last_error: str | None = None
        if provider == "claude":
            self.label = "Claude Code"
            self.process_error = ClaudeProcessError
            self.protocol_error = ClaudeProtocolError
            self.turn_error = ClaudeTurnError
        else:
            self.label = "Codex"
            self.process_error = CodexProcessError
            self.protocol_error = CodexProtocolError
            self.turn_error = CodexTurnError

    def _decode(self, line: str | bytes) -> dict[str, Any]:
        try:
            if isinstance(line, bytes):
                line = line.decode("utf-8")
            event = json.loads(line)
        except (json.JSONDecodeError, UnicodeError) as exc:
            raise self.protocol_error(
                f"{self.label} returned invalid JSONL event data."
            ) from exc
        if not isinstance(event, dict):
            raise self.protocol_error(
                f"{self.label} returned a JSON event that is not an object."
            )
        kind = event.get("type")
        if self.provider == "claude":
            self.terminal = kind == "result"
            if self.terminal and (
                event.get("subtype") not in (None, "success")
                or event.get("is_error") is True
            ):
                self.last_error = str(
                    event.get("result")
                    or event.get("error")
                    or "Claude Code turn failed"
                )
        else:
            self.terminal = kind in ("turn.completed", "turn.failed")
            if kind == "turn.failed":
                error = event.get("error") or {}
                self.last_error = (
                    error.get("message") if isinstance(error, dict) else str(error)
                )
            elif kind == "error":
                self.last_error = str(event.get("message") or "Codex stream error")
        return event

    def _finish(self, return_code: int, stderr: str) -> None:
        if return_code != 0:
            if self.last_error:
                raise self.turn_error(
                    describe_turn_failure(self.label, self.last_error, stderr)
                )
            raise self.process_error(
                process_error_message(
                    self.label,
                    return_code,
                    stderr,
                    include_raw_diagnostics=self.include_raw_diagnostics,
                )
            )
        if self.provider == "claude" and not self.terminal:
            raise self.protocol_error(
                "Claude Code exited without a terminal result event."
            )

    @contextmanager
    def _translate_errors(self, line_limit: int) -> Generator[None, None, None]:
        try:
            yield
        except StreamLineLimitExceeded as exc:
            raise self.protocol_error(
                f"{self.label} returned a JSONL event larger than the {line_limit}-byte safety limit."
            ) from exc
        except (subprocess.TimeoutExpired, TimeoutError) as exc:
            raise self.process_error(
                f"{self.label} timed out after {self.timeout} seconds."
            ) from exc
        except UnicodeError as exc:
            raise self.protocol_error(
                f"{self.label} returned invalid UTF-8 pipe data."
            ) from exc
        except OSError as exc:
            raise self.process_error(
                f"Could not communicate with {self.label}: {exc}"
            ) from exc

    def sync(self, prompt: str) -> Generator[dict[str, Any], None, None]:
        with (
            self._translate_errors(SYNC_STREAM_LINE_LIMIT),
            tempfile.TemporaryFile(mode="w+t", encoding="utf-8") as stderr,
        ):
            process = subprocess.Popen(
                self.command,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=stderr,
                env=self.env,
                cwd=self.cwd,
                **process_group_options(),
            )
            # Exposed so another thread (the gateway's close(), stopping an
            # in-flight turn) can find and terminate this process tree
            # without restructuring this generator-based transport.
            threading.current_thread().subbridge_process = process
            with SyncProcessIO(
                process, prompt, self.timeout, SYNC_STREAM_LINE_LIMIT
            ) as io:
                while line := io.readline():
                    if not line.strip():
                        continue
                    yield self._decode(line)
                    if self.terminal:
                        break
                return_code = process.wait(timeout=io.remaining())
            stderr.seek(0)
            self._finish(return_code, stderr.read())

    async def async_(self, prompt: str) -> AsyncGenerator[dict[str, Any], None]:
        with (
            self._translate_errors(ASYNC_STREAM_LINE_LIMIT),
            tempfile.TemporaryFile(mode="w+t", encoding="utf-8") as stderr,
        ):
            process = await asyncio.create_subprocess_exec(
                *self.command,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=stderr,
                env=self.env,
                cwd=self.cwd,
                limit=ASYNC_STREAM_LINE_LIMIT,
                **process_group_options(),
            )
            deadline = None if self.timeout is None else time.monotonic() + self.timeout
            try:
                await self._send_prompt(process, prompt, deadline)
                async for event in self._read_events(process, deadline):
                    yield event
                await asyncio.wait_for(process.wait(), timeout=_remaining(deadline))
                stderr.seek(0)
                self._finish(process.returncode or 0, stderr.read())
            finally:
                await terminate_async_process_tree(process)

    async def _send_prompt(
        self, process: asyncio.subprocess.Process, prompt: str, deadline: float | None
    ) -> None:
        assert process.stdin is not None
        process.stdin.write(prompt.encode("utf-8"))
        await asyncio.wait_for(process.stdin.drain(), timeout=_remaining(deadline))
        process.stdin.close()

    async def _read_events(
        self, process: asyncio.subprocess.Process, deadline: float | None
    ) -> AsyncGenerator[dict[str, Any], None]:
        assert process.stdout is not None
        while True:
            try:
                line = await asyncio.wait_for(
                    process.stdout.readline(), timeout=_remaining(deadline)
                )
            except ValueError as exc:
                raise StreamLineLimitExceeded() from exc
            if not line:
                return
            yield self._decode(line)
            if self.terminal:
                return


def _remaining(deadline: float | None) -> float | None:
    return None if deadline is None else max(0.0, deadline - time.monotonic())
