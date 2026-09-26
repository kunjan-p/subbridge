"""JSONL transport shared by provider threads over one CLI invocation."""

from __future__ import annotations

import json
import subprocess
import tempfile
import threading
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from ._errors import (
    ClaudeProcessError,
    ClaudeProtocolError,
    ClaudeTurnError,
    CodexProcessError,
    CodexProtocolError,
    CodexTurnError,
)
from ._models import ProviderName
from ._process import (
    SYNC_STREAM_LINE_LIMIT,
    StreamLineLimitExceeded,
    SyncProcessIO,
    describe_turn_failure,
    process_error_message,
    process_group_options,
)


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
            # without restructuring this generator-based transport. Restored
            # to whatever it was before (rather than blanked to None) once
            # this process is done with, so a reused thread (or a thread
            # outside the gateway) never points at a reaped -- possibly
            # recycled -- PID, but nesting also doesn't lose an outer call's
            # own value.
            this_thread = threading.current_thread()
            previous_process = getattr(this_thread, "subbridge_process", None)
            this_thread.subbridge_process = process
            try:
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
            finally:
                this_thread.subbridge_process = previous_process
