"""Tracks in-flight CLI turns so `close()` can stop them before the workdir goes."""

from __future__ import annotations

import threading
import time
from collections.abc import Generator
from contextlib import contextmanager, suppress
from typing import Any

from .._process import terminate_process_tree
from ._errors import GatewayError

# A turn's `_validate_auth()` (see `_claude_thread.py`/`_codex_thread.py`) runs
# before any CLI is spawned, and by itself can block for up to two 5-second
# subprocess calls -- so a turn `close()` catches mid-auth-check may not have
# a process to kill for several seconds. This deadline must comfortably
# outlast that worst case, or `close()` could give up and return before the
# CLI even starts, letting it run past the deleted workdir.
_STOP_DEADLINE_SECONDS = 15.0
_STOP_POLL_SECONDS = 0.02
# Once `_STOP_DEADLINE_SECONDS` is spent, bound how long we wait for each
# remaining thread to notice its process died and unwind, so `close()` still
# cannot hang forever on a straggler.
_STOP_JOIN_SECONDS = 2.0


class TurnLifecycle:
    """Holds the CLI slot semaphore and tracks which thread holds each slot.

    `cli_slot()` registers the calling thread in `_active_threads` and checks
    `_closing`, both under `_lock`, so a turn queued on `slots.acquire()` when
    `close()` starts can never register (and start a CLI) after
    `stop_in_flight_turns()` has already looked for turns to stop: it either
    registered before `_closing` was set (and the stop loop will see and kill
    it) or it observes `_closing` here and never starts at all.
    """

    def __init__(
        self, slots: threading.BoundedSemaphore, turn_timeout: float | None
    ) -> None:
        self.slots = slots
        self.turn_timeout = turn_timeout
        self._lock = threading.Lock()
        self._active_threads: set[threading.Thread] = set()
        self._closing = False

    @contextmanager
    def cli_slot(self) -> Generator[None, None, None]:
        """Hold one of the max_concurrency CLI slots, waiting up to the timeout."""
        if not self.slots.acquire(timeout=self.turn_timeout):
            raise GatewayError(
                503, f"Every CLI slot stayed busy for {self.turn_timeout} seconds."
            )
        thread = threading.current_thread()
        with self._lock:
            if self._closing:
                self.slots.release()
                raise GatewayError(
                    503,
                    "The SubBridge gateway is shutting down.",
                    code="gateway_closing",
                )
            self._active_threads.add(thread)
        try:
            yield
        finally:
            with self._lock:
                self._active_threads.discard(thread)
            self.slots.release()

    def begin_closing(self) -> None:
        """Refuse any turn not yet registered, before `stop_in_flight_turns()` looks.

        Must complete before `stop_in_flight_turns()` starts: `cli_slot()`
        checks this flag and registers in `_active_threads` in the same
        critical section, so once this is set, no new turn can slip past the
        stop loop and start a CLI of its own.
        """
        with self._lock:
            self._closing = True

    def stop_in_flight_turns(self) -> None:
        """Kill every running turn's CLI, then wait (bounded) for its thread to exit.

        Loops rather than acting on one snapshot: a turn that read `_closing`
        as false just before `begin_closing()` set it can still register
        after this starts, and a turn's `subprocess.Popen` (see
        `_stream.EventStream.sync`) may not exist yet on the first look --
        `_validate_auth()` alone can block for several seconds with no
        process to kill yet. So this keeps polling until `_active_threads` is
        empty, killing any thread's process the instant one appears, for as
        long as `_STOP_DEADLINE_SECONDS` (comfortably longer than that worst
        case), so `close()` cannot hang forever on a CLI that never shows up
        at all. In the common case -- no turns in flight, or a turn whose CLI
        is already running -- this returns almost immediately.
        """
        deadline = time.monotonic() + _STOP_DEADLINE_SECONDS
        killed: set[int] = set()
        while time.monotonic() < deadline:
            with self._lock:
                threads = list(self._active_threads)
            if not threads:
                return
            for thread in threads:
                if id(thread) in killed:
                    continue
                process = getattr(thread, "subbridge_process", None)
                if process is not None:
                    terminate_process_tree(process)
                    killed.add(id(thread))
            time.sleep(_STOP_POLL_SECONDS)
        # Every thread still here either never got a process to kill (stuck
        # somewhere odd) or its process is slow to die even after being
        # killed; join what's left with a short bound each so close() still
        # returns promptly instead of hanging on it.
        with self._lock:
            threads = list(self._active_threads)
        for thread in threads:
            thread.join(timeout=_STOP_JOIN_SECONDS)


_ERROR_DRAIN_CHUNK = 64 * 1024
_ERROR_DRAIN_SECONDS = 1.0


def drain_leftover_body(rfile: Any, connection: Any, remaining: int) -> None:
    """Read up to `remaining` bytes from `rfile`, for a bounded total time.

    Uses `rfile.read1()`, not `rfile.read()`: `read()` loops internally,
    issuing as many `recv()` calls as it takes to fill the requested size,
    and only *each* call (not the loop as a whole) honors the socket
    timeout -- a client trickling in a byte at a time could still hold this
    past the intended budget. `read1()` makes at most one `recv()` call, so
    the deadline is re-checked between every network read.
    """
    if remaining <= 0:
        return
    previous_timeout = connection.gettimeout()
    deadline = time.monotonic() + _ERROR_DRAIN_SECONDS
    with suppress(OSError):
        while remaining > 0:
            budget = deadline - time.monotonic()
            if budget <= 0:
                break
            connection.settimeout(budget)
            chunk = rfile.read1(min(remaining, _ERROR_DRAIN_CHUNK))
            if not chunk:
                break
            remaining -= len(chunk)
    with suppress(OSError):
        connection.settimeout(previous_timeout)
