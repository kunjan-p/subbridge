"""close() racing an in-flight (or still-authenticating) CLI turn."""

import importlib.util
import os
import threading
import time
import unittest
from contextlib import suppress
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

# These tests need the (dev-only) anthropic SDK; the release workflow
# installs only requirements-release.txt, so skip cleanly there instead of
# failing to collect this module.
if importlib.util.find_spec("anthropic") is None:
    raise unittest.SkipTest("anthropic is not installed; skipping gateway tests.")

import anthropic
from gateway_support import auth_started_marker, fake_claude_that_stalls
from test_claude import fake_claude

import subbridge


class CloseDuringTurnTests(unittest.TestCase):
    def test_close_stops_an_in_flight_turn_and_returns_quickly(self) -> None:
        with TemporaryDirectory() as tempdir:
            bin_dir = Path(tempdir)
            pid_file = bin_dir / "claude.pid"
            fake_claude_that_stalls(bin_dir, pid_file, sleep_seconds=20)
            path = f"{bin_dir}{os.pathsep}{os.environ.get('PATH', '')}"
            with mock.patch.dict(os.environ, {"PATH": path}):
                gateway = subbridge.serve()
                self.addCleanup(gateway.close)  # close() is idempotent
                client = anthropic.Anthropic(
                    base_url=gateway.anthropic_base_url,
                    api_key=gateway.api_key,
                    max_retries=0,
                )

                def ask() -> None:
                    # Any error is fine here; only close()'s effects matter.
                    with suppress(Exception):
                        client.messages.create(
                            model="sonnet",
                            max_tokens=10,
                            messages=[{"role": "user", "content": "hi"}],
                        )

                worker = threading.Thread(target=ask)
                worker.start()
                deadline = time.monotonic() + 5
                while not pid_file.exists() and time.monotonic() < deadline:
                    time.sleep(0.02)
                self.assertTrue(pid_file.exists(), "fake claude never started")
                pid = int(pid_file.read_text())

                started = time.monotonic()
                gateway.close()
                self.assertLess(time.monotonic() - started, 5)

                deadline = time.monotonic() + 5
                while time.monotonic() < deadline:
                    try:
                        os.kill(pid, 0)
                    except ProcessLookupError:
                        break
                    time.sleep(0.05)
                else:
                    self.fail(f"fake claude (pid {pid}) is still running")

                worker.join(timeout=5)
                self.assertFalse(worker.is_alive())

    @unittest.skipUnless(
        os.name == "posix", "asserting on the SIGKILL exit code is POSIX-only"
    )
    def test_close_during_auth_check_kills_the_cli_once_it_starts(self) -> None:
        """A turn's CLI can still be inside `_validate_auth()` (see
        `_claude_thread.stream()`), with no process yet for `close()` to
        find, when `close()` starts (round 2 fix for C1). `close()` must
        keep waiting -- bounded -- rather than give up and let that turn
        spawn its CLI after the workdir is gone.

        The CLI is killed so promptly once it appears that it can die before
        it gets to write its own pid file (a *good* thing -- see C1), so
        this asserts on the turn's own SIGKILL exit code instead of a pid:
        that only appears once `terminate_process_tree()` has found, killed,
        and reaped an actual process, which is stronger proof that a real
        CLI existed and is now dead than a self-reported pid would be.
        """
        with TemporaryDirectory() as tempdir:
            bin_dir = Path(tempdir)
            pid_file = bin_dir / "claude.pid"
            fake_claude_that_stalls(
                bin_dir, pid_file, sleep_seconds=20, auth_sleep_seconds=3
            )
            marker = auth_started_marker(pid_file)
            path = f"{bin_dir}{os.pathsep}{os.environ.get('PATH', '')}"
            with mock.patch.dict(os.environ, {"PATH": path}):
                gateway = subbridge.serve()
                self.addCleanup(gateway.close)  # close() is idempotent
                client = anthropic.Anthropic(
                    base_url=gateway.anthropic_base_url,
                    api_key=gateway.api_key,
                    max_retries=0,
                )
                result: dict[str, object] = {}

                def ask() -> None:
                    try:
                        client.messages.create(
                            model="sonnet",
                            max_tokens=10,
                            messages=[{"role": "user", "content": "hi"}],
                        )
                        result["status"] = 200
                    except anthropic.AnthropicError as exc:
                        result["status"] = getattr(exc, "status_code", None)
                        result["message"] = str(exc)

                worker = threading.Thread(target=ask)
                worker.start()
                # Wait for the fake auth check to actually start (not just
                # guess a delay), so close() deterministically starts while
                # it is still sleeping and there is no process yet to kill.
                deadline = time.monotonic() + 5
                while not marker.exists() and time.monotonic() < deadline:
                    time.sleep(0.02)
                self.assertTrue(marker.exists(), "auth check never started")

                started = time.monotonic()
                gateway.close()
                elapsed = time.monotonic() - started
                # Long enough to have waited out most of the 3-second auth
                # sleep (proving close() did not give up early), short of
                # the 15-second deadline (proving it did not just hang).
                self.assertGreater(elapsed, 2)
                self.assertLess(elapsed, 10)

                worker.join(timeout=10)
                self.assertFalse(worker.is_alive())
                self.assertNotEqual(result.get("status"), 200)
                self.assertIn("exited with code -9", result.get("message", ""))

    def test_close_refuses_a_queued_turn_instead_of_starting_it(self) -> None:
        with TemporaryDirectory() as tempdir:
            bin_dir = Path(tempdir)
            pid_file = bin_dir / "claude.pid"
            fake_claude_that_stalls(bin_dir, pid_file, sleep_seconds=20)
            path = f"{bin_dir}{os.pathsep}{os.environ.get('PATH', '')}"
            with mock.patch.dict(os.environ, {"PATH": path}):
                gateway = subbridge.serve(max_concurrency=1)
                self.addCleanup(gateway.close)  # close() is idempotent
                client = anthropic.Anthropic(
                    base_url=gateway.anthropic_base_url,
                    api_key=gateway.api_key,
                    max_retries=0,
                )
                statuses: list[int] = []

                def ask(content: str) -> None:
                    try:
                        client.messages.create(
                            model="sonnet",
                            max_tokens=10,
                            messages=[{"role": "user", "content": content}],
                        )
                        statuses.append(200)
                    except anthropic.APIStatusError as exc:
                        statuses.append(exc.status_code)

                first = threading.Thread(target=ask, args=("first",))
                first.start()
                deadline = time.monotonic() + 5
                while not pid_file.exists() and time.monotonic() < deadline:
                    time.sleep(0.02)
                self.assertTrue(pid_file.exists(), "fake claude never started")
                pid = int(pid_file.read_text())

                # Queue a second turn behind the sole slot (max_concurrency=1)
                # so it is blocked in cli_slot()'s slots.acquire() when
                # close() runs -- the race close() must not lose.
                second = threading.Thread(target=ask, args=("second",))
                second.start()
                time.sleep(0.3)

                started = time.monotonic()
                gateway.close()
                self.assertLess(time.monotonic() - started, 5)

                deadline = time.monotonic() + 5
                while time.monotonic() < deadline:
                    try:
                        os.kill(pid, 0)
                    except ProcessLookupError:
                        break
                    time.sleep(0.05)
                else:
                    self.fail(f"fake claude (pid {pid}) is still running")

                first.join(timeout=5)
                second.join(timeout=5)
                self.assertFalse(first.is_alive())
                self.assertFalse(second.is_alive())
                # The queued turn must have been refused, not started: the
                # pid file still names only the first turn's process (a
                # second CLI run would have overwritten it with a new pid).
                self.assertIn(503, statuses)
                self.assertEqual(int(pid_file.read_text()), pid)

    def test_close_stops_an_in_flight_streaming_turn(self) -> None:
        """A streaming turn also registers in `_active_threads` via
        `cli_slot()` (see `GatewayHandler._stream`), so `close()` must be
        able to find and kill its CLI too, not just a non-streaming one.
        """
        with TemporaryDirectory() as tempdir:
            bin_dir = Path(tempdir)
            fake_claude(bin_dir)  # provides the "endless" prompt and its .pid file
            pid_file = bin_dir / "claude.pid"
            path = f"{bin_dir}{os.pathsep}{os.environ.get('PATH', '')}"
            with mock.patch.dict(os.environ, {"PATH": path}):
                gateway = subbridge.serve()
                self.addCleanup(gateway.close)  # close() is idempotent
                client = anthropic.Anthropic(
                    base_url=gateway.anthropic_base_url,
                    api_key=gateway.api_key,
                    max_retries=0,
                )
                got_first_tick = threading.Event()

                def ask() -> None:
                    # Any error is fine here; only close()'s effects matter.
                    with (
                        suppress(Exception),
                        client.messages.stream(
                            model="sonnet",
                            max_tokens=10,
                            messages=[{"role": "user", "content": "endless"}],
                        ) as stream,
                    ):
                        for _ in stream.text_stream:
                            got_first_tick.set()

                # daemon=True: if the assertions below fail and this thread
                # is (unexpectedly) still blocked reading the stream, it
                # must not hang pytest's own process exit.
                worker = threading.Thread(target=ask, daemon=True)
                worker.start()
                self.assertTrue(
                    got_first_tick.wait(timeout=5), "no streamed text arrived"
                )
                deadline = time.monotonic() + 5
                while not pid_file.exists() and time.monotonic() < deadline:
                    time.sleep(0.02)
                self.assertTrue(pid_file.exists(), "fake claude never started")
                pid = int(pid_file.read_text())

                started = time.monotonic()
                gateway.close()
                self.assertLess(time.monotonic() - started, 5)

                deadline = time.monotonic() + 5
                while time.monotonic() < deadline:
                    try:
                        os.kill(pid, 0)
                    except ProcessLookupError:
                        break
                    time.sleep(0.05)
                else:
                    self.fail(f"fake claude (pid {pid}) is still running")

                worker.join(timeout=5)
                self.assertFalse(worker.is_alive())


if __name__ == "__main__":
    unittest.main()
