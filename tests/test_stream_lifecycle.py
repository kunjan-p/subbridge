"""Exercise both clients against real pipes without contacting a provider."""

import json
import subprocess
import sys
import threading
import time
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from subbridge._claude import ClaudeClient
from subbridge._codex import CodexClient
from subbridge._errors import (
    ClaudeProcessError,
    ClaudeProtocolError,
    CodexProcessError,
    CodexProtocolError,
)
from subbridge._process import SyncProcessIO


class StreamLifecycleTests(unittest.TestCase):
    providers = (
        ("claude", ClaudeClient, ClaudeProcessError, ClaudeProtocolError),
        ("codex", CodexClient, CodexProcessError, CodexProtocolError),
    )

    def setUp(self):
        self.directory = TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)

    def client(self, provider, client_type, body):
        cli = Path(self.directory.name) / provider
        cli.write_text(f"#!{sys.executable}\nimport sys, json, time\n{body}\n")
        cli.chmod(0o755)
        client = client_type(**{f"{provider}_path": str(cli)})
        auth = patch.object(client, "_validate_auth")
        auth.start()
        self.addCleanup(auth.stop)
        return client

    def test_closing_full_stream_stops_reader_and_writer(self):
        for provider, client_type, _, _ in self.providers:
            with self.subTest(provider=provider):
                client = self.client(
                    provider,
                    client_type,
                    "sys.stdin.read()\n"
                    'while True: print(\'{"type":"event"}\', flush=True)',
                )
                pumps = []

                def capture(*args, pumps=pumps):
                    io = SyncProcessIO(*args)
                    pumps.append(io)
                    return io

                with patch("subbridge._stream.SyncProcessIO", side_effect=capture):
                    stream = client.start_thread().stream("hello", timeout=3)
                    try:
                        next(stream)
                        io = pumps[0]
                        deadline = time.monotonic() + 2
                        while not io.lines.full() and time.monotonic() < deadline:
                            time.sleep(0.01)
                        self.assertTrue(io.lines.full())
                    finally:
                        stream.close()
                self.assertFalse(io.reader.is_alive())
                self.assertFalse(io.writer.is_alive())
                self.assertIsNotNone(io.process.poll())
                self.assertTrue(io.process.stdout.closed)
                self.assertTrue(io.process.stdin.closed)

    def test_output_is_drained_while_large_prompt_is_sent(self):
        for provider, client_type, _, _ in self.providers:
            with self.subTest(provider=provider):
                terminal = (
                    {"type": "result", "subtype": "success", "result": "ok"}
                    if provider == "claude"
                    else {"type": "turn.completed"}
                )
                client = self.client(
                    provider,
                    client_type,
                    "print(json.dumps({'type':'event', 'text':'x'*1000000}), flush=True)\n"
                    "assert len(sys.stdin.read()) == 1000000\n"
                    f"print({json.dumps(terminal)!r}, flush=True)",
                )
                events = list(client.start_thread().stream("x" * 1000000, timeout=5))
                self.assertEqual(events[-1]["type"], terminal["type"])

    def test_subbridge_process_attribute_is_restored_not_cleared(self):
        """A thread's `subbridge_process` marker (see
        `_lifecycle.stop_in_flight_turns`) might already be set to something
        else before `sync()` runs on it; `sync()` must restore that value
        once its own process is done, not blank it to `None`.
        """
        for provider, client_type, _, _ in self.providers:
            with self.subTest(provider=provider):
                terminal = (
                    {"type": "result", "subtype": "success", "result": "ok"}
                    if provider == "claude"
                    else {"type": "turn.completed"}
                )
                client = self.client(
                    provider,
                    client_type,
                    "sys.stdin.read()\n"
                    + f"print({json.dumps(terminal)!r}, flush=True)",
                )
                sentinel = object()
                thread = threading.current_thread()
                thread.subbridge_process = sentinel
                try:
                    client.start_thread().run("hello", timeout=3)
                    self.assertIs(thread.subbridge_process, sentinel)
                finally:
                    del thread.subbridge_process

    def test_stalled_input_obeys_deadline(self):
        for provider, client_type, process_error, _ in self.providers:
            with self.subTest(provider=provider):
                client = self.client(provider, client_type, "time.sleep(30)")
                started = time.monotonic()
                with self.assertRaisesRegex(process_error, "timed out"):
                    client.start_thread().run("x" * 1000000, timeout=0.1)
                self.assertLess(time.monotonic() - started, 3)

    def test_limit_counts_utf8_bytes_even_without_timeout(self):
        for provider, client_type, _, protocol_error in self.providers:
            with self.subTest(provider=provider):
                client = self.client(
                    provider,
                    client_type,
                    "sys.stdin.read()\n"
                    "print(json.dumps({'type':'event', 'text':'é'*100}, ensure_ascii=False))",
                )
                with (
                    patch("subbridge._stream.SYNC_STREAM_LINE_LIMIT", 180),
                    self.assertRaisesRegex(protocol_error, "larger than"),
                ):
                    client.start_thread().run("hello", timeout=None)

    def test_invalid_utf8_is_a_protocol_error(self):
        for provider, client_type, _, protocol_error in self.providers:
            with self.subTest(provider=provider):
                client = self.client(
                    provider,
                    client_type,
                    "sys.stdin.read()\nsys.stdout.buffer.write(b'\\xff\\n')",
                )
                with self.assertRaises(protocol_error):
                    client.start_thread().run("hello", timeout=2)

    def test_close_reaps_process_and_removes_schema(self):
        """The gateway relies on this: a caller disconnecting mid-stream for
        a json_schema request must still kill the CLI and clean up the
        schema tempfile, not leak either.
        """
        client = self.client(
            "codex",
            CodexClient,
            'sys.stdin.read()\nprint(\'{"type":"event"}\', flush=True)\ntime.sleep(30)',
        )
        captured = {}
        real_popen = subprocess.Popen

        def capture(*args, **kwargs):
            process = real_popen(*args, **kwargs)
            captured["argv"] = args[0]
            captured["process"] = process
            return process

        with patch("subbridge._stream.subprocess.Popen", side_effect=capture):
            stream = client.start_thread().stream(
                "hello", output_schema={"type": "object"}, timeout=3
            )
            try:
                next(stream)
                argv = captured["argv"]
                schema_path = Path(argv[argv.index("--output-schema") + 1])
                self.assertTrue(schema_path.exists())
                self.assertIsNone(captured["process"].poll())
            finally:
                stream.close()
        self.assertIsNotNone(captured["process"].poll())
        self.assertFalse(schema_path.exists())
        self.assertFalse(schema_path.parent.exists())
