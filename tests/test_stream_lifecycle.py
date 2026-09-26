"""Exercise both clients against real pipes without contacting a provider."""

import asyncio
import json
import sys
import threading
import time
import unittest
from dataclasses import asdict
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from subbridge import ClaudeClient, CodexClient
from subbridge._process import SyncProcessIO
from subbridge.errors import (
    ClaudeProcessError,
    ClaudeProtocolError,
    CodexProcessError,
    CodexProtocolError,
)


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
                    client.ask("x" * 1000000, timeout=0.1)
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
                    client.ask("hello", timeout=None)

    def test_invalid_utf8_is_a_protocol_error(self):
        for provider, client_type, _, protocol_error in self.providers:
            with self.subTest(provider=provider):
                client = self.client(
                    provider,
                    client_type,
                    "sys.stdin.read()\nsys.stdout.buffer.write(b'\\xff\\n')",
                )
                with self.assertRaises(protocol_error):
                    client.ask("hello", timeout=2)

    def test_sync_and_async_results_match(self):
        for provider, client_type, _, _ in self.providers:
            events = self.turn_events(provider)
            client = self.client(
                provider,
                client_type,
                "sys.stdin.read()\n"
                + "\n".join(
                    f"print({json.dumps(event)!r}, flush=True)" for event in events
                ),
            )
            for include_events in (False, True):
                with self.subTest(provider=provider, include_events=include_events):
                    options = {"timeout": 3, "include_events": include_events}
                    sync = client.start_thread(model="test-model").run(
                        "hello", **options
                    )
                    async_result = asyncio.run(
                        client.start_thread(model="test-model").run_async(
                            "hello", **options
                        )
                    )
                    self.assertEqual(sync.text, "ok")
                    self.assertEqual(sync.thread_id, "test-session")
                    self.assertEqual(sync.provider, provider)
                    self.assertEqual(sync.model, "test-model")
                    self.assertEqual(sync.usage.input_tokens, 3)
                    self.assertEqual(sync.usage.output_tokens, 5)
                    self.assertEqual(sync.events, events if include_events else [])
                    self.assertEqual(len(sync.items), int(include_events))
                    self.assertEqual(
                        sync.structured_output,
                        {"ok": True} if provider == "claude" else None,
                    )
                    left, right = asdict(sync), asdict(async_result)
                    self.assertGreaterEqual(left.pop("elapsed_seconds"), 0)
                    self.assertGreaterEqual(right.pop("elapsed_seconds"), 0)
                    self.assertEqual(left, right)

    @staticmethod
    def turn_events(provider):
        if provider == "claude":
            return [
                {
                    "type": "assistant",
                    "message": {"content": [{"type": "text", "text": "draft"}]},
                },
                {
                    "type": "result",
                    "subtype": "success",
                    "session_id": "test-session",
                    "result": "ok",
                    "structured_output": {"ok": True},
                    "usage": {"input_tokens": 3, "output_tokens": 5},
                },
            ]
        return [
            {"type": "thread.started", "thread_id": "test-session"},
            {"type": "item.completed", "item": {"type": "agent_message", "text": "ok"}},
            {
                "type": "turn.completed",
                "usage": {"input_tokens": 3, "output_tokens": 5},
            },
        ]

    def test_async_close_reaps_process_and_removes_schema(self):
        async def exercise(client, normalized):
            processes, schema_paths = [], []
            create_process = asyncio.create_subprocess_exec

            async def capture(*args, **kwargs):
                if "--output-schema" in args:
                    schema_paths.append(Path(args[args.index("--output-schema") + 1]))
                process = await create_process(*args, **kwargs)
                processes.append(process)
                return process

            thread = client.start_thread()
            stream_method = (
                thread.stream_normalized_async if normalized else thread.stream_async
            )
            with patch(
                "subbridge._stream.asyncio.create_subprocess_exec", side_effect=capture
            ):
                stream = stream_method(
                    "hello", output_schema={"type": "object"}, timeout=3
                )
                try:
                    await anext(stream)
                    for path in schema_paths:
                        self.assertTrue(path.exists())
                    self.assertIsNone(processes[0].returncode)
                finally:
                    await stream.aclose()
            self.assertIsNotNone(processes[0].returncode)
            for path in schema_paths:
                self.assertFalse(path.exists())

        for provider, client_type, _, _ in self.providers:
            client = self.client(
                provider,
                client_type,
                'sys.stdin.read()\nprint(\'{"type":"event"}\', flush=True)\ntime.sleep(30)',
            )
            for normalized in (False, True):
                with self.subTest(provider=provider, normalized=normalized):
                    asyncio.run(exercise(client, normalized))
