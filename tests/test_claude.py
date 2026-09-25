import asyncio
import json
import os
import textwrap
import time
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any
from unittest import mock

from subbridge import ClaudeClient, TurnResult
from subbridge._claude_thread import ClaudeThread
from subbridge.errors import (
    ClaudeProcessError,
    ClaudeProtocolError,
    ClaudeTurnError,
    ClaudeWrongAuthModeError,
)


def fake_claude(
    tmp_path: Path, auth_method: str = "claude.ai", stall_stdin: bool = False
) -> Path:
    cli = tmp_path / "claude"
    cli.write_text(
        textwrap.dedent(
            f"""\
            #!{os.sys.executable}
            import json
            import sys

            args = sys.argv[1:]
            with open(__file__ + ".calls", "a") as log:
                print(json.dumps(args), file=log)
            if args == ["--version"]:
                print("2.1.test")
            elif args == ["auth", "status", "--json"]:
                print(json.dumps({{"loggedIn": True, "authMethod": {auth_method!r}, "subscriptionType": "max", "email": "dev@example.test"}}))
            elif "--output-format" in args:
                if {stall_stdin!r}:
                    import time
                    time.sleep(2)
                    prompt = sys.stdin.read()
                else:
                    prompt = sys.stdin.read()
                if prompt == "crash":
                    print("account=/private/sensitive", file=sys.stderr)
                    sys.exit(7)
                if prompt == "sleep":
                    import time
                    time.sleep(2)
                if prompt == "no-terminal":
                    print(json.dumps({{"type": "assistant", "message": {{"content": [{{"type": "text", "text": "partial"}}]}}}}))
                    sys.exit(0)
                if prompt == "is-error":
                    print(json.dumps({{"type": "result", "subtype": "success", "is_error": True, "result": "API Error: usage limit reached"}}))
                    sys.exit(0)
                if prompt == "background":
                    import subprocess
                    subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
                if prompt == "plan-denied":
                    print(json.dumps({{"type": "result", "subtype": "error_during_execution", "result": "model is not available on your plan"}}))
                    sys.exit(1)
                if prompt == "which-model":
                    prompt = args[args.index("--model") + 1]
                print(json.dumps({{
                    "type": "assistant",
                    "message": {{"content": [{{"type": "text", "text": "x" * 70000 if prompt == "large-event" else "answer: " + prompt}}]}},
                }}))
                result_event = {{
                    "type": "result",
                    "subtype": "success",
                    "result": "answer: " + prompt,
                    "session_id": "session-123",
                    "usage": {{
                        "input_tokens": 12,
                        "cache_read_input_tokens": 3,
                        "cache_creation_input_tokens": 2,
                        "output_tokens": 7,
                    }},
                }}
                if "--json-schema" in args:
                    result_event["structured_output"] = {{"answer": 42}}
                print(json.dumps(result_event))
            else:
                sys.exit(2)
            """
        ),
        encoding="utf-8",
    )
    cli.chmod(0o755)
    return cli


class ClaudeClientTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = TemporaryDirectory()
        self.tmp_path = Path(self.tempdir.name)

    def tearDown(self) -> None:
        self.tempdir.cleanup()

    def test_claude_status_reports_subscription_auth(self) -> None:
        client = ClaudeClient(claude_path=str(fake_claude(self.tmp_path)))
        status = client.status()
        self.assertTrue(status.installed)
        self.assertTrue(status.authenticated)
        self.assertEqual(status.auth_mode, "claude.ai")
        self.assertEqual(status.version, "2.1.test")

    def test_claude_ask_streams_result_and_usage(self) -> None:
        client = ClaudeClient(claude_path=str(fake_claude(self.tmp_path)))
        result = client.ask("hello", model="sonnet", include_events=True)
        self.assertEqual(result.text, "answer: hello")
        self.assertEqual(result.thread_id, "session-123")
        self.assertIsNotNone(result.usage)
        self.assertEqual(result.usage.input_tokens, 12)
        self.assertEqual(result.usage.cached_input_tokens, 3)
        self.assertEqual(result.usage.cache_write_input_tokens, 2)
        self.assertEqual(result.usage.output_tokens, 7)
        self.assertEqual(len(result.events), 2)
        self.assertEqual(result.provider, "claude")
        self.assertEqual(result.model, "sonnet")
        self.assertGreaterEqual(result.elapsed_seconds, 0)

        private_result = client.ask("hello")
        self.assertEqual(private_result.events, [])
        self.assertEqual(private_result.items, [])

    def test_is_error_result_raises_even_with_success_subtype(self) -> None:
        client = ClaudeClient(claude_path=str(fake_claude(self.tmp_path)))
        with self.assertRaisesRegex(ClaudeTurnError, "usage or rate limit") as raised:
            client.ask("is-error")
        self.assertIn("API Error: usage limit reached", str(raised.exception))

    def test_background_child_holding_stdout_does_not_block_turn(self) -> None:
        client = ClaudeClient(claude_path=str(fake_claude(self.tmp_path)))
        started = time.monotonic()
        result = client.ask("background", timeout=20)
        self.assertEqual(result.text, "answer: background")
        self.assertLess(time.monotonic() - started, 10)

    def test_subscription_only_strips_endpoint_override(self) -> None:
        env = {"ANTHROPIC_BASE_URL": "http://proxy.invalid", "PATH": "/bin"}
        self.assertNotIn("ANTHROPIC_BASE_URL", ClaudeClient(env=env)._environment())
        self.assertIn(
            "ANTHROPIC_BASE_URL",
            ClaudeClient(env=env, subscription_only=False)._environment(),
        )

    def test_resume_rejects_flag_like_thread_id(self) -> None:
        client = ClaudeClient(claude_path=str(fake_claude(self.tmp_path)))
        with self.assertRaises(ValueError):
            client.resume_thread("--last")

    def test_status_hides_raw_account_data_unless_requested(self) -> None:
        client = ClaudeClient(claude_path=str(fake_claude(self.tmp_path)))
        self.assertIsNone(client.status().raw_status)
        self.assertIn("dev@example.test", client.status(include_raw=True).raw_status)

    def test_capabilities_report_no_unavailable_model_catalog(self) -> None:
        client = ClaudeClient(claude_path=str(fake_claude(self.tmp_path)))
        capabilities = client.capabilities()
        self.assertEqual(capabilities.provider, "claude")
        self.assertTrue(capabilities.authenticated)
        self.assertIsNone(capabilities.models)
        self.assertFalse(capabilities.model_access_verified)
        self.assertEqual(capabilities.account_plan, "max")
        self.assertIsNone(capabilities.plan_allowed_models)
        self.assertFalse(capabilities.usage_available)

    def test_async_ask_and_cancellation(self) -> None:
        client = ClaudeClient(claude_path=str(fake_claude(self.tmp_path)))
        result = asyncio.run(client.ask_async("hello", model="haiku"))
        self.assertEqual(result.text, "answer: hello")
        self.assertEqual(result.provider, "claude")
        self.assertEqual(result.model, "haiku")
        self.assertEqual(result.events, [])
        large = asyncio.run(client.ask_async("large-event", timeout=5))
        self.assertEqual(large.text, "answer: large-event")

        async def normalized_events() -> list:
            thread = client.start_thread()
            return [event async for event in thread.stream_normalized_async("hello")]

        normalized = asyncio.run(normalized_events())
        self.assertEqual(
            [event.kind for event in normalized], ["message", "turn_completed"]
        )
        self.assertTrue(all(event.raw is None for event in normalized))

        async def cancel_request() -> None:
            thread = client.start_thread()
            task = asyncio.create_task(thread.run_async("sleep", timeout=10))
            await asyncio.sleep(0.05)
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task

        asyncio.run(cancel_request())

    def test_structured_output_and_protocol_failures(self) -> None:
        client = ClaudeClient(claude_path=str(fake_claude(self.tmp_path)))
        result = client.ask("hello", output_schema={"type": "object"})
        self.assertEqual(result.structured_output, {"answer": 42})
        with self.assertRaisesRegex(
            ClaudeTurnError, "unavailable on this account plan"
        ):
            client.ask("plan-denied")
        with self.assertRaisesRegex(Exception, "terminal result"):
            client.ask("no-terminal")

    def test_sync_stream_rejects_oversized_jsonl_event(self) -> None:
        client = ClaudeClient(claude_path=str(fake_claude(self.tmp_path)))
        with (
            mock.patch("subbridge._stream.SYNC_STREAM_LINE_LIMIT", 1024),
            self.assertRaises(ClaudeProtocolError),
        ):
            client.ask("large-event")

    def test_subscription_only_rejects_api_key_auth(self) -> None:
        client = ClaudeClient(
            claude_path=str(fake_claude(self.tmp_path, auth_method="apiKey")),
        )
        with self.assertRaises(ClaudeWrongAuthModeError):
            client.ask("hello")

    def test_build_command_uses_supported_cli_flags(self) -> None:
        client = ClaudeClient(claude_path=str(fake_claude(self.tmp_path)))
        thread = client.start_thread(
            model="sonnet",
            effort="high",
            cwd=self.tmp_path,
            additional_directories=[self.tmp_path / "extra"],
        )
        command = thread._build_command({"type": "object"})
        self.assertIn("--output-format", command)
        self.assertIn("stream-json", command)
        self.assertIn("--model", command)
        self.assertIn("sonnet", command)
        self.assertIn("--effort", command)
        self.assertIn("--permission-mode", command)
        self.assertNotIn("--cwd", command)
        self.assertIn("--json-schema", command)
        self.assertEqual(
            json.loads(command[command.index("--json-schema") + 1]),
            {"type": "object"},
        )

    def test_default_mode_exposes_only_read_tools(self) -> None:
        client = ClaudeClient(claude_path=str(fake_claude(self.tmp_path)))
        command = client.start_thread()._build_command()
        mode = command[command.index("--permission-mode") + 1]
        self.assertEqual(mode, "dontAsk")
        self.assertIn("--tools=Read,Glob,Grep", command)
        self.assertIn("--strict-mcp-config", command)
        self.assertNotIn("plan", command)
        self.assertIn("--setting-sources", command)
        self.assertEqual(command[command.index("--setting-sources") + 1], "user")

    def test_explicit_mode_passes_through_without_tool_limits(self) -> None:
        client = ClaudeClient(claude_path=str(fake_claude(self.tmp_path)))
        for mode in ("plan", "acceptEdits", "dontAsk", "default"):
            command = client.start_thread(permission_mode=mode)._build_command()
            self.assertEqual(command[command.index("--permission-mode") + 1], mode)
            self.assertFalse(any(arg.startswith("--tools") for arg in command))
            self.assertNotIn("--strict-mcp-config", command)
            self.assertNotIn("--setting-sources", command)

    def test_ask_defaults_to_read_only(self) -> None:
        client = ClaudeClient(claude_path=str(fake_claude(self.tmp_path)))
        captured_commands: list[list[str]] = []

        def capture_run(self: ClaudeThread, *args: Any, **kwargs: Any) -> TurnResult:
            captured_commands.append(self._build_command())
            return TurnResult(text="", thread_id=None, usage=None)

        async def capture_run_async(
            self: ClaudeThread, *args: Any, **kwargs: Any
        ) -> TurnResult:
            captured_commands.append(self._build_command())
            return TurnResult(text="", thread_id=None, usage=None)

        with mock.patch.object(
            ClaudeThread, "run", autospec=True, side_effect=capture_run
        ):
            client.ask("hi")

        with mock.patch.object(
            ClaudeThread,
            "run_async",
            autospec=True,
            side_effect=capture_run_async,
        ):
            asyncio.run(client.ask_async("hi"))

        self.assertEqual(len(captured_commands), 2)
        for command in captured_commands:
            self.assertIn("--tools=Read,Glob,Grep", command)

    def test_request_timeout_stops_the_cli(self) -> None:
        client = ClaudeClient(claude_path=str(fake_claude(self.tmp_path)))
        with self.assertRaisesRegex(ClaudeProcessError, "timed out"):
            client.ask("sleep", timeout=0.05)

    def test_large_prompt_delivery_honors_timeout(self) -> None:
        client = ClaudeClient(
            claude_path=str(fake_claude(self.tmp_path, stall_stdin=True))
        )
        with self.assertRaisesRegex(ClaudeProcessError, "timed out"):
            client.ask("x" * (1024 * 1024), timeout=0.05)

    def test_process_diagnostics_are_redacted_by_default(self) -> None:
        client = ClaudeClient(claude_path=str(fake_claude(self.tmp_path)))
        with self.assertRaises(ClaudeProcessError) as raised:
            client.ask("crash")
        self.assertNotIn("/private/sensitive", str(raised.exception))

        verbose_client = ClaudeClient(
            claude_path=str(fake_claude(self.tmp_path)),
            include_raw_diagnostics=True,
        )
        with self.assertRaises(ClaudeProcessError) as verbose_raised:
            verbose_client.ask("crash")
        self.assertIn("/private/sensitive", str(verbose_raised.exception))


if __name__ == "__main__":
    unittest.main()
