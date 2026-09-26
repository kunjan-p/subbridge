import json
import os
import textwrap
import time
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

from subbridge._claude import ClaudeClient
from subbridge._errors import (
    ClaudeProcessError,
    ClaudeProtocolError,
    ClaudeTurnError,
    ClaudeWrongAuthModeError,
)
from subbridge._models import TurnResult


def run_turn(
    client: ClaudeClient,
    prompt: str,
    *,
    model: str | None = None,
    timeout: float | None = 300,
    output_schema: dict | None = None,
) -> TurnResult:
    """Exercise a turn the way the gateway does: a fresh thread, then run()."""
    return client.start_thread(model=model).run(
        prompt, output_schema=output_schema, timeout=timeout
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
                    import os, time
                    with open(__file__ + ".pid", "w") as pid_file:
                        pid_file.write(str(os.getpid()))
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
                if prompt == "partial-then-error":
                    print(json.dumps({{"type": "assistant", "message": {{"content": [{{"type": "text", "text": "partial"}}]}}}}), flush=True)
                    print(json.dumps({{"type": "result", "subtype": "success", "is_error": True, "result": "API Error: usage limit reached"}}))
                    sys.exit(0)
                if prompt == "endless":
                    import os, time
                    with open(__file__ + ".pid", "w") as pid_file:
                        pid_file.write(str(os.getpid()))
                    while True:
                        print(json.dumps({{"type": "assistant", "message": {{"content": [{{"type": "text", "text": "tick "}}]}}}}), flush=True)
                        time.sleep(0.05)
                if prompt == "two-messages":
                    print(json.dumps({{"type": "assistant", "message": {{"content": [{{"type": "text", "text": "Let me check the file."}}]}}}}), flush=True)
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

    def test_run_streams_result_and_usage(self) -> None:
        client = ClaudeClient(claude_path=str(fake_claude(self.tmp_path)))
        result = run_turn(client, "hello", model="sonnet")
        self.assertEqual(result.text, "answer: hello")
        self.assertEqual(result.thread_id, "session-123")
        self.assertIsNotNone(result.usage)
        self.assertEqual(result.usage.input_tokens, 12)
        self.assertEqual(result.usage.cached_input_tokens, 3)
        self.assertEqual(result.usage.cache_write_input_tokens, 2)
        self.assertEqual(result.usage.output_tokens, 7)
        self.assertEqual(result.provider, "claude")
        self.assertEqual(result.model, "sonnet")
        self.assertGreaterEqual(result.elapsed_seconds, 0)

    def test_is_error_result_raises_even_with_success_subtype(self) -> None:
        client = ClaudeClient(claude_path=str(fake_claude(self.tmp_path)))
        with self.assertRaisesRegex(ClaudeTurnError, "usage or rate limit") as raised:
            run_turn(client, "is-error")
        self.assertIn("API Error: usage limit reached", str(raised.exception))

    def test_background_child_holding_stdout_does_not_block_turn(self) -> None:
        client = ClaudeClient(claude_path=str(fake_claude(self.tmp_path)))
        started = time.monotonic()
        result = run_turn(client, "background", timeout=20)
        self.assertEqual(result.text, "answer: background")
        self.assertLess(time.monotonic() - started, 10)

    def test_subscription_only_strips_endpoint_override(self) -> None:
        env = {"ANTHROPIC_BASE_URL": "http://proxy.invalid", "PATH": "/bin"}
        self.assertNotIn("ANTHROPIC_BASE_URL", ClaudeClient(env=env)._environment())
        self.assertIn(
            "ANTHROPIC_BASE_URL",
            ClaudeClient(env=env, subscription_only=False)._environment(),
        )

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

    def test_structured_output_and_protocol_failures(self) -> None:
        client = ClaudeClient(claude_path=str(fake_claude(self.tmp_path)))
        result = run_turn(client, "hello", output_schema={"type": "object"})
        self.assertEqual(result.structured_output, {"answer": 42})
        with self.assertRaisesRegex(
            ClaudeTurnError, "unavailable on this account plan"
        ):
            run_turn(client, "plan-denied")
        with self.assertRaisesRegex(Exception, "terminal result"):
            run_turn(client, "no-terminal")

    def test_sync_stream_rejects_oversized_jsonl_event(self) -> None:
        client = ClaudeClient(claude_path=str(fake_claude(self.tmp_path)))
        with (
            mock.patch("subbridge._stream.SYNC_STREAM_LINE_LIMIT", 1024),
            self.assertRaises(ClaudeProtocolError),
        ):
            run_turn(client, "large-event")

    def test_subscription_only_rejects_api_key_auth(self) -> None:
        client = ClaudeClient(
            claude_path=str(fake_claude(self.tmp_path, auth_method="apiKey")),
        )
        with self.assertRaises(ClaudeWrongAuthModeError):
            run_turn(client, "hello")

    def test_build_command_uses_supported_cli_flags(self) -> None:
        client = ClaudeClient(claude_path=str(fake_claude(self.tmp_path)))
        thread = client.start_thread(model="sonnet", cwd=self.tmp_path)
        command = thread._build_command({"type": "object"})
        self.assertIn("--output-format", command)
        self.assertIn("stream-json", command)
        self.assertIn("--model", command)
        self.assertIn("sonnet", command)
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

    def test_request_timeout_stops_the_cli(self) -> None:
        client = ClaudeClient(claude_path=str(fake_claude(self.tmp_path)))
        # A slightly more generous timeout than the other timeout tests here,
        # so the fake CLI reliably reaches the line that records its own PID
        # before this kills it; it is still far below the 2-second sleep.
        with self.assertRaisesRegex(ClaudeProcessError, "timed out"):
            run_turn(client, "sleep", timeout=0.3)
        pid = int((self.tmp_path / "claude.pid").read_text())
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            try:
                os.kill(pid, 0)
            except ProcessLookupError:
                return
            time.sleep(0.05)
        self.fail(f"fake claude (pid {pid}) is still running")

    def test_large_prompt_delivery_honors_timeout(self) -> None:
        client = ClaudeClient(
            claude_path=str(fake_claude(self.tmp_path, stall_stdin=True))
        )
        with self.assertRaisesRegex(ClaudeProcessError, "timed out"):
            run_turn(client, "x" * (1024 * 1024), timeout=0.05)

    def test_process_diagnostics_are_redacted_by_default(self) -> None:
        client = ClaudeClient(claude_path=str(fake_claude(self.tmp_path)))
        with self.assertRaises(ClaudeProcessError) as raised:
            run_turn(client, "crash")
        self.assertNotIn("/private/sensitive", str(raised.exception))

        verbose_client = ClaudeClient(
            claude_path=str(fake_claude(self.tmp_path)),
            include_raw_diagnostics=True,
        )
        with self.assertRaises(ClaudeProcessError) as verbose_raised:
            run_turn(verbose_client, "crash")
        self.assertIn("/private/sensitive", str(verbose_raised.exception))


if __name__ == "__main__":
    unittest.main()
