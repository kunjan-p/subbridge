import os
import textwrap
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

from subbridge._codex import CodexClient
from subbridge._codex_thread import CodexThread, ThreadOptions, _schema_file
from subbridge._errors import (
    CodexProcessError,
    CodexTurnError,
    CodexWrongAuthModeError,
)
from subbridge._models import TurnResult


def run_turn(
    client: CodexClient,
    prompt: str,
    *,
    model: str | None = None,
    timeout: float | None = 300,
) -> TurnResult:
    """Exercise a turn the way the gateway does: a fresh thread, then run()."""
    return client.start_thread(model=model).run(prompt, timeout=timeout)


def fake_codex(tmp_path: Path, status: str = "Logged in using ChatGPT") -> Path:
    cli = tmp_path / "codex"
    cli.write_text(
        textwrap.dedent(
            f"""\
            #!{os.sys.executable}
            import json
            import sys
            import time

            args = sys.argv[1:]
            with open(__file__ + ".calls", "a") as log:
                print(json.dumps(args), file=log)
            if args == ["--version"]:
                print("codex-cli test")
            elif args == ["login", "status"]:
                print({status!r})
            elif args == ["debug", "models"]:
                print(json.dumps({{"models": [
                    {{"slug": "gpt-test-small", "display_name": "Test Small", "priority": 1,
                      "visibility": "list", "supported_in_api": True,
                      "model_messages": {{"instructions_template": "DO NOT EXPOSE THIS"}},
                      "supported_reasoning_levels": [{{"effort": "low"}}]}},
                    {{"slug": "hidden-model", "visibility": "hide"}}
                ]}}))
            elif args[:2] == ["exec", "--json"] or ("--json" in args and args[0] == "exec"):
                prompt = sys.stdin.read()
                if prompt == "crash":
                    print("account=/private/sensitive", file=sys.stderr)
                    sys.exit(7)
                if prompt == "sleep":
                    time.sleep(2)
                if prompt == "no-terminal":
                    print(json.dumps({{"type": "item.completed", "item": {{"type": "agent_message", "text": "partial"}}}}), flush=True)
                    sys.exit(0)
                if prompt == "plan-denied":
                    print(json.dumps({{"type": "turn.failed", "error": {{"message": "model is not supported when using Codex with a ChatGPT account"}}}}), flush=True)
                    sys.exit(1)
                if prompt == "usage-limit":
                    print(json.dumps({{"type": "turn.failed", "error": {{"message": "You've hit your usage limit. Try again later."}}}}), flush=True)
                    sys.exit(1)
                if prompt == "partial-then-error":
                    print(json.dumps({{"type": "item.completed", "item": {{"type": "agent_message", "text": "partial"}}}}), flush=True)
                    print(json.dumps({{"type": "turn.failed", "error": {{"message": "You've hit your usage limit."}}}}), flush=True)
                    sys.exit(1)
                if prompt == "endless":
                    import os
                    with open(__file__ + ".pid", "w") as pid_file:
                        pid_file.write(str(os.getpid()))
                    while True:
                        print(json.dumps({{"type": "item.completed", "item": {{"type": "agent_message", "text": "tick "}}}}), flush=True)
                        time.sleep(0.05)
                if prompt == "which-model":
                    prompt = args[args.index("--model") + 1]
                if prompt == "large-event":
                    print(json.dumps({{"type": "event", "payload": "x" * 70000}}), flush=True)
                    sys.exit(0)
                print(json.dumps({{"type": "thread.started", "thread_id": "thread-123"}}), flush=True)
                if prompt == "two-messages":
                    print(json.dumps({{"type": "item.completed", "item": {{"type": "agent_message", "text": "Let me check the file."}}}}), flush=True)
                reply = "answer: " + prompt
                if "--output-schema" in args:
                    reply = json.dumps({{"answer": 42}})
                print(json.dumps({{"type": "item.completed", "item": {{"type": "agent_message", "text": reply}}}}), flush=True)
                print(json.dumps({{"type": "turn.completed", "usage": {{
                    "input_tokens": 11,
                    "cached_input_tokens": 4,
                    "output_tokens": 8,
                    "reasoning_output_tokens": 2,
                }}}}), flush=True)
            else:
                print("unknown command", file=sys.stderr)
                sys.exit(2)
            """
        ),
        encoding="utf-8",
    )
    cli.chmod(0o755)
    return cli


class CodexClientTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = TemporaryDirectory()
        self.tmp_path = Path(self.tempdir.name)

    def tearDown(self) -> None:
        self.tempdir.cleanup()

    def test_status_detects_chatgpt_login(self) -> None:
        client = CodexClient(codex_path=str(fake_codex(self.tmp_path)))
        status = client.status()
        self.assertTrue(status.installed)
        self.assertTrue(status.authenticated)
        self.assertEqual(status.auth_mode, "chatgpt")
        self.assertEqual(status.version, "codex-cli test")

    def test_subscription_only_strips_endpoint_override(self) -> None:
        env = {"OPENAI_BASE_URL": "http://proxy.invalid", "PATH": "/bin"}
        self.assertNotIn("OPENAI_BASE_URL", CodexClient(env=env)._environment())

    def test_run_streams_usage(self) -> None:
        client = CodexClient(codex_path=str(fake_codex(self.tmp_path)))
        result = run_turn(client, "hello", model="gpt-test-small")
        self.assertEqual(result.text, "answer: hello")
        self.assertEqual(result.thread_id, "thread-123")
        self.assertIsNotNone(result.usage)
        self.assertEqual(result.usage.input_tokens, 11)
        self.assertEqual(result.usage.cached_input_tokens, 4)
        self.assertEqual(result.usage.output_tokens, 8)
        self.assertEqual(result.usage.reasoning_output_tokens, 2)
        self.assertEqual(result.provider, "codex")
        self.assertEqual(result.model, "gpt-test-small")
        self.assertGreaterEqual(result.elapsed_seconds, 0)

    def test_status_hides_raw_auth_details_unless_requested(self) -> None:
        client = CodexClient(codex_path=str(fake_codex(self.tmp_path)))
        self.assertIsNone(client.status().raw_status)
        self.assertIn(
            "Logged in using ChatGPT", client.status(include_raw=True).raw_status
        )

    def test_capabilities_expose_sanitized_cli_catalog(self) -> None:
        client = CodexClient(codex_path=str(fake_codex(self.tmp_path)))
        capabilities = client.capabilities()
        self.assertEqual(capabilities.provider, "codex")
        self.assertEqual(capabilities.model_catalog_source, "codex debug models")
        self.assertFalse(capabilities.model_access_verified)
        self.assertEqual(
            [model.id for model in capabilities.models], ["gpt-test-small"]
        )
        self.assertEqual(capabilities.models[0].reasoning_efforts, ("low",))
        self.assertNotIn("instructions_template", repr(capabilities.models))

    def test_subscription_only_rejects_api_key_login(self) -> None:
        cli = fake_codex(self.tmp_path, status="Logged in using an API key")
        client = CodexClient(codex_path=str(cli))
        with self.assertRaises(CodexWrongAuthModeError):
            run_turn(client, "hello")

    def test_build_command_uses_current_json_flag_and_resume(self) -> None:
        client = CodexClient(codex_path=str(fake_codex(self.tmp_path)))
        # `resume_thread()` is gone, but resuming an existing conversation
        # remains a `CodexThread` capability: constructing one with a thread
        # ID directly exercises the same `_build_command()` resume branch.
        thread = CodexThread(
            client=client,
            thread_id="thread-xyz",
            options=ThreadOptions(model="test-model", cwd=self.tmp_path),
        )
        command = thread._build_command()
        self.assertIn("--json", command)
        self.assertNotIn("--experimental-json", command)
        self.assertIn("resume", command)
        self.assertIn("thread-xyz", command)

    def test_timeout_stops_the_cli(self) -> None:
        client = CodexClient(codex_path=str(fake_codex(self.tmp_path)))
        with self.assertRaisesRegex(CodexProcessError, "timed out"):
            run_turn(client, "sleep", timeout=0.05)

    def test_protocol_and_plan_failures_are_explicit(self) -> None:
        client = CodexClient(codex_path=str(fake_codex(self.tmp_path)))
        with self.assertRaisesRegex(
            CodexTurnError, "unavailable on this account plan"
        ) as raised:
            run_turn(client, "plan-denied")
        self.assertIn("not supported when using Codex", str(raised.exception))
        with self.assertRaisesRegex(Exception, "terminal turn.completed"):
            run_turn(client, "no-terminal")

    def test_sync_stream_rejects_oversized_jsonl_event(self) -> None:
        client = CodexClient(codex_path=str(fake_codex(self.tmp_path)))
        with (
            mock.patch("subbridge._stream.SYNC_STREAM_LINE_LIMIT", 1024),
            self.assertRaisesRegex(Exception, "larger than"),
        ):
            run_turn(client, "large-event")

    def test_schema_file_is_removed_after_use(self) -> None:
        with _schema_file({"type": "object"}) as path:
            self.assertIsNotNone(path)
            self.assertTrue(Path(path).exists())
        self.assertFalse(Path(path).exists())

    def test_process_diagnostics_are_redacted_by_default(self) -> None:
        client = CodexClient(codex_path=str(fake_codex(self.tmp_path)))
        with self.assertRaises(CodexProcessError) as raised:
            run_turn(client, "crash")
        self.assertNotIn("/private/sensitive", str(raised.exception))

        verbose_client = CodexClient(
            codex_path=str(fake_codex(self.tmp_path)),
            include_raw_diagnostics=True,
        )
        with self.assertRaises(CodexProcessError) as verbose_raised:
            run_turn(verbose_client, "crash")
        self.assertIn("/private/sensitive", str(verbose_raised.exception))


if __name__ == "__main__":
    unittest.main()
