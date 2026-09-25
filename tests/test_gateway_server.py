"""Keys, bodies, routing, concurrency, and lifecycle of the localhost gateway."""

import http.client
import json
import os
import socket
import sys
import textwrap
import threading
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from contextlib import suppress
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, ClassVar
from unittest import mock

import anthropic
import openai
from gateway_support import GatewayTestCase

import subbridge
from subbridge.gateway._anthropic import MessagesEndpoint
from subbridge.gateway._server import MAX_BODY_BYTES

HELLO_BODY = json.dumps(
    {
        "model": "sonnet",
        "max_tokens": 10,
        "messages": [{"role": "user", "content": "hi"}],
    }
).encode()


class KeyTests(GatewayTestCase):
    def test_wrong_key_is_401_and_never_starts_a_cli(self) -> None:
        wrong_anthropic = anthropic.Anthropic(
            base_url=self.gateway.anthropic_base_url, api_key="sb-local-wrong"
        )
        wrong_openai = openai.OpenAI(
            base_url=self.gateway.openai_base_url, api_key="sb-local-wrong"
        )
        with self.assertRaises(anthropic.AuthenticationError) as raised:
            wrong_anthropic.messages.create(
                model="sonnet",
                max_tokens=10,
                messages=[{"role": "user", "content": "hi"}],
            )
        self.assertNotIn("sb-local-wrong", raised.exception.message)
        with self.assertRaises(openai.AuthenticationError):
            wrong_openai.responses.create(model="gpt-test", input="hi")
        self.assertEqual(self.cli_calls("claude"), [])
        self.assertEqual(self.cli_calls("codex"), [])

    def test_missing_key_is_401(self) -> None:
        status, body = self.post("/v1/messages", HELLO_BODY)
        self.assertEqual(status, 401)
        self.assertEqual(body["error"]["type"], "authentication_error")
        self.assertEqual(self.cli_calls("claude"), [])

    def test_bearer_key_is_accepted(self) -> None:
        client = anthropic.Anthropic(
            base_url=self.gateway.anthropic_base_url, auth_token=self.gateway.api_key
        )
        message = client.messages.create(
            model="sonnet",
            max_tokens=10,
            messages=[{"role": "user", "content": "hello"}],
        )
        self.assertEqual(message.content[0].text, "answer: hello")

    def test_each_gateway_gets_its_own_key(self) -> None:
        with subbridge.serve() as other:
            self.assertNotEqual(other.api_key, self.gateway.api_key)
        self.assertTrue(self.gateway.api_key.startswith("sb-local-"))

    def test_huge_content_length_without_a_key_is_a_quick_401(self) -> None:
        connection = http.client.HTTPConnection(
            "127.0.0.1", self.gateway.port, timeout=10
        )
        self.addCleanup(connection.close)
        connection.putrequest("POST", "/v1/messages")
        connection.putheader("Content-Length", "5000000000")
        connection.endheaders()
        # A real attacker would stall here forever; this test just never
        # sends the other ~5 GB it claimed.
        connection.send(b"{}\n")
        started = time.monotonic()
        response = connection.getresponse()
        body = json.loads(response.read())
        self.assertLess(time.monotonic() - started, 2)
        self.assertEqual(response.status, 401)
        self.assertEqual(body["error"]["type"], "authentication_error")
        self.assertEqual(self.cli_calls("claude"), [])


class BodyTests(GatewayTestCase):
    def test_malformed_json_is_400(self) -> None:
        status, body = self.post("/v1/messages", b"{nope", self.key_headers())
        self.assertEqual(status, 400)
        self.assertEqual(body["error"]["type"], "invalid_request_error")
        self.assertIn("not valid JSON", body["error"]["message"])

    def test_json_that_is_not_an_object_is_400(self) -> None:
        status, _ = self.post("/v1/messages", b"[]", self.key_headers())
        self.assertEqual(status, 400)

    def test_oversized_body_is_413_and_never_starts_a_cli(self) -> None:
        status, body = self.post(
            "/v1/messages", b" " * (MAX_BODY_BYTES + 1), self.key_headers()
        )
        self.assertEqual(status, 413)
        self.assertEqual(body["error"]["type"], "request_too_large")
        self.assertEqual(self.cli_calls("claude"), [])

    def test_flag_like_model_name_is_rejected(self) -> None:
        with self.assertRaises(anthropic.BadRequestError) as raised:
            self.anthropic.messages.create(
                model="--dangerously-skip-permissions",
                max_tokens=10,
                messages=[{"role": "user", "content": "hi"}],
            )
        self.assertIn("`model`", raised.exception.message)
        self.assertEqual(self.cli_calls("claude"), [])


class ConcurrencyTests(GatewayTestCase):
    gateway_options: ClassVar[dict[str, Any]] = {"max_concurrency": 1, "timeout": 30}

    def test_requests_beyond_the_limit_wait_instead_of_failing(self) -> None:
        def ask(index: int) -> str:
            message = self.anthropic.messages.create(
                model="sonnet",
                max_tokens=10,
                messages=[{"role": "user", "content": f"q{index}"}],
            )
            return message.content[0].text

        with ThreadPoolExecutor(max_workers=3) as pool:
            answers = list(pool.map(ask, range(3)))
        self.assertEqual(answers, [f"answer: q{index}" for index in range(3)])


class BusyTests(GatewayTestCase):
    gateway_options: ClassVar[dict[str, Any]] = {"max_concurrency": 1, "timeout": 0.2}

    def test_request_that_never_gets_a_slot_is_503(self) -> None:
        self.gateway._server.slots.acquire()
        self.addCleanup(self.gateway._server.slots.release)
        with self.assertRaises(anthropic.InternalServerError) as raised:
            self.anthropic.messages.create(
                model="sonnet",
                max_tokens=10,
                messages=[{"role": "user", "content": "hi"}],
            )
        self.assertEqual(raised.exception.status_code, 503)
        self.assertEqual(self.turn_calls("claude"), [])


class RetryTests(GatewayTestCase):
    def test_sdk_default_retries_do_not_rerun_a_failed_turn(self) -> None:
        client = anthropic.Anthropic(
            base_url=self.gateway.anthropic_base_url, api_key=self.gateway.api_key
        )
        with self.assertRaises(anthropic.RateLimitError):
            client.messages.create(
                model="sonnet",
                max_tokens=10,
                messages=[{"role": "user", "content": "is-error"}],
            )
        self.assertEqual(len(self.turn_calls("claude")), 1)


class UnsupportedEndpointTests(GatewayTestCase):
    def test_embeddings_get_a_clear_404_in_openai_shape(self) -> None:
        with self.assertRaises(openai.NotFoundError) as raised:
            self.openai.embeddings.create(model="text-embedding-3-small", input="hi")
        self.assertIn("not supported by SubBridge's gateway", raised.exception.message)
        self.assertEqual(raised.exception.code, "unsupported_endpoint")


class UnsupportedMethodTests(GatewayTestCase):
    def test_delete_gets_a_404_in_openai_shape(self) -> None:
        connection = http.client.HTTPConnection(
            "127.0.0.1", self.gateway.port, timeout=10
        )
        self.addCleanup(connection.close)
        connection.request("DELETE", "/v1/messages", headers=self.key_headers())
        response = connection.getresponse()
        body = json.loads(response.read())
        self.assertEqual(response.status, 404)
        self.assertEqual(body["error"]["code"], "unsupported_endpoint")
        self.assertIn("not supported by SubBridge's gateway", body["error"]["message"])
        self.assertEqual(response.getheader("x-should-retry"), "false")
        self.assertEqual(self.cli_calls("claude"), [])


class UnexpectedErrorTests(GatewayTestCase):
    def test_a_bug_in_an_endpoint_gets_a_500_instead_of_a_dropped_connection(
        self,
    ) -> None:
        with (
            mock.patch.object(
                MessagesEndpoint, "parse", side_effect=ValueError("boom")
            ),
            self.assertRaises(anthropic.InternalServerError) as raised,
        ):
            self.anthropic.messages.create(
                model="sonnet",
                max_tokens=10,
                messages=[{"role": "user", "content": "hi"}],
            )
        self.assertEqual(raised.exception.status_code, 500)
        self.assertEqual(self.cli_calls("claude"), [])


class LifecycleTests(unittest.TestCase):
    def test_binds_localhost_and_stops_listening_on_close(self) -> None:
        gateway = subbridge.serve()
        self.assertEqual(gateway._server.server_address[0], "127.0.0.1")
        self.assertEqual(gateway.anthropic_base_url, f"http://127.0.0.1:{gateway.port}")
        self.assertEqual(gateway.openai_base_url, f"http://127.0.0.1:{gateway.port}/v1")
        gateway.close()
        gateway.close()
        with self.assertRaises(ConnectionRefusedError):
            socket.create_connection(("127.0.0.1", gateway.port), timeout=1)

    def test_rejects_a_zero_concurrency_limit(self) -> None:
        with self.assertRaises(ValueError):
            subbridge.serve(max_concurrency=0)


def _fake_claude_that_stalls(bin_dir: Path, pid_file: Path, sleep_seconds: int) -> Path:
    """A minimal fake `claude` that records its PID before stalling.

    Task 5 adds `.pid` files to the shared fake CLIs; this one is local to
    this test so the fix doesn't need to touch those ahead of that task.
    """
    cli = bin_dir / "claude"
    cli.write_text(
        textwrap.dedent(
            f"""\
            #!{sys.executable}
            import json
            import os
            import sys
            import time

            args = sys.argv[1:]
            if args == ["--version"]:
                print("2.1.test")
            elif args == ["auth", "status", "--json"]:
                print(json.dumps({{"loggedIn": True, "authMethod": "claude.ai"}}))
            elif "--output-format" in args:
                prompt = sys.stdin.read()
                with open({str(pid_file)!r}, "w") as pid_file_handle:
                    pid_file_handle.write(str(os.getpid()))
                time.sleep({sleep_seconds})
                print(json.dumps({{"type": "result", "subtype": "success", "result": "answer: " + prompt}}))
            else:
                sys.exit(2)
            """
        ),
        encoding="utf-8",
    )
    cli.chmod(0o755)
    return cli


class CloseDuringTurnTests(unittest.TestCase):
    def test_close_stops_an_in_flight_turn_and_returns_quickly(self) -> None:
        with TemporaryDirectory() as tempdir:
            bin_dir = Path(tempdir)
            pid_file = bin_dir / "claude.pid"
            _fake_claude_that_stalls(bin_dir, pid_file, sleep_seconds=20)
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


if __name__ == "__main__":
    unittest.main()
