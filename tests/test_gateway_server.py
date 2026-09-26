"""Keys, bodies, routing, concurrency, and lifecycle of the localhost gateway."""

import http.client
import importlib.util
import json
import os
import socket
import tempfile
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from typing import Any, ClassVar
from unittest import mock

# The gateway tests need the (dev-only) anthropic and openai SDKs; the
# release workflow installs only requirements-release.txt, so skip cleanly
# there instead of failing to collect this module.
if (
    importlib.util.find_spec("anthropic") is None
    or importlib.util.find_spec("openai") is None
):
    raise unittest.SkipTest(
        "anthropic and openai are not installed; skipping gateway tests."
    )

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

    def test_trickling_body_without_a_key_is_a_quick_401(self) -> None:
        sock = socket.create_connection(("127.0.0.1", self.gateway.port), timeout=10)
        self.addCleanup(sock.close)
        sock.sendall(
            b"POST /v1/messages HTTP/1.1\r\nHost: x\r\nContent-Length: 1000000\r\n\r\n"
        )
        sock.settimeout(0.05)
        started = time.monotonic()
        reply = b""
        # Trickle the declared body in far slower than any single recv's
        # timeout, so a reply that only bounds *each* recv (not the drain
        # as a whole) would still hang here well past the assertion below.
        while time.monotonic() - started < 6:
            sock.sendall(b"a")
            try:
                reply = sock.recv(4096)
                if reply:
                    break
            except TimeoutError:
                pass
            time.sleep(0.3)
        self.assertLess(time.monotonic() - started, 2)
        self.assertIn(b"401", reply)
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

    def test_openai_sdk_default_retries_do_not_rerun_a_failed_codex_turn(self) -> None:
        client = openai.OpenAI(
            base_url=self.gateway.openai_base_url, api_key=self.gateway.api_key
        )
        with self.assertRaises(openai.RateLimitError):
            client.chat.completions.create(
                model="gpt-test",
                messages=[{"role": "user", "content": "usage-limit"}],
            )
        self.assertEqual(len(self.turn_calls("codex")), 1)


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


class ServerHeaderTests(GatewayTestCase):
    def test_server_header_omits_the_python_version(self) -> None:
        connection = http.client.HTTPConnection(
            "127.0.0.1", self.gateway.port, timeout=10
        )
        self.addCleanup(connection.close)
        connection.request("GET", "/", headers=self.key_headers())
        response = connection.getresponse()
        response.read()
        self.assertNotIn("Python/", response.getheader("Server") or "")


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

    def test_cleans_up_the_workdir_when_the_server_fails_to_start(self) -> None:
        """A `GatewayServer()` failure (for example the port is already in
        use) must not leak the empty workdir created just before it.

        Asserts that `cleanup()` is actually called, rather than checking
        the directory is gone afterward: CPython's own refcounting can also
        make an orphaned `TemporaryDirectory` clean itself up via its
        finalizer once nothing (a traceback, say) still references it, which
        would let this test pass even without the fix.
        """
        real_cleanup = tempfile.TemporaryDirectory.cleanup
        calls: list[str] = []

        def spy_cleanup(self: tempfile.TemporaryDirectory) -> None:
            calls.append(self.name)
            real_cleanup(self)

        with (
            mock.patch.object(tempfile.TemporaryDirectory, "cleanup", spy_cleanup),
            mock.patch(
                "subbridge.gateway.GatewayServer", side_effect=OSError("port busy")
            ),
            self.assertRaises(OSError),
        ):
            subbridge.serve()
        self.assertEqual(len(calls), 1)
        self.assertFalse(os.path.exists(calls[0]))


if __name__ == "__main__":
    unittest.main()
