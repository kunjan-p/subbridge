"""Keys, bodies, routing, concurrency, and lifecycle of the localhost gateway."""

import json
import socket
import unittest
from concurrent.futures import ThreadPoolExecutor
from typing import Any, ClassVar

import anthropic
import openai
from gateway_support import GatewayTestCase

import subbridge
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


if __name__ == "__main__":
    unittest.main()
