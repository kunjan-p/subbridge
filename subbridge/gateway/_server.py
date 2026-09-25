"""The localhost HTTP server: routing, key checks, bodies, and replies."""

from __future__ import annotations

import hmac
import json
import threading
from collections.abc import Generator
from contextlib import contextmanager, suppress
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import urlsplit

from ..claude import ClaudeClient
from ..codex import CodexClient
from ..errors import SubBridgeError
from ..models import ProviderName, TurnResult
from ._anthropic import MessagesEndpoint
from ._errors import ErrorStyle, GatewayError, error_body, from_exception
from ._turns import Endpoint, TurnRequest

HOST = "127.0.0.1"
MAX_BODY_BYTES = 10 * 1024 * 1024
_DISCARD_CHUNK = 1024 * 1024

ROUTES: dict[str, type[Endpoint]] = {
    "/v1/messages": MessagesEndpoint,
}


class GatewayServer(ThreadingHTTPServer):
    def __init__(
        self,
        port: int,
        *,
        api_key: str,
        max_concurrency: int,
        turn_timeout: float | None,
        workdir: str,
    ) -> None:
        super().__init__((HOST, port), GatewayHandler)
        self.api_key = api_key
        self.turn_timeout = turn_timeout
        self.workdir = workdir
        self.slots = threading.BoundedSemaphore(max_concurrency)
        self.clients = {"claude": ClaudeClient(), "codex": CodexClient()}

    @contextmanager
    def cli_slot(self) -> Generator[None, None, None]:
        """Hold one of the max_concurrency CLI slots, waiting up to the timeout."""
        if not self.slots.acquire(timeout=self.turn_timeout):
            raise GatewayError(
                503, f"Every CLI slot stayed busy for {self.turn_timeout} seconds."
            )
        try:
            yield
        finally:
            self.slots.release()

    def run(self, provider: ProviderName, request: TurnRequest) -> TurnResult:
        thread = self._thread(provider, request)
        return thread.run(
            request.prompt,
            output_schema=request.output_schema,
            timeout=self.turn_timeout,
        )

    def _thread(self, provider: ProviderName, request: TurnRequest) -> Any:
        client = self.clients[provider]
        return client.start_thread(model=request.model, cwd=self.workdir)

    def key_matches(self, supplied: str | None) -> bool:
        if not supplied:
            return False
        return hmac.compare_digest(supplied.encode(), self.api_key.encode())


class GatewayHandler(BaseHTTPRequestHandler):
    server: GatewayServer
    server_version = "SubBridgeGateway"

    def do_POST(self) -> None:
        route = ROUTES.get(urlsplit(self.path).path)
        self._dispatch(route() if route else None)

    def do_GET(self) -> None:
        self._dispatch(None)

    def _dispatch(self, endpoint: Endpoint | None) -> None:
        style: ErrorStyle = endpoint.style if endpoint else "openai"
        try:
            self._handle(endpoint)
        except ConnectionError:
            return  # the client left before the reply
        except SubBridgeError as exc:
            with suppress(OSError):
                self._send_error(style, from_exception(exc))

    def _handle(self, endpoint: Endpoint | None) -> None:
        body = self._read_body()
        self._check_key()
        if endpoint is None:
            raise _unsupported_route(self.command, urlsplit(self.path).path)
        request = endpoint.parse(_parse_json(body))
        self._respond(endpoint, request)

    def _read_body(self) -> bytes | None:
        """Read the whole body, so closing the socket never resets the reply.

        Returns None when the body is over the size limit (it is discarded).
        """
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            raise GatewayError(400, "Content-Length must be an integer.") from None
        if length < 0:
            raise GatewayError(400, "Content-Length must not be negative.")
        if length <= MAX_BODY_BYTES:
            return self.rfile.read(length)
        while length > 0 and (chunk := self.rfile.read(min(length, _DISCARD_CHUNK))):
            length -= len(chunk)
        return None

    def _check_key(self) -> None:
        bearer = self.headers.get("Authorization", "")
        token = bearer[7:] if bearer[:7].lower() == "bearer " else None
        if not (
            self.server.key_matches(self.headers.get("x-api-key"))
            or self.server.key_matches(token)
        ):
            raise GatewayError(
                401, "Missing or wrong SubBridge gateway key.", code="invalid_api_key"
            )

    def _respond(self, endpoint: Endpoint, request: TurnRequest) -> None:
        with self.server.cli_slot():
            result = self.server.run(endpoint.provider, request)
        self._send_json(200, endpoint.respond(result))

    def _send_error(self, style: ErrorStyle, error: GatewayError) -> None:
        self._send_json(error.status, error_body(style, error))

    def _send_json(self, status: int, payload: dict[str, Any]) -> None:
        data = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        if status >= 400:
            # A retry would start the CLI again and spend more of the plan.
            self.send_header("x-should-retry", "false")
        self.end_headers()
        self.wfile.write(data)

    def log_request(self, code: int | str = "-", size: int | str = "-") -> None:
        """Stay quiet per request; errors still reach log_error."""


def _unsupported_route(method: str, path: str) -> GatewayError:
    return GatewayError(
        404,
        f"{method} {path} is not supported by SubBridge's gateway. It serves "
        "POST /v1/messages, /v1/chat/completions, and /v1/responses only.",
        code="unsupported_endpoint",
    )


def _parse_json(body: bytes | None) -> dict[str, Any]:
    if body is None:
        raise GatewayError(
            413,
            f"Request body is larger than {MAX_BODY_BYTES} bytes.",
            code="request_too_large",
        )
    try:
        payload = json.loads(body)
    except (json.JSONDecodeError, UnicodeDecodeError):
        raise GatewayError(
            400, "Request body is not valid JSON.", code="invalid_json"
        ) from None
    if not isinstance(payload, dict):
        raise GatewayError(
            400, "Request body must be a JSON object.", code="invalid_json"
        )
    return payload
