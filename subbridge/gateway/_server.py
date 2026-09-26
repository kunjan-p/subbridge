"""The localhost HTTP server: routing, key checks, bodies, replies, and SSE writing."""

from __future__ import annotations

import hmac
import json
import logging
import threading
import time
from collections.abc import Iterable
from contextlib import AbstractContextManager, closing, suppress
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import urlsplit

from .._claude import ClaudeClient
from .._codex import CodexClient
from .._errors import SubBridgeError
from .._models import ProviderName, TurnResult
from ._anthropic import MessagesEndpoint
from ._errors import ErrorStyle, GatewayError, error_body, from_exception
from ._lifecycle import TurnLifecycle, drain_leftover_body
from ._openai_chat import ChatCompletionsEndpoint
from ._openai_responses import ResponsesEndpoint
from ._turns import Endpoint, Frame, TextStream, TurnRequest

HOST = "127.0.0.1"
MAX_BODY_BYTES = 10 * 1024 * 1024
_HANDLER_TIMEOUT = 30.0
# `rfile.read1()` makes at most one `recv()` call, so re-checking the
# deadline every `_READ_CHUNK` bytes (rather than after one `read()` that
# could loop internally forever) is what lets `_read_exact()` bound the
# whole body, not just each individual network read.
_READ_CHUNK = 64 * 1024

ROUTES: dict[str, type[Endpoint]] = {
    "/v1/messages": MessagesEndpoint,
    "/v1/chat/completions": ChatCompletionsEndpoint,
    "/v1/responses": ResponsesEndpoint,
}

_logger = logging.getLogger(__name__)


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
        self._lifecycle = TurnLifecycle(self.slots, turn_timeout)

    def cli_slot(self) -> AbstractContextManager[None]:
        return self._lifecycle.cli_slot()

    def begin_closing(self) -> None:
        self._lifecycle.begin_closing()

    def stop_in_flight_turns(self) -> None:
        self._lifecycle.stop_in_flight_turns()

    def run(self, provider: ProviderName, request: TurnRequest) -> TurnResult:
        thread = self._thread(provider, request)
        return thread.run(
            request.prompt,
            output_schema=request.output_schema,
            timeout=self.turn_timeout,
        )

    def open_stream(self, provider: ProviderName, request: TurnRequest) -> TextStream:
        thread = self._thread(provider, request)
        events = thread.stream_normalized(
            request.prompt,
            output_schema=request.output_schema,
            timeout=self.turn_timeout,
        )
        return TextStream(
            provider, events, only_final_message=request.output_schema is not None
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
    # Empty rather than the default "Python/X.Y.Z", so the Server header
    # doesn't advertise the interpreter version to anything on this machine
    # that can reach the gateway.
    sys_version = ""
    # Bounds every blocking read on this connection, so a client that opens a
    # request and then never sends (or finishes) it cannot hold this thread
    # forever.
    timeout = _HANDLER_TIMEOUT

    def do_POST(self) -> None:
        route = ROUTES.get(urlsplit(self.path).path)
        self._dispatch(route() if route else None)

    def do_GET(self) -> None:
        self._dispatch(None)

    do_PUT = do_GET
    do_DELETE = do_GET
    do_PATCH = do_GET
    do_OPTIONS = do_GET

    def do_HEAD(self) -> None:
        self._dispatch(None, send_body=False)

    def _dispatch(self, endpoint: Endpoint | None, *, send_body: bool = True) -> None:
        style: ErrorStyle = endpoint.style if endpoint else "openai"
        self._send_body = send_body
        self._consumed = 0
        try:
            self._handle(endpoint)
        except OSError:
            # The client left, or a write to it timed out (it stopped
            # reading). Either way we can't reply -- and must not try to,
            # since a status line may already be partway out.
            return
        except SubBridgeError as exc:
            self._fail(style, from_exception(exc))
        except Exception:
            # Logged (with traceback) rather than swallowed, and still
            # answered here so the caller sees a reply, not a retry-loop.
            _logger.exception("Unhandled error while handling a gateway request")
            self._fail(
                style,
                GatewayError(500, "The SubBridge gateway hit an unexpected error."),
            )

    def _handle(self, endpoint: Endpoint | None) -> None:
        # The key is checked before any body is read, so a caller with a
        # wrong or missing key cannot make this thread wait on a body it
        # never intends to (fully) send.
        self._check_key()
        if endpoint is None:
            raise _unsupported_route(self.command, urlsplit(self.path).path)
        request = endpoint.parse(_parse_json(self._read_body()))
        if request.stream:
            self._stream(endpoint, request)
        else:
            self._respond(endpoint, request)

    def _declared_length(self) -> int:
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            raise GatewayError(400, "Content-Length must be an integer.") from None
        if length < 0:
            raise GatewayError(400, "Content-Length must not be negative.")
        return length

    def _read_body(self) -> bytes:
        length = self._declared_length()
        if length > MAX_BODY_BYTES:
            raise GatewayError(
                413,
                f"Request body is larger than {MAX_BODY_BYTES} bytes.",
                code="request_too_large",
            )
        return self._read_exact(length)

    def _read_exact(self, length: int) -> bytes:
        """Read exactly `length` bytes, bounded by one overall deadline.

        `self.rfile.read(length)` only has `self.timeout` bound each
        individual `recv()` it issues, not the read as a whole, so a client
        sending one byte every `_HANDLER_TIMEOUT - epsilon` seconds could
        hold this thread here indefinitely. This instead re-sets the socket
        timeout to what's left of one overall deadline before every
        `read1()` call (at most one `recv()` each, like
        `drain_leftover_body()`), so the deadline is checked between every
        network read, not just once per call. Like `drain_leftover_body()`,
        it also restores the connection's previous timeout before
        returning (successfully or not), so a near-zero leftover budget
        from this read doesn't leak into the response write, an SSE
        stream, or the next keep-alive request on this connection.
        """
        chunks: list[bytes] = []
        remaining = length
        deadline = time.monotonic() + _HANDLER_TIMEOUT
        previous_timeout = self.connection.gettimeout()
        try:
            while remaining > 0:
                budget = deadline - time.monotonic()
                if budget <= 0:
                    raise self._body_timeout()
                self.connection.settimeout(budget)
                try:
                    chunk = self.rfile.read1(min(remaining, _READ_CHUNK))
                except TimeoutError:
                    # A `read1()` that times out here always means our own
                    # deadline (not just this one recv) has elapsed, since
                    # its timeout above was set to exactly what's left of
                    # it. Raise straight from here instead of looping back
                    # to check that again: `self.rfile` refuses every read
                    # after the first timeout with a plain `OSError`
                    # (Python's `SocketIO` remembers it once timed out),
                    # which `_dispatch()` would swallow with no reply at
                    # all rather than see as this `GatewayError`.
                    raise self._body_timeout() from None
                if not chunk:
                    raise GatewayError(
                        400,
                        "Request body ended before Content-Length bytes arrived.",
                    )
                chunks.append(chunk)
                remaining -= len(chunk)
                self._consumed += len(chunk)
            return b"".join(chunks)
        finally:
            with suppress(OSError):
                self.connection.settimeout(previous_timeout)

    def _body_timeout(self) -> GatewayError:
        self.close_connection = True
        return GatewayError(
            408, "Timed out waiting for the request body.", code="request_timeout"
        )

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

    def _stream(self, endpoint: Endpoint, request: TurnRequest) -> None:
        # Held for the whole stream, not just until a result comes back, so
        # close() can find and kill this turn's CLI mid-stream too.
        with self.server.cli_slot():
            texts = self.server.open_stream(endpoint.provider, request)
            with closing(texts):
                texts.prime()
                self._start_sse()
                self._write_stream(endpoint, texts)

    def _write_stream(self, endpoint: Endpoint, texts: TextStream) -> None:
        try:
            self._write_frames(endpoint.stream(texts))
        except OSError:
            # The client left, or a write timed out because it stopped
            # reading; closing the stream stops the CLI either way.
            return
        except SubBridgeError as exc:
            with suppress(OSError):
                self._write_frames(endpoint.stream_error(from_exception(exc)))
        except Exception:
            # The 200 status line is already on the wire, so this cannot
            # become a fresh JSON 500 like `_dispatch()`'s bare except does
            # for a request that never started streaming; send the same
            # generic failure as an SSE error event instead.
            _logger.exception("Unhandled error while streaming a gateway reply")
            with suppress(OSError):
                self._write_frames(
                    endpoint.stream_error(
                        GatewayError(
                            500, "The SubBridge gateway hit an unexpected error."
                        )
                    )
                )

    def _fail(self, style: ErrorStyle, error: GatewayError) -> None:
        # A reply to an error is often sent with request bytes still unread
        # (an unchecked key, an oversized body). Draining a bounded amount
        # keeps the kernel from resetting the connection instead of
        # delivering this reply.
        self._drain_leftover_body()
        with suppress(OSError):
            self._send_error(style, error)

    def _drain_leftover_body(self) -> None:
        try:
            remaining = self._declared_length() - self._consumed
        except GatewayError:
            return
        drain_leftover_body(self.rfile, self.connection, remaining)

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
            self.send_header("Connection", "close")
        self.end_headers()
        if self._send_body:
            self.wfile.write(data)

    def _start_sse(self) -> None:
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()

    def _write_frames(self, frames: Iterable[Frame]) -> None:
        for event, data in frames:
            self.wfile.write(sse_frame(event, data))

    def log_request(self, code: int | str = "-", size: int | str = "-") -> None:
        """Stay quiet per request; errors still reach log_error."""


def sse_frame(event: str | None, data: Any) -> bytes:
    payload = data if isinstance(data, str) else json.dumps(data, separators=(",", ":"))
    head = f"event: {event}\n" if event else ""
    return f"{head}data: {payload}\n\n".encode()


def _unsupported_route(method: str, path: str) -> GatewayError:
    return GatewayError(
        404,
        f"{method} {path} is not supported by SubBridge's gateway. It serves "
        "POST /v1/messages, /v1/chat/completions, and /v1/responses only.",
        code="unsupported_endpoint",
    )


def _parse_json(body: bytes) -> dict[str, Any]:
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
