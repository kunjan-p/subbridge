"""The localhost HTTP server: routing, key checks, bodies, and replies."""

from __future__ import annotations

import hmac
import json
import logging
import threading
import time
from collections.abc import Generator
from contextlib import contextmanager, suppress
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import urlsplit

from .._process import terminate_process_tree
from ..claude import ClaudeClient
from ..codex import CodexClient
from ..errors import SubBridgeError
from ..models import ProviderName, TurnResult
from ._anthropic import MessagesEndpoint
from ._errors import ErrorStyle, GatewayError, error_body, from_exception
from ._turns import Endpoint, TurnRequest

HOST = "127.0.0.1"
MAX_BODY_BYTES = 10 * 1024 * 1024
_HANDLER_TIMEOUT = 30.0
_ERROR_DRAIN_CHUNK = 64 * 1024
_ERROR_DRAIN_SECONDS = 1.0
_STOP_GRACE_SECONDS = 2.0
_STOP_POLL_SECONDS = 0.02

ROUTES: dict[str, type[Endpoint]] = {
    "/v1/messages": MessagesEndpoint,
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
        self._active_lock = threading.Lock()
        self._active_threads: set[threading.Thread] = set()
        self._closing = False

    @contextmanager
    def cli_slot(self) -> Generator[None, None, None]:
        """Hold one of the max_concurrency CLI slots, waiting up to the timeout.

        Registering in `_active_threads` happens in the same critical
        section as the `_closing` check, so a turn that was queued on
        `slots.acquire()` while `close()` started can never register (and
        start a CLI) after `stop_in_flight_turns()` has already looked: it
        either registered before `_closing` was set (and `close()`'s loop
        will see and stop it) or it observes `_closing` here and never
        starts at all.
        """
        if not self.slots.acquire(timeout=self.turn_timeout):
            raise GatewayError(
                503, f"Every CLI slot stayed busy for {self.turn_timeout} seconds."
            )
        thread = threading.current_thread()
        with self._active_lock:
            if self._closing:
                self.slots.release()
                raise GatewayError(
                    503,
                    "The SubBridge gateway is shutting down.",
                    code="gateway_closing",
                )
            self._active_threads.add(thread)
        try:
            yield
        finally:
            with self._active_lock:
                self._active_threads.discard(thread)
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

    def begin_closing(self) -> None:
        """Refuse any turn that has not yet registered a slot.

        Must be called (and must complete) before `stop_in_flight_turns()`
        starts looking for turns to stop: `cli_slot()` checks this flag and
        registers in `_active_threads` in the same critical section, so once
        this is set, no new thread can slip past `stop_in_flight_turns()`'s
        loop and start a CLI of its own.
        """
        with self._active_lock:
            self._closing = True

    def stop_in_flight_turns(self) -> None:
        """Kill the CLI process tree of every turn still running, then wait
        (bounded) until every request thread has noticed and exited.

        This loops -- rather than acting on one snapshot -- because a turn
        that read `_closing` as false (in `cli_slot()`) just before
        `begin_closing()` set it can still appear in `_active_threads` after
        this starts, and because a turn's `subprocess.Popen` (see
        `_stream.EventStream.sync`) may not exist yet the first time this
        looks. `begin_closing()` guarantees no *other* new turn joins
        `_active_threads` after that point, so this loop is bounded to
        turns that were already starting; it still has an overall deadline
        so `close()` cannot hang.
        """
        deadline = time.monotonic() + _STOP_GRACE_SECONDS
        killed: set[int] = set()
        while time.monotonic() < deadline:
            with self._active_lock:
                threads = list(self._active_threads)
            if not threads:
                return
            for thread in threads:
                if id(thread) in killed:
                    continue
                process = getattr(thread, "subbridge_process", None)
                if process is not None:
                    terminate_process_tree(process)
                    killed.add(id(thread))
            time.sleep(_STOP_POLL_SECONDS)
        # The deadline passed (a straggler's process was never found, or
        # never finished after being killed); join what's left with a short
        # bound each so close() still returns promptly.
        with self._active_lock:
            threads = list(self._active_threads)
        for thread in threads:
            thread.join(timeout=_STOP_GRACE_SECONDS)


class GatewayHandler(BaseHTTPRequestHandler):
    server: GatewayServer
    server_version = "SubBridgeGateway"
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
        except ConnectionError:
            return  # the client left before the reply
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
        body = self.rfile.read(length)
        self._consumed = length
        return body

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

    def _fail(self, style: ErrorStyle, error: GatewayError) -> None:
        # A reply to an error is often sent with request bytes still unread
        # (an unchecked key, an oversized body). Draining a bounded amount
        # keeps the kernel from resetting the connection instead of
        # delivering this reply; the drain has its own short timeout so a
        # caller that declares a huge body and then never sends it cannot
        # make this wait long for a reply.
        self._drain_leftover_body()
        with suppress(OSError):
            self._send_error(style, error)

    def _drain_leftover_body(self) -> None:
        """Read the body in pieces of at most 64 KB for a bounded total time.

        Uses `rfile.read1()`, not `rfile.read()`: `read()` loops internally,
        issuing as many underlying `recv()` calls as it takes to fill the
        requested size, and only *each* of those calls (not the loop as a
        whole) honors the socket timeout -- a client trickling in a byte
        every so often (below the per-call timeout each time) could still
        hold this thread well past the intended budget. `read1()` makes at
        most one `recv()` call (returning early with whatever that yields,
        or from `rfile`'s own buffer first if anything is already sitting
        there -- so nothing it already buffered while parsing the request
        is lost), so the deadline below is re-checked between every network
        read, not just every 64 KB.

        A caller that is actually sending its (possibly oversized) body
        drains fully well inside the budget on a local connection. A caller
        that declared a huge body and then stalls, or trickles it in far
        slower than that, gets cut off once the budget runs out.
        """
        try:
            remaining = self._declared_length() - self._consumed
        except GatewayError:
            return
        if remaining <= 0:
            return
        previous_timeout = self.connection.gettimeout()
        deadline = time.monotonic() + _ERROR_DRAIN_SECONDS
        with suppress(OSError):
            while remaining > 0:
                budget = deadline - time.monotonic()
                if budget <= 0:
                    break
                self.connection.settimeout(budget)
                chunk = self.rfile.read1(min(remaining, _ERROR_DRAIN_CHUNK))
                if not chunk:
                    break
                remaining -= len(chunk)
        with suppress(OSError):
            self.connection.settimeout(previous_timeout)

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

    def log_request(self, code: int | str = "-", size: int | str = "-") -> None:
        """Stay quiet per request; errors still reach log_error."""


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
