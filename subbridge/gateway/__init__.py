"""A localhost gateway that lets the official Anthropic and OpenAI SDKs use the CLIs."""

from __future__ import annotations

import atexit
import os
import secrets
import tempfile
import threading
import warnings
from types import TracebackType
from typing import Self

from ._process_env import client_environment, proxy_capture_warning
from ._server import HOST, GatewayServer

__all__ = ["Gateway", "serve", "use_subscription"]


class Gateway:
    """A running gateway on 127.0.0.1. Use it as a context manager or call close()."""

    def __init__(
        self, *, port: int, max_concurrency: int, timeout: float | None
    ) -> None:
        if max_concurrency < 1:
            raise ValueError("max_concurrency must be at least 1.")
        self.api_key = "sb-local-" + secrets.token_urlsafe(24)
        # The CLIs run in an empty directory, like an API that has no files.
        self._workdir = tempfile.TemporaryDirectory(prefix="subbridge-gateway-")
        try:
            self._server = GatewayServer(
                port,
                api_key=self.api_key,
                max_concurrency=max_concurrency,
                turn_timeout=timeout,
                workdir=self._workdir.name,
            )
        except Exception:
            # For example the port is already in use: nothing to close yet,
            # but the empty workdir was already created and would otherwise
            # leak.
            self._workdir.cleanup()
            raise
        self.port: int = self._server.server_address[1]
        self._thread = threading.Thread(
            target=self._server.serve_forever, name="subbridge-gateway", daemon=True
        )
        self._thread.start()
        self._closed = False

    @property
    def anthropic_base_url(self) -> str:
        return f"http://{HOST}:{self.port}"

    @property
    def openai_base_url(self) -> str:
        return f"http://{HOST}:{self.port}/v1"

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        # Refuse any turn still queued for a slot before anything else, so
        # one cannot start a CLI after stop_in_flight_turns() has already
        # looked for turns to stop.
        self._server.begin_closing()
        self._server.shutdown()
        # Stop any turn still running before removing its working directory.
        self._server.stop_in_flight_turns()
        self._server.server_close()
        self._thread.join()
        self._workdir.cleanup()

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()


def serve(
    port: int = 0, max_concurrency: int = 4, timeout: float | None = 300
) -> Gateway:
    """Start the gateway in a background thread on 127.0.0.1.

    ``port=0`` picks a free port. At most ``max_concurrency`` CLI turns run at
    once; further requests wait up to ``timeout`` seconds for a slot, and each
    turn is stopped after ``timeout`` seconds.
    """
    return Gateway(port=port, max_concurrency=max_concurrency, timeout=timeout)


class _EnvGateway(Gateway):
    """A `Gateway` that also restores the four SDK env vars it set, once, on close."""

    _previous_env: dict[str, str | None]

    def close(self) -> None:
        already_closed = self._closed
        super().close()
        if already_closed:
            return
        for name, value in self._previous_env.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
        _clear_use_subscription_singleton(self)


_use_subscription_lock = threading.Lock()
_use_subscription_singleton: Gateway | None = None
_use_subscription_atexit_registered = False


def _clear_use_subscription_singleton(gateway: Gateway) -> None:
    global _use_subscription_singleton
    with _use_subscription_lock:
        if _use_subscription_singleton is gateway:
            _use_subscription_singleton = None


def _close_use_subscription_at_exit() -> None:
    with _use_subscription_lock:
        gateway = _use_subscription_singleton
    if gateway is not None:
        gateway.close()


def use_subscription(
    *, port: int = 0, max_concurrency: int = 4, timeout: float = 300
) -> Gateway:
    """Point the official Anthropic and OpenAI SDKs at your signed-in CLIs.

    Starts the gateway (like `serve()`) and sets `ANTHROPIC_BASE_URL`,
    `ANTHROPIC_API_KEY`, `OPENAI_BASE_URL`, and `OPENAI_API_KEY` in
    `os.environ`. Call this *before* constructing `Anthropic()` or
    `OpenAI()`: both SDKs read those four variables once, at construction,
    so a client built earlier keeps whatever endpoint and key it already
    had.

    This changes the whole process's environment, not just the calling
    thread or module, so any code, and any subprocess started afterwards
    that inherits the environment, picks up the gateway too.

    Idempotent and thread-safe: while a gateway from an earlier call is
    still running, a later call returns that same `Gateway` and changes
    nothing, even if given different arguments. Closing the returned
    `Gateway` -- with `close()`, a `with` block, or automatically at
    interpreter exit -- restores the four variables to whatever they held
    before this call (removing any that were unset), and the next call
    starts a fresh gateway with a new key.

    In prototype code::

        import subbridge as sb

        sb.use_subscription()
        from anthropic import Anthropic

        client = Anthropic()  # no base_url or api_key needed

    Delete the `sb.use_subscription()` line in production and set a real
    `ANTHROPIC_API_KEY` (or `OPENAI_API_KEY`) instead; nothing else about
    the code above needs to change.
    """
    global _use_subscription_singleton, _use_subscription_atexit_registered
    with _use_subscription_lock:
        if _use_subscription_singleton is not None:
            return _use_subscription_singleton
        warning = proxy_capture_warning()
        if warning:
            warnings.warn(warning, stacklevel=2)
        gateway = _EnvGateway(
            port=port, max_concurrency=max_concurrency, timeout=timeout
        )
        new_values = client_environment(gateway)
        gateway._previous_env = {name: os.environ.get(name) for name in new_values}
        os.environ.update(new_values)
        _use_subscription_singleton = gateway
        if not _use_subscription_atexit_registered:
            atexit.register(_close_use_subscription_at_exit)
            _use_subscription_atexit_registered = True
        return gateway
