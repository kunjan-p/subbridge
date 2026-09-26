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
    """A `Gateway` that also restores the four SDK env vars it set, once, on close.

    The singleton hand-off and env restore happen first, under
    `_use_subscription_lock`, *before* the (possibly slow, up to several
    seconds while an in-flight CLI turn is stopped) real shutdown in
    `super().close()`. A `use_subscription()` call racing `close()` must
    never be handed back a gateway that has already committed to closing
    (I1): once `close()` has cleared the singleton under the lock, a
    racing `use_subscription()` finds it gone and starts a fresh one,
    rather than waiting out the slow shutdown only to get back a gateway
    that is (or is about to be) `_closed`.
    """

    _previous_env: dict[str, str | None]
    # A separate flag from the base class's `_closed`: that one gates the
    # real shutdown in `Gateway.close()` (so it must stay False until this
    # method actually calls `super().close()`), while this one guards the
    # env-restore/singleton bookkeeping so a second `close()` call -- even
    # one racing this one on another thread -- cannot run it twice.
    _env_restored: bool = False

    def close(self) -> None:
        global _use_subscription_singleton
        with _use_subscription_lock:
            if not self._env_restored:
                self._env_restored = True
                if _use_subscription_singleton is self:
                    _use_subscription_singleton = None
                # I2: restore a variable only if it still holds what this
                # call set; leave one the caller reassigned in between
                # alone, rather than overwriting it back.
                current = client_environment(self)
                for name, original in self._previous_env.items():
                    if os.environ.get(name) != current.get(name):
                        continue
                    if original is None:
                        os.environ.pop(name, None)
                    else:
                        os.environ[name] = original
        super().close()


_use_subscription_lock = threading.Lock()
_use_subscription_singleton: Gateway | None = None
_use_subscription_atexit_registered = False


def _close_use_subscription_at_exit() -> None:
    with _use_subscription_lock:
        gateway = _use_subscription_singleton
    if gateway is not None:
        gateway.close()


def use_subscription(
    *, port: int = 0, max_concurrency: int = 4, timeout: float | None = 300
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
    that inherits the environment, picks up the gateway too. The gateway
    itself is shared the same way: it is one gateway for the whole
    process, not one per caller, so closing it -- with `close()`, or by
    leaving a `with subbridge.use_subscription():` block -- closes it for
    every other piece of code in the process that is still relying on it,
    not just the caller that closed it.

    Idempotent and thread-safe: while a gateway from an earlier call is
    still running, a later call (even from another thread, even with
    different arguments) returns that same `Gateway` and changes nothing.
    Closing the returned `Gateway` restores the four variables, but only
    the ones still holding what this call set: a variable your own code
    reassigned in between (``os.environ["OPENAI_API_KEY"] = "sk-..."``,
    say) is left as your code set it, not overwritten back. The next
    `use_subscription()` call after a close starts a fresh gateway with a
    new key.

    Registers an `atexit` cleanup once, the first time this is called, so
    the gateway stops (and any in-flight CLI turn is killed) at normal
    interpreter exit even if nothing ever calls `close()`. `atexit` runs
    callbacks in reverse registration order, so anything that registered
    its own cleanup before this function was first called runs *after*
    this one has already closed the gateway; and some process exits (a
    hard kill, or a daemon thread -- like the gateway's own server thread
    -- still running when the interpreter would otherwise exit) skip
    `atexit` entirely, so this is a safety net, not a guarantee.

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
