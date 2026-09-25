"""A localhost gateway that lets the official Anthropic and OpenAI SDKs use the CLIs."""

from __future__ import annotations

import secrets
import tempfile
import threading
from types import TracebackType
from typing import Self

from ._server import HOST, GatewayServer

__all__ = ["Gateway", "serve"]


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
        self._server = GatewayServer(
            port,
            api_key=self.api_key,
            max_concurrency=max_concurrency,
            turn_timeout=timeout,
            workdir=self._workdir.name,
        )
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
        # Stop any turn still running before removing its working directory,
        # and before joining the serving thread waits on nothing else.
        self._server.stop_in_flight_turns()
        self._server.shutdown()
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
