"""Map SubBridge failures to HTTP errors in each API's error shape."""

from __future__ import annotations

import subprocess
from typing import Any, Literal

from .._process import failure_kind
from ..errors import (
    ClaudeNotAuthenticatedError,
    ClaudeNotInstalledError,
    ClaudeProcessError,
    ClaudeProtocolError,
    ClaudeTurnError,
    ClaudeWrongAuthModeError,
    CodexNotAuthenticatedError,
    CodexNotInstalledError,
    CodexProcessError,
    CodexProtocolError,
    CodexTurnError,
    CodexWrongAuthModeError,
    SubBridgeError,
)

ErrorStyle = Literal["anthropic", "openai"]

_UNAVAILABLE = (
    ClaudeNotInstalledError,
    ClaudeNotAuthenticatedError,
    ClaudeWrongAuthModeError,
    CodexNotInstalledError,
    CodexNotAuthenticatedError,
    CodexWrongAuthModeError,
)
_FAILED_RUN = (ClaudeTurnError, CodexTurnError, ClaudeProcessError, CodexProcessError)
_PROTOCOL = (ClaudeProtocolError, CodexProtocolError)
_FIXED_STATUS = ((_UNAVAILABLE, 503), (_PROTOCOL, 502))
_KIND_STATUS = {
    "rate_limit": (429, "rate_limit_exceeded"),
    "model": (404, "model_not_found"),
    "plan": (403, "model_not_available"),
}
_ANTHROPIC_TYPES = {
    400: "invalid_request_error",
    401: "authentication_error",
    403: "permission_error",
    404: "not_found_error",
    413: "request_too_large",
    429: "rate_limit_error",
    504: "timeout_error",
}


class GatewayError(SubBridgeError):
    """An HTTP error the gateway sends back in the caller's API shape."""

    def __init__(
        self,
        status: int,
        message: str,
        *,
        param: str | None = None,
        code: str | None = None,
    ) -> None:
        super().__init__(message)
        self.status = status
        self.message = message
        self.param = param
        self.code = code


def from_exception(exc: SubBridgeError) -> GatewayError:
    if isinstance(exc, GatewayError):
        return exc
    if isinstance(exc, _FAILED_RUN):
        return _failed_run(exc)
    for types, status in _FIXED_STATUS:
        if isinstance(exc, types):
            return GatewayError(status, str(exc))
    return GatewayError(
        500, "The SubBridge gateway failed while handling this request."
    )


def _failed_run(exc: SubBridgeError) -> GatewayError:
    if isinstance(exc.__cause__, (subprocess.TimeoutExpired, TimeoutError)):
        return GatewayError(504, str(exc))
    status, code = _KIND_STATUS.get(failure_kind(str(exc)) or "", (502, None))
    return GatewayError(status, str(exc), code=code)


def error_body(style: ErrorStyle, error: GatewayError) -> dict[str, Any]:
    if style == "anthropic":
        kind = _ANTHROPIC_TYPES.get(error.status, "api_error")
        return {"type": "error", "error": {"type": kind, "message": error.message}}
    return {"error": openai_error(error)}


def openai_error(error: GatewayError) -> dict[str, Any]:
    if error.status == 429:
        kind = "rate_limit_exceeded"
    else:
        kind = "invalid_request_error" if error.status < 500 else "server_error"
    return {
        "message": error.message,
        "type": kind,
        "param": error.param,
        "code": error.code,
    }
