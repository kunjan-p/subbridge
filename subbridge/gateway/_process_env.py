"""The four SDK environment variables, and the shared proxy-capture warning.

Kept separate from `gateway/__init__.py` so `subbridge.cli` and
`use_subscription()` (also in `gateway/__init__.py`) share one
implementation of each, rather than two copies drifting apart.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from . import Gateway

PROXY_VARIABLES = ("http_proxy", "https_proxy", "all_proxy")
PROXY_NOTE = (
    "If HTTP_PROXY, HTTPS_PROXY, or ALL_PROXY is set, add 127.0.0.1 to "
    "NO_PROXY too, or the SDKs may send the gateway key and your prompts to "
    "that proxy instead of the gateway."
)


def proxy_capture_warning() -> str | None:
    """A one-line warning when a proxy could intercept the gateway's own traffic.

    The official SDKs treat the gateway's `127.0.0.1` base URL like any other
    HTTP endpoint, so `HTTP_PROXY`, `HTTPS_PROXY`, or `ALL_PROXY` (either
    case) still applies to it unless `NO_PROXY`/`no_proxy` exempts
    `127.0.0.1`.
    """
    active = any(
        os.environ.get(name.upper()) or os.environ.get(name) for name in PROXY_VARIABLES
    )
    if not active:
        return None
    no_proxy = os.environ.get("NO_PROXY") or os.environ.get("no_proxy") or ""
    exempted = {entry.strip() for entry in no_proxy.split(",")}
    if "127.0.0.1" in exempted:
        return None
    return f"subbridge: {PROXY_NOTE}"


def client_environment(gateway: Gateway) -> dict[str, str]:
    """The four variables the official SDKs read for their endpoint and key."""
    return {
        "ANTHROPIC_BASE_URL": gateway.anthropic_base_url,
        "ANTHROPIC_API_KEY": gateway.api_key,
        "OPENAI_BASE_URL": gateway.openai_base_url,
        "OPENAI_API_KEY": gateway.api_key,
    }
