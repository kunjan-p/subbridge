from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

ProviderName = Literal["claude", "codex"]


@dataclass
class CodexStatus:
    installed: bool
    authenticated: bool
    auth_mode: str | None = None
    version: str | None = None
    executable: str | None = None
    # Raw CLI status can contain account identifiers. It is populated only
    # when callers explicitly request include_raw=True.
    raw_status: str | None = None
    error: str | None = None


@dataclass
class ClaudeStatus:
    installed: bool
    authenticated: bool
    auth_mode: str | None = None
    version: str | None = None
    executable: str | None = None
    raw_status: str | None = None
    error: str | None = None
    account_plan: str | None = None


@dataclass(frozen=True)
class ModelInfo:
    """A model advertised by a local CLI, not an account entitlement."""

    id: str
    display_name: str | None = None
    reasoning_efforts: tuple[str, ...] = ()
    is_default: bool = False


@dataclass(frozen=True)
class ProviderCapabilities:
    """Non-generative preflight information reported by an installed CLI.

    `models` is None when the CLI has no queryable local catalog. A catalog
    describes CLI-known models and does not prove account access or billing.
    """

    provider: ProviderName
    installed: bool
    authenticated: bool
    auth_mode: str | None = None
    version: str | None = None
    executable: str | None = None
    models: tuple[ModelInfo, ...] | None = None
    model_catalog_source: str | None = None
    model_access_verified: bool = False
    account_plan: str | None = None
    plan_source: str | None = None
    plan_allowed_models: tuple[str, ...] | None = None
    usage_available: bool = False
    usage_source: str | None = None
    error: str | None = None


@dataclass
class StreamEvent:
    """Provider-neutral event; provider-specific payload is opt-in."""

    provider: ProviderName
    kind: Literal["thread_started", "message", "turn_completed", "turn_error", "event"]
    provider_event: str
    thread_id: str | None = None
    text: str | None = None
    usage: Usage | None = None
    raw: dict[str, Any] | None = None


@dataclass
class Usage:
    input_tokens: int = 0
    cached_input_tokens: int = 0
    cache_write_input_tokens: int = 0
    output_tokens: int = 0
    reasoning_output_tokens: int = 0


@dataclass
class TurnResult:
    text: str
    thread_id: str | None
    usage: Usage | None
    items: list[dict[str, Any]] = field(default_factory=list)
    events: list[dict[str, Any]] = field(default_factory=list)
    provider: ProviderName | None = None
    model: str | None = None
    elapsed_seconds: float | None = None
    structured_output: Any | None = None
