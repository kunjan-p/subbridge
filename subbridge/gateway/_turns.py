"""Types shared by the gateway's server and its API endpoints."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from ..models import ProviderName, TurnResult
from ._errors import ErrorStyle


@dataclass
class TurnRequest:
    """One API request, reduced to what the CLI needs."""

    prompt: str
    model: str | None = None
    stream: bool = False
    output_schema: dict[str, Any] | None = None


class Endpoint(Protocol):
    """One API route: it parses its requests and renders replies in its shape."""

    provider: ProviderName
    style: ErrorStyle

    def parse(self, body: dict[str, Any]) -> TurnRequest: ...

    def respond(self, result: TurnResult) -> dict[str, Any]: ...
