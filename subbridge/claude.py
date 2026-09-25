from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any

from ._claude_thread import (
    ClaudePermissionMode,
    ClaudeThread,
    ClaudeThreadOptions,
)
from ._process import validate_thread_id
from .errors import (
    ClaudeNotAuthenticatedError,
    ClaudeNotInstalledError,
    ClaudeProcessError,
    ClaudeProtocolError,
    ClaudeTurnError,
    ClaudeWrongAuthModeError,
)
from .models import ClaudeStatus, ProviderCapabilities, TurnResult

# Re-exported so pre-split import paths keep working.
__all__ = [
    "ClaudeClient",
    "ClaudePermissionMode",
    "ClaudeProcessError",
    "ClaudeProtocolError",
    "ClaudeThread",
    "ClaudeThreadOptions",
    "ClaudeTurnError",
]


class ClaudeClient:
    """Python wrapper around the locally installed Claude Code CLI."""

    def __init__(
        self,
        claude_path: str | None = None,
        *,
        subscription_only: bool = True,
        env: dict[str, str] | None = None,
        include_raw_diagnostics: bool = False,
    ) -> None:
        self.claude_path = claude_path or shutil.which("claude")
        self.subscription_only = subscription_only
        self._custom_env = env
        self.include_raw_diagnostics = include_raw_diagnostics

    def _environment(self) -> dict[str, str]:
        env = (
            dict(self._custom_env) if self._custom_env is not None else dict(os.environ)
        )
        if self.subscription_only:
            env.pop("ANTHROPIC_API_KEY", None)
            env.pop("CLAUDE_CODE_OAUTH_TOKEN", None)
            env.pop("ANTHROPIC_AUTH_TOKEN", None)
            # A custom endpoint passes the claude.ai auth check but routes
            # requests (and the login token) to another host.
            env.pop("ANTHROPIC_BASE_URL", None)
        return env

    def installed(self) -> bool:
        return self.claude_path is not None

    def status(self, *, include_raw: bool = False) -> ClaudeStatus:
        if not self.claude_path:
            return ClaudeStatus(installed=False, authenticated=False)

        env = self._environment()
        version = None
        error = None
        try:
            version_result = subprocess.run(
                [self.claude_path, "--version"],
                capture_output=True,
                text=True,
                env=env,
                check=False,
                timeout=5,
            )
            version = (
                version_result.stdout.strip() or version_result.stderr.strip() or None
            )
        except (OSError, subprocess.SubprocessError):
            error = "Claude Code version query failed."
        try:
            status_result = subprocess.run(
                [self.claude_path, "auth", "status", "--json"],
                capture_output=True,
                text=True,
                env=env,
                check=False,
                timeout=5,
            )
            raw = status_result.stdout.strip() or status_result.stderr.strip()
        except (OSError, subprocess.SubprocessError):
            return ClaudeStatus(
                installed=True,
                authenticated=False,
                version=version,
                executable=self.claude_path,
                raw_status=None,
                error="Claude Code auth status query failed.",
            )
        try:
            payload = json.loads(status_result.stdout)
        except (json.JSONDecodeError, TypeError):
            payload = {}
            error = error or "Claude Code auth status returned invalid JSON."
        if not isinstance(payload, dict):
            payload = {}
            error = error or "Claude Code auth status returned an unexpected shape."
        if status_result.returncode != 0:
            error = error or "Claude Code auth status command failed."
        auth_mode = payload.get("authMethod")
        authenticated = payload.get("loggedIn") is True
        return ClaudeStatus(
            installed=True,
            authenticated=authenticated,
            auth_mode=auth_mode,
            account_plan=payload.get("subscriptionType")
            if isinstance(payload.get("subscriptionType"), str)
            else None,
            version=version,
            executable=self.claude_path,
            raw_status=raw if include_raw else None,
            error=error,
        )

    def capabilities(self) -> ProviderCapabilities:
        """Inspect the local CLI and auth state without making a model request.

        Claude Code exposes model selection but no stable non-interactive model
        catalog command, so this method deliberately reports ``models=None``.
        """
        status = self.status()
        return ProviderCapabilities(
            provider="claude",
            installed=status.installed,
            authenticated=status.authenticated,
            auth_mode=status.auth_mode,
            account_plan=status.account_plan,
            plan_source="claude auth status --json" if status.account_plan else None,
            version=status.version,
            executable=status.executable,
            models=None,
            model_catalog_source=None,
            model_access_verified=False,
            plan_allowed_models=None,
            usage_available=False,
            usage_source=None,
            error=status.error,
        )

    def _validate_auth(self) -> ClaudeStatus:
        status = self.status()
        if not status.installed:
            raise ClaudeNotInstalledError(
                "Claude Code CLI is not installed or is not in PATH."
            )
        if not status.authenticated:
            raise ClaudeNotAuthenticatedError(
                "Claude Code is not authenticated. Run `claude auth login` or `/login`."
            )
        if self.subscription_only and status.auth_mode != "claude.ai":
            raise ClaudeWrongAuthModeError(
                "Subscription-only mode requires Claude.ai authentication; "
                f"Claude Code reported auth mode {status.auth_mode!r}."
            )
        return status

    def start_thread(self, **kwargs: Any) -> ClaudeThread:
        return ClaudeThread(self, None, ClaudeThreadOptions(**kwargs))

    def resume_thread(self, thread_id: str, **kwargs: Any) -> ClaudeThread:
        return ClaudeThread(
            self, validate_thread_id(thread_id), ClaudeThreadOptions(**kwargs)
        )

    def ask(
        self,
        prompt: str,
        *,
        model: str | None = None,
        effort: str | None = None,
        cwd: str | Path | None = None,
        additional_directories: list[str | Path] | None = None,
        permission_mode: ClaudePermissionMode = "read-only",
        output_schema: dict[str, Any] | None = None,
        timeout: float | None = 300,
        include_events: bool = False,
    ) -> TurnResult:
        thread = self.start_thread(
            model=model,
            effort=effort,
            cwd=cwd,
            additional_directories=additional_directories,
            permission_mode=permission_mode,
        )
        return thread.run(
            prompt,
            output_schema=output_schema,
            timeout=timeout,
            include_events=include_events,
        )

    async def ask_async(
        self,
        prompt: str,
        *,
        model: str | None = None,
        effort: str | None = None,
        cwd: str | Path | None = None,
        additional_directories: list[str | Path] | None = None,
        permission_mode: ClaudePermissionMode = "read-only",
        output_schema: dict[str, Any] | None = None,
        timeout: float | None = 300,
        include_events: bool = False,
    ) -> TurnResult:
        thread = self.start_thread(
            model=model,
            effort=effort,
            cwd=cwd,
            additional_directories=additional_directories,
            permission_mode=permission_mode,
        )
        return await thread.run_async(
            prompt,
            output_schema=output_schema,
            timeout=timeout,
            include_events=include_events,
        )
