from __future__ import annotations

import json
import os
import shutil
import subprocess
from typing import Any

from ._codex_thread import CodexThread, ThreadOptions
from ._errors import (
    CodexNotAuthenticatedError,
    CodexNotInstalledError,
    CodexWrongAuthModeError,
)
from ._models import CodexStatus, ModelInfo, ProviderCapabilities


class CodexClient:
    """
    Python wrapper around the locally installed Codex CLI.

    By default this client is subscription-only:
    - ChatGPT login is allowed.
    - API-key auth is rejected.
    - OPENAI_API_KEY, CODEX_API_KEY and OPENAI_BASE_URL are
      removed from the child environment.

    The user's normal Codex/ChatGPT local authentication
    remains available through HOME / CODEX_HOME.
    """

    def __init__(
        self,
        codex_path: str | None = None,
        *,
        subscription_only: bool = True,
        env: dict[str, str] | None = None,
        include_raw_diagnostics: bool = False,
    ) -> None:
        self.codex_path = codex_path or shutil.which("codex")

        self.subscription_only = subscription_only
        self._custom_env = env
        self.include_raw_diagnostics = include_raw_diagnostics

    def _environment(self) -> dict[str, str]:

        if self._custom_env is not None:
            env = dict(self._custom_env)
        else:
            env = dict(os.environ)

        if self.subscription_only:
            # Important:
            # do not allow our experiment to accidentally use
            # normal API authentication.
            env.pop("OPENAI_API_KEY", None)
            env.pop("CODEX_API_KEY", None)
            env.pop("OPENAI_BASE_URL", None)

        return env

    def installed(self) -> bool:
        return self.codex_path is not None

    def status(self, *, include_raw: bool = False) -> CodexStatus:

        if not self.codex_path:
            return CodexStatus(
                installed=False,
                authenticated=False,
            )

        env = self._environment()
        version = None
        error = None
        try:
            version_result = subprocess.run(
                [self.codex_path, "--version"],
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
            error = "Codex CLI version query failed."
        try:
            status_result = subprocess.run(
                [self.codex_path, "login", "status"],
                capture_output=True,
                text=True,
                env=env,
                check=False,
                timeout=5,
            )
            raw = status_result.stdout.strip() or status_result.stderr.strip()
        except (OSError, subprocess.SubprocessError):
            return CodexStatus(
                installed=True,
                authenticated=False,
                version=version,
                executable=self.codex_path,
                error="Codex CLI auth status query failed.",
            )

        lower = raw.lower()

        authenticated = False
        auth_mode = None

        if "logged in using chatgpt" in lower:
            authenticated = True
            auth_mode = "chatgpt"

        elif "logged in using an api key" in lower:
            authenticated = True
            auth_mode = "api_key"

        elif "logged in using agent identity" in lower:
            authenticated = True
            auth_mode = "agent_identity"

        elif "not logged in" in lower:
            authenticated = False
        elif status_result.returncode != 0:
            error = error or "Codex CLI auth status query failed."
        elif not raw:
            error = error or "Codex CLI returned an unrecognized auth status."

        return CodexStatus(
            installed=True,
            authenticated=authenticated,
            auth_mode=auth_mode,
            version=version,
            executable=self.codex_path,
            raw_status=raw if include_raw else None,
            error=error,
        )

    def capabilities(self) -> ProviderCapabilities:
        """Report local CLI/auth details and the CLI-known model catalog.

        The Codex catalog is not an account entitlement or billing check.
        """
        status = self.status()
        models: tuple[ModelInfo, ...] | None = None
        source: str | None = None
        if status.installed and self.codex_path:
            try:
                catalog = subprocess.run(
                    [self.codex_path, "debug", "models"],
                    capture_output=True,
                    text=True,
                    env=self._environment(),
                    check=False,
                    timeout=5,
                )
                if catalog.returncode != 0:
                    raise ValueError("Codex model catalog command failed")
                payload = json.loads(catalog.stdout)
                entries = payload.get("models", [])
                models = tuple(
                    ModelInfo(
                        id=entry["slug"],
                        display_name=entry.get("display_name"),
                        reasoning_efforts=tuple(
                            level.get("effort", "")
                            for level in entry.get("supported_reasoning_levels", [])
                            if level.get("effort")
                        ),
                        is_default=entry.get("priority") == 1,
                    )
                    for entry in entries
                    if isinstance(entry, dict)
                    and isinstance(entry.get("slug"), str)
                    and entry.get("visibility", "list") == "list"
                )
                source = "codex debug models"
            except (
                OSError,
                subprocess.SubprocessError,
                json.JSONDecodeError,
                AttributeError,
                TypeError,
                KeyError,
                ValueError,
            ):
                models = None
                source = None
        return ProviderCapabilities(
            provider="codex",
            installed=status.installed,
            authenticated=status.authenticated,
            auth_mode=status.auth_mode,
            version=status.version,
            executable=status.executable,
            models=models,
            model_catalog_source=source,
            model_access_verified=False,
            account_plan=None,
            plan_source=None,
            plan_allowed_models=None,
            usage_available=False,
            usage_source=None,
            error=status.error,
        )

    def _validate_auth(self) -> CodexStatus:

        status = self.status()

        if not status.installed:
            raise CodexNotInstalledError(
                "Codex CLI is not installed or is not in PATH."
            )

        if not status.authenticated:
            raise CodexNotAuthenticatedError(
                "Codex is not authenticated. Run: codex login"
            )

        if self.subscription_only and status.auth_mode != "chatgpt":
            raise CodexWrongAuthModeError(
                "SubBridge is running in subscription-only mode, "
                f"but Codex auth mode is {status.auth_mode!r}. "
                "Authenticate using ChatGPT with `codex login`."
            )

        return status

    def start_thread(
        self,
        **kwargs: Any,
    ) -> CodexThread:

        options = ThreadOptions(**kwargs)

        return CodexThread(
            client=self,
            thread_id=None,
            options=options,
        )
