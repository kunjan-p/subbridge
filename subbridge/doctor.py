"""Local, non-generative diagnostics for SubBridge provider CLIs."""

from __future__ import annotations

import argparse
import json

from .claude import ClaudeClient
from .codex import CodexClient


def collect_doctor_report() -> dict[str, dict]:
    """Return CLI/auth/catalog facts without sending model requests."""
    report = {}
    for name, client in (("claude", ClaudeClient()), ("codex", CodexClient())):
        capability = client.capabilities()
        report[name] = {
            "installed": capability.installed,
            "authenticated": capability.authenticated,
            "auth_mode": capability.auth_mode,
            "version": capability.version,
            "account_plan": capability.account_plan,
            "plan_source": capability.plan_source,
            "cli_known_models": [model.id for model in capability.models]
            if capability.models is not None
            else None,
            "plan_allowed_models": list(capability.plan_allowed_models)
            if capability.plan_allowed_models is not None
            else None,
            "model_access_verified": capability.model_access_verified,
            "usage_available": capability.usage_available,
            "usage_source": capability.usage_source,
            "error": capability.error,
            "note": "Plan entitlement and current quota are not verified by this non-generative check.",
        }
    return report


def main() -> int:
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "doctor":
        sys.argv.pop(1)
    parser = argparse.ArgumentParser(
        prog="subbridge doctor",
        description="Check local provider CLI, login, model catalog, and plan/usage visibility.",
    )
    parser.add_argument(
        "--json", action="store_true", help="Print machine-readable JSON."
    )
    args = parser.parse_args()
    report = collect_doctor_report()
    if args.json:
        print(json.dumps(report, indent=2))
        return (
            0
            if all(
                info["installed"] and info["authenticated"] for info in report.values()
            )
            else 1
        )
    for name, info in report.items():
        print(f"{name.title()}")
        print(
            f"  CLI: {'installed' if info['installed'] else 'not found'}"
            + (f" ({info['version']})" if info["version"] else "")
        )
        print(
            f"  Login: {info['auth_mode'] or ('authenticated' if info['authenticated'] else 'not authenticated')}"
        )
        print(f"  Plan: {info['account_plan'] or 'unknown (not exposed by CLI)'}")
        models = info["cli_known_models"]
        print(
            f"  CLI-known models: {', '.join(models) if models else ('unavailable' if models is None else 'none reported')}"
        )
        allowed = info["plan_allowed_models"]
        print(
            f"  Plan-allowed models: {', '.join(allowed) if allowed else 'unknown (not verified; use provider model picker or an explicit request)'}"
        )
        print(
            f"  Current usage/quota: {'available' if info['usage_available'] else 'unknown (not exposed by CLI)'}"
        )
        if info["error"]:
            print(f"  Diagnostic: {info['error']}")
    return (
        0
        if all(info["installed"] and info["authenticated"] for info in report.values())
        else 1
    )


if __name__ == "__main__":
    raise SystemExit(main())
