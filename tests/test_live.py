"""Opt-in live compatibility smoke tests for locally signed-in CLIs."""

import asyncio
import os
import unittest

from subbridge import ClaudeClient, CodexClient


@unittest.skipUnless(
    os.environ.get("SUBBRIDGE_RUN_LIVE_TESTS") == "1",
    "set SUBBRIDGE_RUN_LIVE_TESTS=1 to use signed-in CLIs",
)
class LiveCliSmokeTests(unittest.TestCase):
    def test_claude_code_live_smoke(self) -> None:
        client = ClaudeClient()
        capabilities = client.capabilities()
        self.assertTrue(capabilities.installed, "Claude Code CLI is not installed")
        self.assertTrue(capabilities.authenticated, "Claude Code is not signed in")
        self.assertEqual(
            capabilities.auth_mode,
            "claude.ai",
            "subscription-only Claude sign-in required",
        )
        result = asyncio.run(
            client.ask_async(
                "Reply with one short sentence confirming this local CLI smoke test.",
                model=os.environ.get("SUBBRIDGE_CLAUDE_MODEL", "haiku"),
                timeout=90,
            )
        )
        self.assertTrue(result.text.strip())
        self.assertEqual(result.provider, "claude")
        self.assertTrue(result.thread_id)

    def test_codex_live_smoke(self) -> None:
        client = CodexClient()
        capabilities = client.capabilities()
        self.assertTrue(capabilities.installed, "Codex CLI is not installed")
        self.assertTrue(capabilities.authenticated, "Codex CLI is not signed in")
        self.assertEqual(
            capabilities.auth_mode,
            "chatgpt",
            "subscription-only ChatGPT sign-in required",
        )
        model = os.environ.get("SUBBRIDGE_CODEX_MODEL", "gpt-6-luna")
        result = asyncio.run(
            client.ask_async(
                "Reply with one short sentence confirming this local CLI smoke test.",
                model=model,
                reasoning_effort="low",
                timeout=90,
            )
        )
        self.assertTrue(result.text.strip())
        self.assertEqual(result.provider, "codex")
        self.assertTrue(result.thread_id)
