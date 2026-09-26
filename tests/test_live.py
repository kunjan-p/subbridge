"""Opt-in live compatibility smoke tests for locally signed-in CLIs.

These go through the product: a real gateway, driven by the official
Anthropic and OpenAI SDKs, exactly as a prototype would use it.
"""

import importlib.util
import os
import unittest

# The release workflow installs only requirements-release.txt, so skip
# cleanly here instead of failing to collect this module.
if (
    importlib.util.find_spec("anthropic") is None
    or importlib.util.find_spec("openai") is None
):
    raise unittest.SkipTest(
        "anthropic and openai are not installed; skipping live tests."
    )

from anthropic import Anthropic
from openai import OpenAI

import subbridge


@unittest.skipUnless(
    os.environ.get("SUBBRIDGE_RUN_LIVE_TESTS") == "1",
    "set SUBBRIDGE_RUN_LIVE_TESTS=1 to use signed-in CLIs",
)
class LiveGatewaySmokeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.gateway = subbridge.serve()
        self.addCleanup(self.gateway.close)

    def test_claude_code_live_smoke_via_anthropic_sdk(self) -> None:
        client = Anthropic(
            base_url=self.gateway.anthropic_base_url,
            api_key=self.gateway.api_key,
            max_retries=0,
        )
        model = os.environ.get("SUBBRIDGE_CLAUDE_MODEL", "sonnet")
        message = client.messages.create(
            model=model,
            max_tokens=64,
            messages=[
                {
                    "role": "user",
                    "content": (
                        "Reply with one short sentence confirming this local "
                        "CLI smoke test."
                    ),
                }
            ],
        )
        text = "".join(block.text for block in message.content if block.type == "text")
        self.assertTrue(text.strip())

    def test_codex_live_smoke_via_openai_sdk(self) -> None:
        client = OpenAI(
            base_url=self.gateway.openai_base_url,
            api_key=self.gateway.api_key,
            max_retries=0,
        )
        model = os.environ.get("SUBBRIDGE_CODEX_MODEL", "gpt-6-luna")
        response = client.responses.create(
            model=model,
            input=(
                "Reply with one short sentence confirming this local CLI smoke test."
            ),
        )
        self.assertTrue(response.output_text.strip())


if __name__ == "__main__":
    unittest.main()
