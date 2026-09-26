"""The official anthropic and openai SDKs, pointed at your signed-in CLIs.

Prototype without running `subbridge serve` or `subbridge run` yourself:

    python examples/use_subscription.py

In production, delete the `subbridge.use_subscription()` line below, set
real ANTHROPIC_API_KEY and OPENAI_API_KEY values, and set CLAUDE_MODEL and
OPENAI_MODEL to model IDs those APIs accept.
"""

import os

import subbridge
from anthropic import Anthropic
from openai import OpenAI

QUESTION = "What is 17 * 23? Return only the number."


def main() -> None:
    subbridge.use_subscription()  # Deleted in production.
    claude = Anthropic().messages.create(
        model=os.environ.get("CLAUDE_MODEL", "sonnet"),
        max_tokens=100,
        messages=[{"role": "user", "content": QUESTION}],
    )
    print("Claude:", claude.content[0].text)
    codex = OpenAI().chat.completions.create(
        model=os.environ.get("OPENAI_MODEL", "gpt-6-luna"),
        messages=[{"role": "user", "content": QUESTION}],
    )
    print("Codex:", codex.choices[0].message.content)


if __name__ == "__main__":
    main()
