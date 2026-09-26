"""The official anthropic and openai SDKs, pointed at SubBridge's gateway.

Prototype through your signed-in CLIs:

    subbridge run -- python examples/official_sdks.py

In production, run the same file with real ANTHROPIC_API_KEY and OPENAI_API_KEY
values, and set CLAUDE_MODEL and OPENAI_MODEL to model IDs the APIs accept.
"""

import os

from anthropic import Anthropic
from openai import OpenAI

QUESTION = "What is 17 * 23? Return only the number."


def main() -> None:
    claude = Anthropic().messages.create(
        model=os.environ.get("CLAUDE_MODEL", "haiku"),
        max_tokens=100,
        messages=[{"role": "user", "content": QUESTION}],
    )
    print("Claude:", claude.content[0].text)
    codex = OpenAI().responses.create(
        model=os.environ.get("OPENAI_MODEL", "gpt-6-luna"), input=QUESTION
    )
    print("Codex:", codex.output_text)


if __name__ == "__main__":
    main()
