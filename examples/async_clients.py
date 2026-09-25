"""Run independent provider requests concurrently inside an asyncio app."""

import asyncio

from subbridge import ClaudeClient, CodexClient


async def main() -> None:
    claude_result, codex_result = await asyncio.gather(
        ClaudeClient().ask_async(
            "Give one short definition of an idempotent operation.",
            model="haiku",
            timeout=90,
        ),
        CodexClient().ask_async(
            "Give one short definition of an idempotent operation.",
            model="gpt-6-luna",
            reasoning_effort="low",
            timeout=90,
        ),
    )
    for result in (claude_result, codex_result):
        print(f"[{result.provider} / {result.model}] {result.text}")
        print("Usage:", result.usage)


if __name__ == "__main__":
    asyncio.run(main())
