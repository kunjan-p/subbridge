"""A quick, low-complexity prompt using Claude Code's Haiku alias."""

from subbridge import ClaudeClient


def main() -> None:
    result = ClaudeClient().ask(
        "In one sentence, explain what a Python virtual environment does.",
        model="haiku",
        timeout=90,
    )
    print(result.text)
    print("Usage:", result.usage)


if __name__ == "__main__":
    main()
