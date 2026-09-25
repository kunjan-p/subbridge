"""Ask Claude Code one question with a chosen model and effort."""

from subbridge import ClaudeClient


def main() -> None:
    result = ClaudeClient().ask(
        "Explain forward kinematics in one paragraph.",
        model="sonnet",
        effort="medium",
    )
    print(result.text)
    print(result.usage)


if __name__ == "__main__":
    main()
