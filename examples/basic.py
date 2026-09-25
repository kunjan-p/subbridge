"""Check Codex sign-in status, then ask one question."""

from subbridge import CodexClient


def main() -> None:
    codex = CodexClient(subscription_only=True)
    print(codex.status())

    response = codex.ask("What is 17 * 23? Return only the number.")
    print(response.text)
    print(response.usage)


if __name__ == "__main__":
    main()
