"""Stream raw Codex events as the turn runs."""

from subbridge import CodexClient


def main() -> None:
    thread = CodexClient().start_thread(
        model="gpt-5.3-codex",
        reasoning_effort="medium",
    )
    for event in thread.stream("Explain this repository."):
        print(event)


if __name__ == "__main__":
    main()
