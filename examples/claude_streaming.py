"""Stream raw Claude Code events, then show the session ID for resuming."""

from subbridge import ClaudeClient


def main() -> None:
    thread = ClaudeClient().start_thread(model="sonnet")
    for event in thread.stream("Explain this repository in one paragraph."):
        print(event)
    print("Session:", thread.id)


if __name__ == "__main__":
    main()
