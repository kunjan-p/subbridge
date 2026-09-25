"""Keep context across turns by reusing one Codex thread."""

from subbridge import CodexClient


def main() -> None:
    thread = CodexClient().start_thread(
        model="gpt-5.3-codex",
        reasoning_effort="medium",
    )

    first = thread.run("My project is a CNC controller.")
    print(first.text)
    print("Thread:", thread.id)

    second = thread.run("What project did I just tell you about?")
    print(second.text)


if __name__ == "__main__":
    main()
