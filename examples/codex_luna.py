"""A concise task using the efficient GPT-6 Luna model in Codex."""

from subbridge import CodexClient


def main() -> None:
    result = CodexClient().ask(
        "Write a Python function called slugify that lowercases text and "
        "replaces runs of spaces with hyphens. Return code only.",
        model="gpt-6-luna",
        reasoning_effort="low",
        timeout=90,
    )
    print(result.text)
    print("Usage:", result.usage)


if __name__ == "__main__":
    main()
