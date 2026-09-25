# Security policy

## Supported versions

SubBridge is in alpha. Only the latest release on [PyPI](https://pypi.org/project/subbridge/) receives security fixes, so upgrade before reporting a problem.

| Version | Supported |
| --- | --- |
| 0.1.x (latest alpha) | Yes |
| Anything older | No |

## Reporting a vulnerability

Please report vulnerabilities privately through GitHub: open the repository's [Security tab](https://github.com/kunjan-p/subbridge/security) and choose "Report a vulnerability". Do not open a public issue, pull request, or discussion for a security problem.

Include the SubBridge version, your Python version and operating system, the `claude` or `codex` CLI version, and the steps or code that reproduce the problem.

You will get an acknowledgement on the report, and it stays private while a fix is prepared. Once a fixed release is on PyPI, the advisory is published with credit to you unless you prefer to stay anonymous. If the report turns out not to be a vulnerability, you will get an explanation of why.

## What counts as a vulnerability

SubBridge's security promises are its defaults, so a way around any of them is in scope:

- Subscription-only mode letting a request use an API key, an endpoint override, or a non-subscription sign-in.
- A run escaping the Codex `read-only` sandbox or the Claude Code `read-only` mode (Read, Glob, and Grep only) that SubBridge requested.
- Raw CLI status, events, or stderr appearing in results or error messages without the matching opt-in (`include_raw`, `include_events`, or `include_raw_diagnostics`).
- A CLI process that keeps running after a timeout, error, or cancellation.
- Input that makes SubBridge pass unintended flags or arguments to a CLI.

Vulnerabilities in the `claude` or `codex` CLIs themselves belong with [Anthropic](https://www.anthropic.com/responsible-disclosure-policy) or [OpenAI](https://openai.com/security/disclosure/).
