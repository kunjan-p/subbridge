# Security policy

## Supported versions

SubBridge is in alpha. Only the latest release on [PyPI](https://pypi.org/project/subbridge/) receives security fixes, so upgrade before reporting a problem.

| Version | Supported |
| --- | --- |
| 0.2.x (latest alpha) | Yes |
| Anything older | No |

## Reporting a vulnerability

Please report vulnerabilities privately through GitHub: open the repository's [Security tab](https://github.com/kunjan-p/subbridge/security) and choose "Report a vulnerability". Do not open a public issue, pull request, or discussion for a security problem.

Include the SubBridge version, your Python version and operating system, the `claude` or `codex` CLI version, and the steps or code that reproduce the problem.

You will get an acknowledgement on the report, and it stays private while a fix is prepared. Once a fixed release is on PyPI, the advisory is published with credit to you unless you prefer to stay anonymous. If the report turns out not to be a vulnerability, you will get an explanation of why.

## What counts as a vulnerability

SubBridge's security promises are its defaults, so a way around any of them is in scope:

- Subscription-only mode letting a gateway request use an API key, an endpoint override, or a non-subscription sign-in.
- A run escaping the Codex `read-only` sandbox or the Claude Code `read-only` mode (Read, Glob, and Grep only) that SubBridge always requests.
- The CLI's stderr, raw event stream, or raw sign-in status appearing in a gateway response, a gateway error message, or `subbridge doctor` output. The error message the CLI reports for a failed turn, and its version string, are shown by design.
- A CLI process that keeps running after a timeout, error, or cancellation.
- Input that makes SubBridge pass unintended flags or arguments to a CLI, including the `model` or message text of a gateway request.
- The gateway (`subbridge serve`, `subbridge run`, `subbridge.serve()`) accepting connections on any address other than `127.0.0.1`, or sending CORS headers that let a web page read its replies.
- A way to use the gateway without the key it generated, or to learn that key over the network.
- A gateway request that starts a CLI (even its version or sign-in check) without a valid key.

Hooks, plugins, and settings that you configured yourself run with your permissions by design, so code they run is not an escape from read-only mode. Likewise, software running as your own user can read the gateway key, for example from the environment of a `subbridge run` command, just as it can use your CLI sign-in; that is not a gateway vulnerability.

Vulnerabilities in the `claude` or `codex` CLIs themselves belong with [Anthropic](https://www.anthropic.com/responsible-disclosure-policy) or [OpenAI](https://openai.com/security/disclosure/).
