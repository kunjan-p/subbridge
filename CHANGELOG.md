# Changelog

All notable changes to SubBridge are recorded here. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow [PEP 440](https://peps.python.org/pep-0440/).

## [Unreleased]

### Changed

- Claude Code now runs in a new `read-only` permission mode by default, replacing `plan`. The model gets only the Read, Glob, and Grep tools and no MCP servers, where plan mode still offered Bash, Edit, and MCP tools and often added remarks about planning to plain answers. Pass `permission_mode="plan"` for the old behavior.

- The package description now says what SubBridge does, and the package lists search keywords.

## [0.1.0a1]

First public alpha.

### Added

- `ClaudeClient` and `CodexClient`, which run the locally installed `claude` and `codex` CLIs.
- One-shot `ask()` and `ask_async()`, plus threads through `start_thread()` and `resume_thread()` with `run`, `stream`, and `stream_normalized` in sync and async forms.
- `TurnResult` with text, thread ID, token usage, model, elapsed time, and Claude's `structured_output` when an `output_schema` is given.
- Provider-neutral `StreamEvent`s through `normalize_event` and the `stream_normalized` methods.
- `status()` and `capabilities()` for install, sign-in, and CLI model-catalog checks without a model request.
- The `subbridge doctor` command, with `--json` output.
- Subscription-only mode, on by default, which strips API keys and endpoint overrides from the CLI's environment and requires Claude.ai or ChatGPT sign-in.
- Read-only defaults: the Codex `read-only` sandbox and the Claude Code `plan` permission mode.
- Redaction of raw CLI status, events, and stderr unless explicitly requested.
- Timeouts and cancellation that stop the CLI's whole process group.
- Turn errors that keep the CLI's own message next to SubBridge's hint, such as when a usage limit resets.
- Support for Python 3.11 through 3.14.

[Unreleased]: https://github.com/kunjan-p/subbridge/compare/v0.1.0a1...HEAD
[0.1.0a1]: https://github.com/kunjan-p/subbridge/releases/tag/v0.1.0a1
