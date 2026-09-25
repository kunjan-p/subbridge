# Compatibility

## Supported runtime

The package targets Python 3.11 and later. CI runs the unit suite on Python 3.11, 3.12, 3.13, and 3.14.

SubBridge launches the provider CLIs rather than calling provider APIs, so it depends on the commands and JSONL output below:

| Provider | Required CLI contract | Optional capability |
| --- | --- | --- |
| Codex | `codex login status`; `codex exec --json` | `codex debug models` catalog |
| Claude Code | `claude auth status --json`; `claude -p --output-format stream-json --verbose` | None currently queried |

Codex exposes its model catalog through a `debug` command, which may be missing or change between CLI releases. When it is unavailable, `CodexClient.capabilities()` still returns install and sign-in details and reports `models=None`. The catalog lists the models the CLI knows about. It does not show which of them the signed-in account can use or how they are billed. Claude Code has no stable non-interactive model catalog, so SubBridge reports its model list as unavailable.

## Verified CLI versions

Live smoke checks last ran on 2026-09-25 on macOS with Python 3.14.7.

| CLI | Last live-verified version | Verification |
| --- | --- | --- |
| Codex CLI | 0.156.1 | Signed-in ChatGPT account; one-shot and resumed CLI flows |
| Claude Code | 2.1.282 | Signed-in Claude.ai account; one-shot CLI flow |

Codex CLI 0.157.0 is installed on the test machine but has not completed a live run yet, because the account hit its usage limit.

These are the exact versions that passed, not minimum requirements. Support for older and newer releases is best effort until they pass the opt-in live smoke tests. When a CLI command or its output format is incompatible, SubBridge raises a provider-specific error. It does not check which models an account can use.

## Run local live smoke tests

The default test suite uses fake CLI executables and never contacts a model. To test the CLIs installed on your machine, install the development extra and opt in:

```bash
python -m pip install -e ".[dev]"
SUBBRIDGE_RUN_LIVE_TESTS=1 python -m unittest discover -s tests -p 'test_live.py' -v
```

The tests use your existing CLI sign-in in subscription-only mode. They inspect local capabilities, then make one short asynchronous model request per provider. The smoke tests default to Haiku and GPT-6 Luna; set `SUBBRIDGE_CLAUDE_MODEL` or `SUBBRIDGE_CODEX_MODEL` to pick another model the CLI supports. Whether your account can use that model still depends on your plan.
