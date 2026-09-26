# SubBridge

SubBridge lets you use the official `anthropic` and `openai` Python SDKs through the Claude Code and Codex CLIs you are already signed in to, so requests go through your Claude.ai or ChatGPT subscription instead of an API key.

![use_subscription() with the official anthropic SDK](https://raw.githubusercontent.com/kunjan-p/subbridge/main/assets/subbridge-sdk-demo.gif)

> SubBridge is alpha software. Read [Limitations](#limitations) before depending on it.

## Why

Getting an API key at your organization often means a ticket, an approval, and a key that can be revoked later. SubBridge lets you prototype a prompt or a flow today, on the subscription your team already has, within your plan and your organization's policy. When it works, ship the same code with a real API key.

Claude Pro, Max, and Enterprise seats work through Claude Code's Claude.ai sign-in (the maintainer has tested an Enterprise seat). Codex works with a ChatGPT sign-in. Other plans have not been tested.

## Install

```bash
python -m pip install subbridge
```

You need Python 3.11+ and at least one of the official CLIs, installed and signed in separately. SubBridge does not bundle or install them.

| Provider | CLI | Sign in with |
| --- | --- | --- |
| Claude Code | [`claude`](https://code.claude.com/docs/en/cli-usage) | a Claude.ai account (`claude auth login`) |
| Codex | [`codex`](https://github.com/openai/codex) | a ChatGPT account (`codex login`), see [Codex with your ChatGPT plan](https://help.openai.com/en/articles/11369540-using-codex-with-your-chatgpt-plan) |

Check your setup without sending a prompt:

```bash
subbridge doctor          # readable summary
subbridge doctor --json   # machine-readable
```

For each provider it reports whether the CLI is installed and signed in, the auth mode, the CLI version, the plan (when the CLI exposes one), and the models the CLI knows about. `cli_known_models` is what the CLI lists, not what your plan allows.

To use the official SDKs through the gateway (below), also run `pip install anthropic openai`. SubBridge itself has no runtime dependencies.

## Use the official SDKs

### One line in your script

Call `subbridge.use_subscription()` before constructing `Anthropic()` or `OpenAI()`, not after: both SDKs read their base URL and key once, at construction, from environment variables it sets.

```python
import os

import subbridge

subbridge.use_subscription()

from anthropic import Anthropic
from openai import OpenAI

claude = Anthropic().messages.create(
    model=os.environ.get("CLAUDE_MODEL", "sonnet"),
    max_tokens=500,
    messages=[{"role": "user", "content": "Name one prime number."}],
)
codex = OpenAI().responses.create(
    model=os.environ.get("OPENAI_MODEL", "gpt-6-luna"), input="Name one prime number."
)
print(claude.content[0].text, codex.output_text)
```

It is one gateway for the whole process, not one per caller. Closing it, including by leaving a `with subbridge.use_subscription():` block, closes it for any other code in the process still relying on it.

### Or wrap the command, no code changes

```bash
subbridge run -- python app.py
```

This is the script above without its two `subbridge` lines. `run` sets exactly `ANTHROPIC_BASE_URL`, `ANTHROPIC_API_KEY`, `OPENAI_BASE_URL`, and `OPENAI_API_KEY` for that one command, and stops the gateway when it exits. Ctrl+C and SIGTERM (macOS and Linux) go to the command, not the gateway; `run` exits with the command's own exit code. See `subbridge run --help` for the exit codes of other outcomes, such as a command not found or a port already in use.

### Or run it yourself

```bash
subbridge serve              # picks an available port
subbridge serve --port 8765
```

Runs in the foreground and stops on Ctrl+C. Prints the base URLs, the key, and `export` lines to paste into another terminal. The key changes every time the gateway starts. From Python:

```python
import subbridge
from anthropic import Anthropic

with subbridge.serve() as gw:
    client = Anthropic(base_url=gw.anthropic_base_url, api_key=gw.api_key)
    reply = client.messages.create(
        model="sonnet", max_tokens=500, messages=[{"role": "user", "content": "hi"}]
    )
    print(reply.content[0].text)
```

Pass `gw.openai_base_url` (ends in `/v1`) to `openai.OpenAI(base_url=..., api_key=gw.api_key)`. `subbridge.serve(port=0, max_concurrency=4, timeout=300)` runs the gateway in a background thread: port `0` picks an available port, at most `max_concurrency` CLI turns run at once, extra requests wait up to `timeout` seconds for a slot to open, and each turn is stopped after `timeout` seconds. Closing the gateway, however you close it, stops any CLI turn still in flight. Call `gw.close()` when you do not use `with`.

### Going to production

Delete the `subbridge.use_subscription()` line (or drop `subbridge run`), set a real `ANTHROPIC_API_KEY` / `OPENAI_API_KEY`, and set `CLAUDE_MODEL` / `OPENAI_MODEL` to model IDs the real APIs accept. The gateway passes `model` to the CLI's `--model` flag as-is, so short names like `sonnet` work only in the prototype:

```bash
# Prototype: your signed-in CLIs answer through the local gateway; app.py's default model names work here.
subbridge run -- python app.py

# Production: the real APIs answer, billed to your API keys.
ANTHROPIC_API_KEY=sk-ant-... OPENAI_API_KEY=sk-... CLAUDE_MODEL=<your-anthropic-model-id> OPENAI_MODEL=<your-openai-model-id> python app.py
```

With `subbridge run` there is nothing to comment out; `app.py` reads its model names from `CLAUDE_MODEL` and `OPENAI_MODEL` either way, the pattern [`examples/official_sdks.py`](https://github.com/kunjan-p/subbridge/blob/main/examples/official_sdks.py) uses.

### What the gateway supports

| Endpoint | Answered by | Works | Rejected with a 400 |
| --- | --- | --- | --- |
| `POST /v1/messages` | Claude Code | `messages.create`, `messages.stream`, `stream=True`, `system`, text content | `tools`, `tool_choice`, `mcp_servers`, `container`, image and document blocks, structured output (`output_config.format`) |
| `POST /v1/chat/completions` | Codex | `chat.completions.create`, `stream=True`, `chat.completions.parse`, `response_format` with `json_schema`, `system` and `developer` messages | `tools`, `tool_choice`, `functions`, `function_call`, `audio`, `n` above 1, image and audio parts, `response_format` of type `json_object` |
| `POST /v1/responses` | Codex | `responses.create`, `responses.stream`, `stream=True`, `responses.parse`, `text.format` with `json_schema`, `instructions`, text or message `input` | `tools`, `tool_choice`, `previous_response_id`, `conversation`, `prompt`, `background`, image and file inputs, `text.format` of type `json_object` |

- Each request runs one CLI turn with SubBridge's read-only defaults.
- The whole conversation in the request is sent to the CLI as one transcript. The gateway keeps nothing between requests.
- Parameters not in the table, such as `temperature`, `top_p`, `max_tokens`, `stop`, `seed`, `metadata`, and `user`, are accepted and ignored.
- A non-streamed reply carries only the final answer.
- Errors come back in each API's own error shape, so the SDKs raise their usual exceptions:

| Cause | Status |
| --- | --- |
| Usage or rate limit | 429 |
| Model the CLI does not know | 404 |
| Model your plan lacks | 403 |
| CLI missing, signed out, or no turn slot open | 503 |
| Timeout | 504 |
| Other CLI failure | 502 |
| Endpoint other than the three above | 404 |
| Missing or wrong gateway key | 401 |

Every JSON error response carries `x-should-retry: false`, so the SDKs do not start the CLI again for a call that already failed.

### What it can't do

- Embeddings, fine-tuning, batches, files, audio (speech or transcription), moderation, or any endpoint other than the three above.
- Custom tools or function calling. The CLI's agent has only its own read-only tools.
- Images, documents, or audio in a request, for either provider.
- Sampling control and output parity with the API. `temperature`, `max_tokens`, and similar parameters are ignored, and Claude Code and Codex add their own agent behavior. A prototype that works through the gateway shows your prompt and flow work; it does not promise the same output from the API in production.
- API speed. Each request starts a CLI process, which takes seconds. Streaming arrives in larger pieces than from the API, because the CLIs emit whole messages: a plain-text stream can include the agent's interim messages, each separated by a blank line, before its final answer, and a stream with a JSON schema (`response_format` with `json_schema`, or `text.format`) waits out the whole turn and delivers only the final answer.

## Safety and security

- **Subscription-only mode is always on.** Before starting a CLI, the gateway removes `ANTHROPIC_API_KEY`, `ANTHROPIC_AUTH_TOKEN`, `CLAUDE_CODE_OAUTH_TOKEN`, and `ANTHROPIC_BASE_URL` (Claude), or `OPENAI_API_KEY`, `CODEX_API_KEY`, and `OPENAI_BASE_URL` (Codex), from its environment, then refuses to run unless the CLI reports Claude.ai or ChatGPT sign-in. There is no option to turn this off.
- **Read-only, always.** Codex always runs in its `read-only` sandbox, and Claude Code always runs in SubBridge's `read-only` mode: the model gets only the Read, Glob, and Grep tools and no MCP servers, and settings from the working directory, such as a cloned repository's hooks, are ignored. Hooks and plugins from your own user settings still run. Permission prompts are off, so a turn never stops to wait for input.
- **CLI diagnostics stay private.** Replies carry only the parsed answer. Error messages carry SubBridge's description of the failure plus the error message the CLI itself reported (for example, when a usage limit resets). The CLI's stderr, raw event stream, and raw sign-in status never appear. `subbridge doctor` reports install status, the CLI's version string, sign-in mode, and the model catalog, and no other raw CLI output.
- **Cleanup is automatic.** A turn that times out, and any turn still in flight when the gateway closes, has its whole CLI process group killed, so no CLI process outlives the gateway.
- **The gateway is local-only.** It listens on `127.0.0.1` only, with no option to listen elsewhere, and sends no CORS headers. Every request needs the key the gateway generated, as `x-api-key` or `Authorization: Bearer`; a missing or wrong key gets a 401 before any CLI starts. Software running as your own user can read the key, for example from the environment of a `subbridge run` command, the same as it can any local credential.
- **The CLIs can still read files.** They start in an empty temporary directory rather than the one you ran the command from, but that is only a starting point, not a sandbox boundary: Codex's read-only sandbox can still read files elsewhere on your machine, and Claude Code follows your own permission settings. Do not send untrusted text through SubBridge on a machine with files you would not show the model.
- **Watch your proxy settings.** If `HTTP_PROXY`, `HTTPS_PROXY`, or `ALL_PROXY` is set, add `127.0.0.1` to `NO_PROXY`, or the SDKs may send the gateway key and your prompts to that proxy instead of the gateway. `subbridge run`, `subbridge serve`, and `subbridge.use_subscription()` warn about this on startup.
- SubBridge never reads credential files; the CLIs handle sign-in.

See [SECURITY.md](https://github.com/kunjan-p/subbridge/blob/main/SECURITY.md) to report a vulnerability.

## Limitations

- SubBridge depends on the CLIs' command-line flags and JSON event formats, which can change between CLI releases. [COMPATIBILITY.md](https://github.com/kunjan-p/subbridge/blob/main/COMPATIBILITY.md) lists the CLI contract and the last live-verified versions.
- A signed-in CLI does not guarantee that your plan includes a model or that you have quota left. When a request is refused, the error includes the CLI's own message, such as when a usage limit resets.
- Model aliases like `haiku` and `gpt-6-luna` work only if your CLI version and plan offer them.
- Each call starts the CLI with your own configuration. Claude Code runs your hooks and loads your plugins every time, which can add seconds to each request, and a hook that injects text can change the reply.

## Examples

Runnable scripts live in [`examples/`](https://github.com/kunjan-p/subbridge/tree/main/examples):

| Script | Shows |
| --- | --- |
| [`official_sdks.py`](https://github.com/kunjan-p/subbridge/blob/main/examples/official_sdks.py) | The official `anthropic` and `openai` SDKs through the gateway (`pip install anthropic openai` first): `subbridge run -- python examples/official_sdks.py` |
| [`use_subscription.py`](https://github.com/kunjan-p/subbridge/blob/main/examples/use_subscription.py) | The same SDKs with `subbridge.use_subscription()` instead, so no wrapper command is needed |
| [`hybrid.py`](https://github.com/kunjan-p/subbridge/blob/main/examples/hybrid.py) | A review loop through the SDKs: Claude drafts code, Codex reviews it, Claude revises, Codex checks the revision |

From a clone of this repository, run one directly: `python examples/use_subscription.py`. The hybrid example passes text between one-shot calls only; it does not share threads or run the generated code, and the script itself does not read your repository.

## Development

```bash
python -m pip install -e ".[dev]"
python -m pytest
```

The test suite uses fake CLI scripts, so it runs offline without contacting a real CLI. Two live smoke tests start a gateway and make one short request per provider through it, using the official SDKs; they are skipped unless you opt in:

```bash
SUBBRIDGE_RUN_LIVE_TESTS=1 python -m pytest tests/test_live.py -v
```

Set `SUBBRIDGE_CLAUDE_MODEL` or `SUBBRIDGE_CODEX_MODEL` to test other models.

Planned work includes tool calling, image and document input, and more request features through the gateway; see [ROADMAP.md](https://github.com/kunjan-p/subbridge/blob/main/ROADMAP.md). Contributions are welcome: [CONTRIBUTING.md](https://github.com/kunjan-p/subbridge/blob/main/CONTRIBUTING.md) covers setup, tests, and pull requests, and [CHANGELOG.md](https://github.com/kunjan-p/subbridge/blob/main/CHANGELOG.md) lists what changed in each release.

## License

MIT. See [LICENSE](https://github.com/kunjan-p/subbridge/blob/main/LICENSE).
