# SubBridge

SubBridge lets Python code call Claude Code and Codex through the CLIs you have already signed in to, so requests go through your Claude.ai or ChatGPT subscription rather than an API key.

It runs the official `claude` and `codex` command-line tools on your machine and adds a small Python API for one-shot prompts, resumable conversations, streaming events, and async code. It has no runtime dependencies.

It also runs a local gateway, so a script written for the official `anthropic` and `openai` Python SDKs can use those CLIs unchanged: prototype on the subscription your team already has, ship with a real API key. See [Use the official Anthropic and OpenAI SDKs](#use-the-official-anthropic-and-openai-sdks).

> SubBridge is alpha software. Read [Limitations](#limitations) before depending on it.

## Install

```bash
python -m pip install subbridge
```

You also need Python 3.11+ and at least one of the official CLIs, installed and signed in separately. SubBridge does not bundle or install them.

| Provider | CLI | Sign in with |
| --- | --- | --- |
| Claude Code | [`claude`](https://code.claude.com/docs/en/cli-usage) | a Claude.ai account (`claude auth login`) |
| Codex | [`codex`](https://github.com/openai/codex) | a ChatGPT account (`codex login`), see [Codex with your ChatGPT plan](https://help.openai.com/en/articles/11369540-using-codex-with-your-chatgpt-plan) |

Check your setup without sending a prompt:

```bash
subbridge doctor
```

## Quick start

```python
from subbridge import ClaudeClient, CodexClient

claude = ClaudeClient().ask("What is 17 * 23? Return only the number.", model="haiku")
print(claude.text)

codex = CodexClient().ask("What is 17 * 23? Return only the number.")
print(codex.text, codex.usage)
```

Keep context across turns with a thread:

```python
from subbridge import CodexClient

thread = CodexClient().start_thread(model="gpt-6-luna")
thread.run("My project is a CNC controller.")
print(thread.run("What project did I just mention?").text)

# Later, even in another process:
resumed = CodexClient().resume_thread(thread.id)
```

## API at a glance

Both clients have the same methods. Their options differ and follow the table.

| Call | What it does |
| --- | --- |
| `client.ask(prompt, ...)` / `await client.ask_async(prompt, ...)` | One prompt, returns a `TurnResult` |
| `client.start_thread(...)` / `client.resume_thread(thread_id, ...)` | A conversation that keeps context |
| `thread.run(prompt)` / `await thread.run_async(prompt)` | Send a turn, return a `TurnResult` |
| `thread.stream(prompt)` / `thread.stream_async(prompt)` | Yield the CLI's raw JSON events |
| `thread.stream_normalized(prompt)` / `thread.stream_normalized_async(prompt)` | Yield provider-neutral `StreamEvent`s |
| `client.status()` | Installed, signed in, auth mode, CLI version |
| `client.capabilities()` | `status()` plus CLI-known models and plan, without a model request |

`TurnResult` fields: `text`, `thread_id`, `usage` (`Usage` token counts), `provider`, `model`, `elapsed_seconds`, `structured_output`, plus `events` and `items` when you pass `include_events=True`.

Every turn call accepts `timeout` (seconds, default `300`, `None` for no limit) and `output_schema` (a JSON Schema dict). With a schema, Claude returns the parsed object in `structured_output`; Codex returns the schema-shaped JSON in `text`.

Claude Code accepts `model`, `effort`, `cwd`, `additional_directories`, and `permission_mode` (`"read-only"` by default; also Claude Code's own `"plan"`, `"default"`, `"acceptEdits"`, and `"dontAsk"`).

Codex accepts `model`, `reasoning_effort`, `sandbox` (`"read-only"` by default; also `"workspace-write"`, `"danger-full-access"`), `cwd`, `approval_policy`, `network_access`, `web_search`, `additional_directories`, and `skip_git_repo_check` (default `True`). Its turn calls also take `images`.

Every exception derives from `subbridge.errors.SubBridgeError`. Each provider has its own `NotInstalled`, `NotAuthenticated`, `WrongAuthMode`, `Process` (exit or timeout), `Turn` (the model turn failed, for example a usage limit), and `Protocol` (unexpected CLI output) errors, such as `ClaudeTurnError` and `CodexProcessError`.

### Async and cancellation

```python
import asyncio
from subbridge import ClaudeClient, CodexClient


async def main() -> None:
    claude, codex = await asyncio.gather(
        ClaudeClient().ask_async("Define idempotent in one line.", model="haiku"),
        CodexClient().ask_async("Define idempotent in one line.", model="gpt-6-luna"),
    )
    print(claude.text, codex.text, sep="\n")


asyncio.run(main())
```

Cancelling the task kills the CLI's whole process group, including anything the CLI started.

## Use the official Anthropic and OpenAI SDKs

Prototype on the subscription your team already has, ship with a real API key.

If getting an API key at your organization means a ticket, an approval, and a key that can be revoked later, you can still try a prompt or a flow on your own machine today. SubBridge's gateway is a small server on `127.0.0.1` that speaks the Anthropic Messages API and the OpenAI Chat Completions and Responses APIs. Code written for the official `anthropic` and `openai` Python SDKs sends its requests there, and your signed-in `claude` and `codex` CLIs answer them, within what your plan and your organization's policy allow. When the prototype works, run the same code against the real APIs with a real key.

Claude Pro, Max, and Enterprise seats work through Claude Code's Claude.ai sign-in; the maintainer has tested an Enterprise seat. Codex works with a ChatGPT sign-in. Other plans have not been tested.

### Run a script unchanged

The official SDKs read their endpoint and key from environment variables. `subbridge run` starts the gateway, sets `ANTHROPIC_BASE_URL`, `ANTHROPIC_API_KEY`, `OPENAI_BASE_URL`, and `OPENAI_API_KEY` for one command, and stops the gateway when that command exits:

```bash
subbridge run -- python app.py
```

`app.py` contains no SubBridge code:

```python
from anthropic import Anthropic
from openai import OpenAI

claude = Anthropic().messages.create(
    model="sonnet",
    max_tokens=500,
    messages=[{"role": "user", "content": "Name one prime number."}],
)
print(claude.content[0].text)

codex = OpenAI().responses.create(model="gpt-6-luna", input="Name one prime number.")
print(codex.output_text)
```

Ctrl+C and SIGTERM during `subbridge run` go to `app.py`, not to the gateway; `run` exits with `app.py`'s own exit code once it does. See `subbridge run --help` for the exit codes of the other outcomes, such as a command that is not found or a port already in use.

### From prototype to production

The same script, two commands:

```bash
# Prototype: your signed-in CLIs answer through the local gateway.
subbridge run -- python app.py

# Production: the real APIs answer, billed to your API keys.
ANTHROPIC_API_KEY=sk-ant-... OPENAI_API_KEY=sk-... python app.py
```

There is nothing to comment out. Check the model names: the gateway passes `model` to the CLI's `--model` flag as-is, so short names such as `sonnet` work in the prototype but not with the API. Use a name both accept, or read it from an environment variable as [`examples/official_sdks.py`](https://github.com/kunjan-p/subbridge/blob/main/examples/official_sdks.py) does.

### Other tools, and Python code

`subbridge serve` runs the gateway in the foreground. It prints the base URLs, the key, and `export` lines to paste into another terminal, and stops on Ctrl+C:

```bash
subbridge serve              # picks a free port
subbridge serve --port 8765
```

The key changes every time the gateway starts. To start the gateway from Python instead:

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

Pass `gw.openai_base_url`, which ends in `/v1`, to `openai.OpenAI(base_url=..., api_key=gw.api_key)`. `subbridge.serve(port=0, max_concurrency=4, timeout=300)` runs the gateway in a background thread: port `0` picks a free port, at most `max_concurrency` CLI turns run at once, extra requests wait up to `timeout` seconds for a free slot, and each turn is stopped after `timeout` seconds. Call `gw.close()` when you do not use `with`. Closing the gateway, whether by `close()`, the end of a `with` block, Ctrl+C in `subbridge serve`, or the command given to `subbridge run` finishing, stops any CLI turn still in flight.

### What works

| Endpoint | Answered by | Works | Rejected with a 400 |
| --- | --- | --- | --- |
| `POST /v1/messages` | Claude Code | `messages.create`, `messages.stream`, `stream=True`, `system`, text content | `tools`, `tool_choice`, `mcp_servers`, `container`, image and document blocks, structured output (`output_config.format`) |
| `POST /v1/chat/completions` | Codex | `chat.completions.create`, `stream=True`, `chat.completions.parse`, `response_format` with `json_schema`, `system` and `developer` messages | `tools`, `tool_choice`, `functions`, `function_call`, `audio`, `n` above 1, image and audio parts, `response_format` of type `json_object` |
| `POST /v1/responses` | Codex | `responses.create`, `responses.stream`, `stream=True`, `responses.parse`, `text.format` with `json_schema`, `instructions`, text or message `input` | `tools`, `tool_choice`, `previous_response_id`, `conversation`, `prompt`, `background`, image and file inputs |

Each request runs one CLI turn with SubBridge's read-only defaults. The whole conversation in the request is sent to the CLI as one transcript, and the gateway keeps nothing between requests. Parameters not in the table, such as `temperature`, `top_p`, `max_tokens`, `stop`, `seed`, `metadata`, and `user`, are accepted and ignored.

A non-streamed reply carries only the agent's final answer. Errors come back in each API's own error shape, so the SDKs raise their usual exceptions: a usage or rate limit is a 429, a model the CLI does not know is a 404, a model your plan lacks is a 403, a CLI that is missing or signed out is a 503, a timeout is a 504, and other CLI failures are a 502. Every error response carries `x-should-retry: false`, so the SDKs do not start the CLI again for a call that already failed.

### What the gateway can't do

- Embeddings, fine-tuning, batches, files, audio (speech or transcription), moderation, or any endpoint other than the three above. They get a 404 that says SubBridge's gateway does not support them.
- Custom tools or function calling. The CLI's agent has only its own read-only tools.
- Images, documents, or audio in a request, for either provider.
- Sampling control. `temperature`, `max_tokens`, and similar parameters are ignored, and Claude Code and Codex add their own agent instructions and behavior. A prototype that works through the gateway shows that your prompt and flow work; it does not promise the same output from the API in production.
- API speed. Each request starts a CLI process, which takes seconds, so the gateway is slower than the API. Streaming arrives in larger pieces than from the API, because the CLIs emit whole messages. A plain-text streamed reply can include the agent's interim messages, each separated from the next by a blank line, before its final answer. A streamed request with a JSON schema (`response_format` with `json_schema`, or `text.format`) waits out the whole turn and delivers only the final answer, the same as a non-streamed reply to the same request.

### Gateway security

The gateway listens on `127.0.0.1` only, with no option to listen on other addresses, and sends no CORS headers. Every request must carry the key the gateway generated, as `x-api-key` or `Authorization: Bearer`; a missing or wrong key gets a 401 before any CLI starts. Software running as your user can read the key, for example from the environment of a `subbridge run` command, as it can any local credential. The CLIs run in an empty temporary directory, not your project folder, so a gateway request can't read your files.

## `subbridge doctor`

```bash
subbridge doctor          # readable summary
subbridge doctor --json   # machine-readable
```

For each provider it reports whether the CLI is installed and signed in, the auth mode, the CLI version, the plan (when the CLI exposes one), and the models the CLI knows about. It never sends a prompt. `cli_known_models` is what the CLI lists, not what your plan allows.

## Safety defaults

The defaults stop a script from spending API credits or editing your files unless you opt in.

- Subscription-only mode is on. Before starting a CLI, SubBridge removes `ANTHROPIC_API_KEY`, `ANTHROPIC_AUTH_TOKEN`, `CLAUDE_CODE_OAUTH_TOKEN`, and `ANTHROPIC_BASE_URL` (Claude), or `OPENAI_API_KEY`, `CODEX_API_KEY`, and `OPENAI_BASE_URL` (Codex), from its environment. It then refuses to run unless the CLI reports Claude.ai or ChatGPT sign-in. Pass `subscription_only=False` to allow API keys, proxies, Bedrock, or Vertex.
- Codex runs in its `read-only` sandbox. Claude Code runs in SubBridge's `read-only` mode: the model gets only the Read, Glob, and Grep tools and no MCP servers, and settings from the working directory, such as a cloned repository's hooks, are ignored. Hooks and plugins from your own user settings still run. Permission prompts are off, so a run never stops to wait for input.
- `status()` omits the raw CLI output unless you pass `include_raw=True`. Turn results omit raw events unless you pass `include_events=True`. Error messages omit the CLI's stderr unless the client is created with `include_raw_diagnostics=True`. Raw output can contain prompts, file paths, and account details.
- Timeouts, errors, Ctrl+C, and cancelled tasks kill the whole CLI process group, so no CLI process outlives your script.
- The gateway listens on `127.0.0.1` only and refuses any request without the key it generated when it started. Its CLIs run with the same defaults as above.

SubBridge never reads credential files; the CLIs handle sign-in.

## Limitations

- SubBridge depends on the CLIs' command-line flags and JSON event formats, which can change between CLI releases. [COMPATIBILITY.md](https://github.com/kunjan-p/subbridge/blob/main/COMPATIBILITY.md) lists the CLI contract and the last live-verified versions.
- A signed-in CLI does not guarantee that your plan includes a model or that you have quota left. When a request is refused, the error includes the CLI's own message, such as when a usage limit resets.
- `TurnResult.usage` counts the tokens of one turn. The CLIs do not report how much of your plan quota remains.
- Model aliases like `haiku` and `gpt-6-luna` work only if your CLI version and plan offer them.
- Each call starts the CLI with your own configuration. Claude Code runs your hooks and loads your plugins every time, which can add seconds to each request, and a hook that injects text can change the reply.

## Examples

Runnable scripts live in [`examples/`](https://github.com/kunjan-p/subbridge/tree/main/examples):

| Script | Shows |
| --- | --- |
| [`claude_haiku.py`](https://github.com/kunjan-p/subbridge/blob/main/examples/claude_haiku.py), [`codex_luna.py`](https://github.com/kunjan-p/subbridge/blob/main/examples/codex_luna.py) | One quick prompt per provider |
| [`conversation.py`](https://github.com/kunjan-p/subbridge/blob/main/examples/conversation.py) | A multi-turn Codex thread |
| [`streaming.py`](https://github.com/kunjan-p/subbridge/blob/main/examples/streaming.py), [`claude_streaming.py`](https://github.com/kunjan-p/subbridge/blob/main/examples/claude_streaming.py) | Raw event streaming |
| [`async_clients.py`](https://github.com/kunjan-p/subbridge/blob/main/examples/async_clients.py) | Both providers concurrently with asyncio |
| [`hybrid.py`](https://github.com/kunjan-p/subbridge/blob/main/examples/hybrid.py) | A review loop: Claude drafts code, Codex reviews it, Claude revises, Codex checks the revision |
| [`official_sdks.py`](https://github.com/kunjan-p/subbridge/blob/main/examples/official_sdks.py) | The official `anthropic` and `openai` SDKs through the gateway: `subbridge run -- python examples/official_sdks.py` |

From a clone of this repository:

```bash
python examples/claude_haiku.py
```

The hybrid example passes text between one-shot calls only. It does not share threads, read your repository, or run the generated code.

## Roadmap

Planned work includes an agent guide, custom Python tools for agents, and per-action approval. See [ROADMAP.md](https://github.com/kunjan-p/subbridge/blob/main/ROADMAP.md).

## Development

```bash
python -m pip install -e ".[dev]"
python -m pytest
```

The test suite uses fake CLI scripts, so it is free and runs offline. Two live smoke tests make one short request per provider through your signed-in CLIs; they are skipped unless you opt in:

```bash
SUBBRIDGE_RUN_LIVE_TESTS=1 python -m pytest tests/test_live.py -v
```

Set `SUBBRIDGE_CLAUDE_MODEL` or `SUBBRIDGE_CODEX_MODEL` to test other models.

Contributions are welcome. [CONTRIBUTING.md](https://github.com/kunjan-p/subbridge/blob/main/CONTRIBUTING.md) covers setup, tests, and pull requests, and [CHANGELOG.md](https://github.com/kunjan-p/subbridge/blob/main/CHANGELOG.md) lists what changed in each release.

## License

MIT. See [LICENSE](https://github.com/kunjan-p/subbridge/blob/main/LICENSE).
