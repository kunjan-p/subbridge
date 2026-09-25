# Roadmap

Each SubBridge call already runs the CLI's own agent loop, so you can drive agents today: widen `sandbox` or `permission_mode`, set `cwd`, and chain calls in Python as [`examples/hybrid.py`](examples/hybrid.py) does. The planned work builds on that.

## Planned

- An agent example and guide: a sandboxed loop where Codex or Claude Code edits a project, runs its tests, and reports back.
- Custom tools, so agents can call your Python functions and not only the CLI's built-in tools. A local MCP server is the likely route.
- Per-action approval, so your code can allow or deny each file edit or command as it happens instead of fixing permissions before the run.

Items have no dates. Order reflects current priority and can change.

## Suggest something

Open an [issue](https://github.com/kunjan-p/subbridge/issues) describing what you want to build and where SubBridge gets in the way. Pull requests are welcome; see [CONTRIBUTING.md](CONTRIBUTING.md).
