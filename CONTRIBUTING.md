# Contributing

Thanks for helping. Bug reports, fixes, and ideas from the [roadmap](ROADMAP.md) are all welcome. For a larger change, open an issue first so we can agree on the approach before you write code.

## Set up

You need Python 3.11 or later. The `claude` and `codex` CLIs are only needed for the optional live tests.

```bash
git clone https://github.com/kunjan-p/subbridge.git
cd subbridge
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
```

## Run the checks

CI runs these on every pull request, on Python 3.11 through 3.14:

```bash
python -m pytest
ruff check .
ruff format --check .
```

`ruff format .` fixes formatting, and `ruff check --fix .` fixes most lint findings. The dev extra pins the ruff version so everyone gets the same rules.

## How the tests work

The unit tests never contact a model. They write small fake `claude` and `codex` scripts that print the JSON events a real CLI would, then point the clients at them. `fake_claude` in `tests/test_claude.py` and `fake_codex` in `tests/test_codex.py` show the pattern. Test new behavior the same way, including its failure paths.

The live smoke tests in `tests/test_live.py` use your signed-in CLIs and your subscription quota, so they only run when you opt in:

```bash
SUBBRIDGE_RUN_LIVE_TESTS=1 python -m pytest tests/test_live.py -v
```

Run them when you change how SubBridge talks to a CLI.

## Pull requests

- Keep SubBridge free of runtime dependencies. Tools needed only for development go in the `dev` extra.
- Add or update tests for any change in behavior.
- If you change the CLI commands, flags, or events SubBridge relies on, update [COMPATIBILITY.md](COMPATIBILITY.md).
- Describe user-visible changes under `Unreleased` in [CHANGELOG.md](CHANGELOG.md).
- Do not weaken the safety defaults (subscription-only mode, read-only sandbox and permission mode, redaction) without discussing it in an issue first.

## Security

Please do not report security problems in public issues. Use GitHub's private vulnerability reporting on the repository's Security tab instead.

## Releases

Maintainers release by moving the `Unreleased` entries in CHANGELOG.md under a new version heading, setting the same version in `pyproject.toml`, and pushing a tag such as `v0.1.0a2`. The publish workflow checks that the tag matches the package version, runs the tests, and uploads to PyPI.
