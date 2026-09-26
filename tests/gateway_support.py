"""A gateway backed by the fake CLIs from test_claude.py and test_codex.py."""

import http.client
import importlib.util
import json
import os
import sys
import textwrap
import time
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, ClassVar
from unittest import mock

# The gateway tests need the (dev-only) anthropic and openai SDKs; the
# release workflow installs only requirements-release.txt, so skip cleanly
# there instead of failing to collect every module that imports this one.
if (
    importlib.util.find_spec("anthropic") is None
    or importlib.util.find_spec("openai") is None
):
    raise unittest.SkipTest(
        "anthropic and openai are not installed; skipping gateway tests."
    )

from anthropic import Anthropic
from openai import OpenAI
from test_claude import fake_claude
from test_codex import fake_codex

import subbridge


class GatewayTestCase(unittest.TestCase):
    """Starts fake `claude` and `codex` on PATH and a gateway in front of them."""

    gateway_options: ClassVar[dict[str, Any]] = {}

    def setUp(self) -> None:
        tempdir = TemporaryDirectory()
        self.addCleanup(tempdir.cleanup)
        self.bin = Path(tempdir.name)
        fake_claude(self.bin)
        fake_codex(self.bin)
        path = f"{self.bin}{os.pathsep}{os.environ.get('PATH', '')}"
        env = mock.patch.dict(os.environ, {"PATH": path})
        env.start()
        self.addCleanup(env.stop)
        self.gateway = subbridge.serve(**self.gateway_options)
        self.addCleanup(self.gateway.close)
        self.anthropic = Anthropic(
            base_url=self.gateway.anthropic_base_url,
            api_key=self.gateway.api_key,
            max_retries=0,
        )
        self.openai = OpenAI(
            base_url=self.gateway.openai_base_url,
            api_key=self.gateway.api_key,
            max_retries=0,
        )

    def cli_calls(self, name: str) -> list[list[str]]:
        """Every argument list the fake CLI was started with."""
        log = self.bin / f"{name}.calls"
        if not log.exists():
            return []
        return [json.loads(line) for line in log.read_text().splitlines()]

    def turn_calls(self, name: str) -> list[list[str]]:
        """Only the invocations that ran a model turn (not status checks)."""
        marker = "--output-format" if name == "claude" else "--json"
        return [args for args in self.cli_calls(name) if marker in args]

    def assert_cli_exits(self, name: str) -> None:
        pid = int((self.bin / f"{name}.pid").read_text())
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            try:
                os.kill(pid, 0)
            except ProcessLookupError:
                return
            time.sleep(0.05)
        self.fail(f"fake {name} (pid {pid}) is still running")

    def post(
        self, path: str, body: bytes, headers: dict[str, str] | None = None
    ) -> tuple[int, dict[str, Any]]:
        """Send raw bytes, for requests the SDKs refuse to build."""
        connection = http.client.HTTPConnection(
            "127.0.0.1", self.gateway.port, timeout=10
        )
        self.addCleanup(connection.close)
        connection.request("POST", path, body=body, headers=headers or {})
        response = connection.getresponse()
        return response.status, json.loads(response.read())

    def key_headers(self) -> dict[str, str]:
        return {"x-api-key": self.gateway.api_key, "Content-Type": "application/json"}


def auth_started_marker(pid_file: Path) -> Path:
    """Where `fake_claude_that_stalls` records that its auth check has begun.

    A test can wait for this file to appear to know the fake CLI's `auth
    status --json` call is now sleeping (rather than guessing a delay), so it
    can start `close()` deterministically while that turn has no process yet.
    """
    return pid_file.with_name(pid_file.name + ".auth-started")


def fake_claude_that_stalls(
    bin_dir: Path,
    pid_file: Path,
    sleep_seconds: int,
    *,
    auth_sleep_seconds: float = 0,
) -> Path:
    """A minimal fake `claude` that records its turn's PID before stalling.

    `auth_sleep_seconds` slows down `auth status --json`, standing in for a
    slow or loaded CLI, so a test can start `close()` while a turn is still
    inside `_validate_auth()` and has not spawned a process yet -- see
    `auth_started_marker()` to wait for that check to actually begin.
    """
    marker = auth_started_marker(pid_file)
    cli = bin_dir / "claude"
    cli.write_text(
        textwrap.dedent(
            f"""\
            #!{sys.executable}
            import json
            import os
            import sys
            import time

            args = sys.argv[1:]
            if args == ["--version"]:
                print("2.1.test")
            elif args == ["auth", "status", "--json"]:
                with open({str(marker)!r}, "w") as marker_file:
                    marker_file.write("1")
                time.sleep({auth_sleep_seconds})
                print(json.dumps({{"loggedIn": True, "authMethod": "claude.ai"}}))
            elif "--output-format" in args:
                prompt = sys.stdin.read()
                with open({str(pid_file)!r}, "w") as pid_file_handle:
                    pid_file_handle.write(str(os.getpid()))
                time.sleep({sleep_seconds})
                print(json.dumps({{"type": "result", "subtype": "success", "result": "answer: " + prompt}}))
            else:
                sys.exit(2)
            """
        ),
        encoding="utf-8",
    )
    cli.chmod(0o755)
    return cli
