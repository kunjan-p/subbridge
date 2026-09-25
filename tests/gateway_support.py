"""A gateway backed by the fake CLIs from test_claude.py and test_codex.py."""

import http.client
import json
import os
import time
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, ClassVar
from unittest import mock

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
