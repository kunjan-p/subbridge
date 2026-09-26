from __future__ import annotations

import json
from collections.abc import Generator
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ._errors import ClaudeNotInstalledError
from ._events import normalize_event
from ._models import StreamEvent, TurnResult
from ._result import TurnCollector
from ._stream import EventStream

if TYPE_CHECKING:
    from ._claude import ClaudeClient


# SubBridge always runs Claude Code in this read-only mode; there is no
# option to widen it. Claude Code's plan mode also blocks writes, but it
# still offers Bash, Edit, and MCP tools and makes the model talk about
# planning. Here the model only has these tools and no MCP servers. Project
# and local settings from the working directory are ignored, so a
# repository's own hooks don't run.
READ_ONLY_FLAGS = (
    "--permission-mode",
    "dontAsk",
    "--tools=Read,Glob,Grep",
    "--strict-mcp-config",
    "--setting-sources",
    "user",
)


@dataclass
class ClaudeThreadOptions:
    model: str | None = None
    cwd: str | Path | None = None


class ClaudeThread:
    def __init__(self, client: ClaudeClient, options: ClaudeThreadOptions) -> None:
        self.client = client
        self.id: str | None = None
        self.options = options

    def _build_command(self, output_schema: dict[str, Any] | None = None) -> list[str]:
        if not self.client.claude_path:
            raise ClaudeNotInstalledError("Claude Code CLI not found.")
        cmd = [
            self.client.claude_path,
            "-p",
            "--output-format",
            "stream-json",
            "--verbose",
            "--permission-prompts",
            "none",
        ]
        cmd.extend(READ_ONLY_FLAGS)
        if self.options.model:
            cmd.extend(["--model", self.options.model])
        if output_schema is not None:
            cmd.extend(
                ["--json-schema", json.dumps(output_schema, separators=(",", ":"))]
            )
        return cmd

    def stream(
        self,
        prompt: str,
        *,
        output_schema: dict[str, Any] | None = None,
        timeout: float | None = 300,
    ) -> Generator[dict[str, Any], None, None]:
        self.client._validate_auth()
        transport = EventStream(
            "claude",
            self._build_command(output_schema),
            env=self.client._environment(),
            cwd=self.options.cwd,
            timeout=timeout,
            include_raw_diagnostics=self.client.include_raw_diagnostics,
        )
        with closing(transport.sync(prompt)) as events:
            for event in events:
                if event.get("type") == "result":
                    self.id = event.get("session_id") or self.id
                yield event

    def stream_normalized(
        self,
        prompt: str,
        *,
        include_raw: bool = False,
        output_schema: dict[str, Any] | None = None,
        timeout: float | None = 300,
    ) -> Generator[StreamEvent, None, None]:
        with closing(
            self.stream(prompt, output_schema=output_schema, timeout=timeout)
        ) as events:
            for event in events:
                yield normalize_event(
                    "claude", event, include_raw=include_raw, thread_id=self.id
                )

    def run(
        self,
        prompt: str,
        *,
        output_schema: dict[str, Any] | None = None,
        timeout: float | None = 300,
    ) -> TurnResult:
        collector = TurnCollector("claude", self.options.model)
        with closing(
            self.stream(prompt, output_schema=output_schema, timeout=timeout)
        ) as events:
            for event in events:
                collector.add(event)
        return collector.finish(self.id)
