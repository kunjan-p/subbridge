from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncGenerator, Generator
from contextlib import aclosing, closing
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from ._result import TurnCollector
from ._stream import EventStream
from .errors import ClaudeNotInstalledError
from .events import normalize_event
from .models import StreamEvent, TurnResult

if TYPE_CHECKING:
    from .claude import ClaudeClient


ClaudePermissionMode = Literal["read-only", "default", "acceptEdits", "plan", "dontAsk"]

# "read-only" is SubBridge's own mode. Claude Code's plan mode also blocks
# writes, but it still offers Bash, Edit, and MCP tools and makes the model
# talk about planning. Here the model only has these tools and no MCP servers.
READ_ONLY_FLAGS = [
    "--permission-mode",
    "dontAsk",
    "--tools=Read,Glob,Grep",
    "--strict-mcp-config",
]


@dataclass
class ClaudeThreadOptions:
    model: str | None = None
    effort: str | None = None
    cwd: str | Path | None = None
    additional_directories: list[str | Path] | None = None
    permission_mode: ClaudePermissionMode = "read-only"


class ClaudeThread:
    def __init__(
        self,
        client: ClaudeClient,
        thread_id: str | None,
        options: ClaudeThreadOptions,
    ) -> None:
        self.client = client
        self.id = thread_id
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
        if self.options.permission_mode == "read-only":
            cmd.extend(READ_ONLY_FLAGS)
        else:
            cmd.extend(["--permission-mode", self.options.permission_mode])
        if self.options.model:
            cmd.extend(["--model", self.options.model])
        if self.options.effort:
            cmd.extend(["--effort", self.options.effort])
        # Claude Code receives cwd from the subprocess rather than a CLI flag.
        if self.options.additional_directories:
            for directory in self.options.additional_directories:
                cmd.extend(["--add-dir", str(Path(directory).expanduser())])
        if self.id:
            cmd.extend(["--resume", self.id])
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
        include_events: bool = False,
    ) -> TurnResult:
        collector = TurnCollector("claude", self.options.model, include_events)
        with closing(
            self.stream(prompt, output_schema=output_schema, timeout=timeout)
        ) as events:
            for event in events:
                collector.add(event)
        return collector.finish(self.id)

    async def stream_async(
        self,
        prompt: str,
        *,
        output_schema: dict[str, Any] | None = None,
        timeout: float | None = 300,
    ) -> AsyncGenerator[dict[str, Any], None]:
        await asyncio.to_thread(self.client._validate_auth)
        transport = EventStream(
            "claude",
            self._build_command(output_schema),
            env=self.client._environment(),
            cwd=self.options.cwd,
            timeout=timeout,
            include_raw_diagnostics=self.client.include_raw_diagnostics,
        )
        async with aclosing(transport.async_(prompt)) as events:
            async for event in events:
                if event.get("type") == "result":
                    self.id = event.get("session_id") or self.id
                yield event

    async def stream_normalized_async(
        self,
        prompt: str,
        *,
        include_raw: bool = False,
        output_schema: dict[str, Any] | None = None,
        timeout: float | None = 300,
    ) -> AsyncGenerator[StreamEvent, None]:
        async with aclosing(
            self.stream_async(prompt, output_schema=output_schema, timeout=timeout)
        ) as events:
            async for event in events:
                yield normalize_event(
                    "claude", event, include_raw=include_raw, thread_id=self.id
                )

    async def run_async(
        self,
        prompt: str,
        *,
        output_schema: dict[str, Any] | None = None,
        timeout: float | None = 300,
        include_events: bool = False,
    ) -> TurnResult:
        collector = TurnCollector("claude", self.options.model, include_events)
        async with aclosing(
            self.stream_async(prompt, output_schema=output_schema, timeout=timeout)
        ) as events:
            async for event in events:
                collector.add(event)
        return collector.finish(self.id)
