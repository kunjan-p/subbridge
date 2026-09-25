from __future__ import annotations

import asyncio
import json
import tempfile
from collections.abc import AsyncGenerator, Generator
from contextlib import aclosing, closing, contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from ._result import TurnCollector
from ._stream import EventStream
from .errors import CodexNotInstalledError
from .events import normalize_event
from .models import StreamEvent, TurnResult

if TYPE_CHECKING:
    from .codex import CodexClient


SandboxMode = Literal[
    "read-only",
    "workspace-write",
    "danger-full-access",
]

ReasoningEffort = Literal[
    "minimal",
    "low",
    "medium",
    "high",
    "xhigh",
    "max",
    "ultra",
    "persistent",
]

ApprovalPolicy = Literal[
    "never",
    "on-request",
    "on-failure",
    "untrusted",
]

WebSearchMode = Literal[
    "disabled",
    "cached",
    "live",
]


@dataclass
class ThreadOptions:
    model: str | None = None
    reasoning_effort: ReasoningEffort | None = None

    sandbox: SandboxMode = "read-only"

    cwd: str | Path | None = None

    skip_git_repo_check: bool = True

    approval_policy: ApprovalPolicy | None = None

    network_access: bool | None = None
    web_search: WebSearchMode | None = None

    additional_directories: list[str | Path] | None = None


class CodexThread:
    def __init__(
        self,
        client: CodexClient,
        thread_id: str | None,
        options: ThreadOptions,
    ) -> None:
        self.client = client
        self.id = thread_id
        self.options = options

    def _build_command(
        self,
        *,
        schema_file: str | None = None,
        images: list[str | Path] | None = None,
    ) -> list[str]:
        if not self.client.codex_path:
            raise CodexNotInstalledError("Codex CLI not found.")
        options = self.options
        if options.additional_directories and self.id:
            raise ValueError(
                "Codex CLI does not support --add-dir for resumed sessions. "
                "Start a new thread with those directories instead."
            )
        cmd = [self.client.codex_path, "exec"]
        if self.id:
            cmd.extend(["resume", self.id])
        cmd.append("--json")
        if options.model:
            cmd.extend(["--model", options.model])
        if options.sandbox and not self.id:
            cmd.extend(["--sandbox", options.sandbox])
        if options.skip_git_repo_check:
            cmd.append("--skip-git-repo-check")
        if self.id and options.sandbox:
            cmd.extend(["--config", f'sandbox_mode="{options.sandbox}"'])
        if schema_file:
            cmd.extend(["--output-schema", schema_file])
        for key, value in (
            ("model_reasoning_effort", options.reasoning_effort),
            ("web_search", options.web_search),
            ("approval_policy", options.approval_policy),
        ):
            if value:
                cmd.extend(["--config", f'{key}="{value}"'])
        if options.network_access is not None:
            value = "true" if options.network_access else "false"
            cmd.extend(["--config", f"sandbox_workspace_write.network_access={value}"])
        for directory in options.additional_directories or []:
            cmd.extend(["--add-dir", str(Path(directory).expanduser())])
        for image in images or []:
            cmd.extend(["--image", str(Path(image).expanduser())])
        return cmd

    def stream(
        self,
        prompt: str,
        *,
        images: list[str | Path] | None = None,
        output_schema: dict[str, Any] | None = None,
        timeout: float | None = 300,
    ) -> Generator[dict[str, Any], None, None]:
        self.client._validate_auth()
        with _schema_file(output_schema) as schema_path:
            transport = EventStream(
                "codex",
                self._build_command(schema_file=schema_path, images=images),
                env=self.client._environment(),
                cwd=self.options.cwd,
                timeout=timeout,
                include_raw_diagnostics=self.client.include_raw_diagnostics,
            )
            with closing(transport.sync(prompt)) as events:
                for event in events:
                    if event.get("type") == "thread.started":
                        self.id = event.get("thread_id")
                    yield event

    def stream_normalized(
        self,
        prompt: str,
        *,
        include_raw: bool = False,
        images: list[str | Path] | None = None,
        output_schema: dict[str, Any] | None = None,
        timeout: float | None = 300,
    ) -> Generator[StreamEvent, None, None]:
        with closing(
            self.stream(
                prompt, images=images, output_schema=output_schema, timeout=timeout
            )
        ) as events:
            for event in events:
                yield normalize_event(
                    "codex", event, include_raw=include_raw, thread_id=self.id
                )

    async def stream_async(
        self,
        prompt: str,
        *,
        images: list[str | Path] | None = None,
        output_schema: dict[str, Any] | None = None,
        timeout: float | None = 300,
    ) -> AsyncGenerator[dict[str, Any], None]:
        await asyncio.to_thread(self.client._validate_auth)
        with _schema_file(output_schema) as schema_path:
            transport = EventStream(
                "codex",
                self._build_command(schema_file=schema_path, images=images),
                env=self.client._environment(),
                cwd=self.options.cwd,
                timeout=timeout,
                include_raw_diagnostics=self.client.include_raw_diagnostics,
            )
            async with aclosing(transport.async_(prompt)) as events:
                async for event in events:
                    if event.get("type") == "thread.started":
                        self.id = event.get("thread_id")
                    yield event

    async def stream_normalized_async(
        self,
        prompt: str,
        *,
        include_raw: bool = False,
        images: list[str | Path] | None = None,
        output_schema: dict[str, Any] | None = None,
        timeout: float | None = 300,
    ) -> AsyncGenerator[StreamEvent, None]:
        async with aclosing(
            self.stream_async(
                prompt, images=images, output_schema=output_schema, timeout=timeout
            )
        ) as events:
            async for event in events:
                yield normalize_event(
                    "codex", event, include_raw=include_raw, thread_id=self.id
                )

    def run(
        self,
        prompt: str,
        *,
        images: list[str | Path] | None = None,
        output_schema: dict[str, Any] | None = None,
        timeout: float | None = 300,
        include_events: bool = False,
    ) -> TurnResult:
        collector = TurnCollector("codex", self.options.model, include_events)
        with closing(
            self.stream(
                prompt, images=images, output_schema=output_schema, timeout=timeout
            )
        ) as events:
            for event in events:
                collector.add(event)
        return collector.finish(self.id)

    async def run_async(
        self,
        prompt: str,
        *,
        images: list[str | Path] | None = None,
        output_schema: dict[str, Any] | None = None,
        timeout: float | None = 300,
        include_events: bool = False,
    ) -> TurnResult:
        collector = TurnCollector("codex", self.options.model, include_events)
        async with aclosing(
            self.stream_async(
                prompt, images=images, output_schema=output_schema, timeout=timeout
            )
        ) as events:
            async for event in events:
                collector.add(event)
        return collector.finish(self.id)


@contextmanager
def _schema_file(schema: dict[str, Any] | None) -> Generator[str | None, None, None]:
    if schema is None:
        yield None
        return
    with tempfile.TemporaryDirectory(prefix="subbridge-") as directory:
        path = Path(directory) / "schema.json"
        path.write_text(json.dumps(schema, separators=(",", ":")), encoding="utf-8")
        yield str(path)
