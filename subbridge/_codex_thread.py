from __future__ import annotations

import json
import tempfile
from collections.abc import Generator
from contextlib import closing, contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ._errors import CodexNotInstalledError
from ._events import normalize_event
from ._models import StreamEvent, TurnResult
from ._result import TurnCollector
from ._stream import EventStream

if TYPE_CHECKING:
    from ._codex import CodexClient


@dataclass
class ThreadOptions:
    model: str | None = None
    cwd: str | Path | None = None
    skip_git_repo_check: bool = True


class CodexThread:
    def __init__(self, client: CodexClient, options: ThreadOptions) -> None:
        self.client = client
        self.id: str | None = None
        self.options = options

    def _build_command(self, *, schema_file: str | None = None) -> list[str]:
        if not self.client.codex_path:
            raise CodexNotInstalledError("Codex CLI not found.")
        options = self.options
        cmd = [self.client.codex_path, "exec", "--json"]
        if options.model:
            cmd.extend(["--model", options.model])
        # SubBridge always runs Codex in read-only sandbox mode; there is no
        # option to widen it.
        cmd.extend(["--sandbox", "read-only"])
        if options.skip_git_repo_check:
            cmd.append("--skip-git-repo-check")
        if schema_file:
            cmd.extend(["--output-schema", schema_file])
        return cmd

    def stream(
        self,
        prompt: str,
        *,
        output_schema: dict[str, Any] | None = None,
        timeout: float | None = 300,
    ) -> Generator[dict[str, Any], None, None]:
        self.client._validate_auth()
        with _schema_file(output_schema) as schema_path:
            transport = EventStream(
                "codex",
                self._build_command(schema_file=schema_path),
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
        output_schema: dict[str, Any] | None = None,
        timeout: float | None = 300,
    ) -> Generator[StreamEvent, None, None]:
        with closing(
            self.stream(prompt, output_schema=output_schema, timeout=timeout)
        ) as events:
            for event in events:
                yield normalize_event(
                    "codex", event, include_raw=include_raw, thread_id=self.id
                )

    def run(
        self,
        prompt: str,
        *,
        output_schema: dict[str, Any] | None = None,
        timeout: float | None = 300,
    ) -> TurnResult:
        collector = TurnCollector("codex", self.options.model)
        with closing(
            self.stream(prompt, output_schema=output_schema, timeout=timeout)
        ) as events:
            for event in events:
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
