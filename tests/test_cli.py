import io
import json
import os
import signal
import socket
import subprocess
import sys
import textwrap
import time
import tomllib
import unittest
from contextlib import redirect_stdout, suppress
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

from subbridge import cli

GATEWAY_VARIABLES = {
    "ANTHROPIC_BASE_URL",
    "ANTHROPIC_API_KEY",
    "OPENAI_BASE_URL",
    "OPENAI_API_KEY",
}


class DispatchTests(unittest.TestCase):
    def test_console_script_points_at_the_dispatcher(self) -> None:
        pyproject = Path(__file__).resolve().parents[1] / "pyproject.toml"
        scripts = tomllib.loads(pyproject.read_text())["project"]["scripts"]
        self.assertEqual(scripts["subbridge"], "subbridge.cli:main")

    def test_doctor_json_is_unchanged(self) -> None:
        report = {"claude": {"installed": True, "authenticated": True}}
        output = io.StringIO()
        with (
            mock.patch("subbridge.doctor.collect_doctor_report", return_value=report),
            redirect_stdout(output),
        ):
            code = cli.main(["doctor", "--json"])
        self.assertEqual(code, 0)
        self.assertEqual(json.loads(output.getvalue()), report)

    def test_no_command_prints_help(self) -> None:
        output = io.StringIO()
        with redirect_stdout(output):
            self.assertEqual(cli.main([]), 2)
        self.assertIn("serve", output.getvalue())

    def test_doctor_help_has_a_description(self) -> None:
        output = io.StringIO()
        with redirect_stdout(output), self.assertRaises(SystemExit):
            cli.main(["doctor", "--help"])
        self.assertIn(
            "Check local provider CLI, login, model catalog, and plan/usage visibility.",
            output.getvalue(),
        )

    def test_run_help_lists_exit_codes(self) -> None:
        output = io.StringIO()
        with redirect_stdout(output), self.assertRaises(SystemExit):
            cli.main(["run", "--help"])
        text = output.getvalue()
        for phrase in ("128", "126", "127", "usage error"):
            self.assertIn(phrase, text)


class PortTests(unittest.TestCase):
    def test_invalid_port_is_a_usage_error(self) -> None:
        for value in ("-1", "70000", "not-a-number"):
            with (
                self.subTest(value=value),
                mock.patch("sys.stderr", io.StringIO()),
                self.assertRaises(SystemExit) as ctx,
            ):
                cli.main(["serve", "--port", value])
            self.assertEqual(ctx.exception.code, 2)


class ServeTests(unittest.TestCase):
    def test_serve_prints_urls_key_and_exports(self) -> None:
        output = io.StringIO()
        with (
            mock.patch.object(cli, "_wait_for_interrupt"),
            redirect_stdout(output),
        ):
            self.assertEqual(cli.main(["serve"]), 0)
        text = output.getvalue()
        for name in GATEWAY_VARIABLES:
            self.assertIn(f"export {name}=", text)
        self.assertIn("http://127.0.0.1:", text)
        self.assertIn("sb-local-", text)

    def test_busy_port_is_a_one_line_error(self) -> None:
        with socket.socket() as busy:
            busy.bind(("127.0.0.1", 0))
            busy.listen()
            port = busy.getsockname()[1]
            port_args = ["--port", str(port)]
            for argv in (["serve", *port_args], ["run", *port_args, "--", "true"]):
                with (
                    self.subTest(argv=argv),
                    mock.patch("sys.stderr", io.StringIO()) as stderr,
                ):
                    self.assertEqual(cli.main(argv), 1)
                    self.assertIn(
                        f"cannot listen on 127.0.0.1:{port}", stderr.getvalue()
                    )


class RunTests(unittest.TestCase):
    def run_child(self, code: str) -> tuple[int, dict[str, str]]:
        with TemporaryDirectory() as directory:
            dump = Path(directory) / "env.json"
            script = f"import json, os; open({str(dump)!r}, 'w').write(json.dumps(dict(os.environ)))\n{code}"
            exit_code = cli.main(["run", "--", sys.executable, "-c", script])
            return exit_code, json.loads(dump.read_text())

    def test_run_sets_exactly_the_four_sdk_variables(self) -> None:
        exit_code, child = self.run_child("")
        self.assertEqual(exit_code, 0)
        changed = {
            name for name, value in child.items() if os.environ.get(name) != value
        }
        self.assertEqual(changed, GATEWAY_VARIABLES)
        self.assertEqual(child["ANTHROPIC_API_KEY"], child["OPENAI_API_KEY"])
        self.assertTrue(child["ANTHROPIC_API_KEY"].startswith("sb-local-"))
        self.assertEqual(child["OPENAI_BASE_URL"], child["ANTHROPIC_BASE_URL"] + "/v1")

    def test_run_returns_the_child_exit_code(self) -> None:
        exit_code, _ = self.run_child("raise SystemExit(3)")
        self.assertEqual(exit_code, 3)

    def test_run_without_a_command_is_a_usage_error(self) -> None:
        with redirect_stdout(io.StringIO()), mock.patch("sys.stderr", io.StringIO()):
            self.assertEqual(cli.main(["run"]), 2)

    def test_run_reports_a_missing_command(self) -> None:
        with mock.patch("sys.stderr", io.StringIO()) as stderr:
            self.assertEqual(cli.main(["run", "--", "subbridge-no-such-command"]), 127)
        self.assertIn("command not found", stderr.getvalue())

    def test_run_reports_a_non_executable_command(self) -> None:
        with TemporaryDirectory() as directory:
            with mock.patch("sys.stderr", io.StringIO()) as stderr:
                self.assertEqual(cli.main(["run", "--", directory]), 126)
            self.assertIn("permission denied", stderr.getvalue())


@unittest.skipUnless(os.name == "posix", "process groups and SIGTERM are POSIX-only")
class SignalTests(unittest.TestCase):
    """`subbridge run` must let its child handle Ctrl+C, not kill it (round 1 fix)."""

    def spawn(self, child_code: str, *, new_session: bool = False) -> subprocess.Popen:
        return subprocess.Popen(
            [
                sys.executable,
                "-m",
                "subbridge.cli",
                "run",
                "--",
                sys.executable,
                "-c",
                child_code,
            ],
            start_new_session=new_session,
        )

    def test_run_lets_the_child_handle_sigint_and_returns_its_own_exit_code(
        self,
    ) -> None:
        child_code = (
            "import signal, sys, time\n"
            "signal.signal(signal.SIGINT, lambda *_: None)\n"
            "time.sleep(1)\n"
            "sys.exit(5)\n"
        )
        proc = self.spawn(child_code, new_session=True)
        try:
            time.sleep(0.3)
            os.killpg(proc.pid, signal.SIGINT)
            self.assertEqual(proc.wait(timeout=10), 5)
        finally:
            if proc.poll() is None:
                os.killpg(proc.pid, signal.SIGKILL)
                proc.wait()

    def test_run_forwards_sigterm_to_the_child(self) -> None:
        child_code = (
            "import signal, sys, time\n"
            "signal.signal(signal.SIGTERM, lambda *_: sys.exit(7))\n"
            "time.sleep(10)\n"
        )
        proc = self.spawn(child_code)
        try:
            time.sleep(0.3)
            proc.send_signal(signal.SIGTERM)
            self.assertEqual(proc.wait(timeout=10), 7)
        finally:
            if proc.poll() is None:
                proc.kill()
                proc.wait()


def _fake_claude_that_stalls(
    bin_dir: Path, sleep_seconds: int, *, auth_sleep_seconds: float = 0
) -> Path:
    """A minimal fake `claude` for exercising `subbridge run`'s own cleanup.

    Local to this file (unlike `gateway_support.fake_claude_that_stalls`) so
    this module needs no dev-only SDK to collect or run -- it must still work
    against only requirements-release.txt (see .github/workflows/publish.yml).
    `auth_sleep_seconds` slows `auth status --json` down, so the turn's own
    CLI has not even been spawned yet by the time the child below fires its
    request and exits -- the exact race C1 is about.
    """
    cli_path = bin_dir / "claude"
    cli_path.write_text(
        textwrap.dedent(
            f"""\
            #!{sys.executable}
            import json
            import sys
            import time

            args = sys.argv[1:]
            if args == ["--version"]:
                print("2.1.test")
            elif args == ["auth", "status", "--json"]:
                time.sleep({auth_sleep_seconds})
                print(json.dumps({{"loggedIn": True, "authMethod": "claude.ai"}}))
            elif "--output-format" in args:
                sys.stdin.read()
                time.sleep({sleep_seconds})
                print(json.dumps({{"type": "result", "subtype": "success", "result": "answer"}}))
            else:
                sys.exit(2)
            """
        ),
        encoding="utf-8",
    )
    cli_path.chmod(0o755)
    return cli_path


_FIRE_AND_FORGET_CHILD = textwrap.dedent(
    """\
    import http.client
    import json
    import os
    import threading
    from urllib.parse import urlsplit

    sent = threading.Event()


    def fire():
        url = urlsplit(os.environ["ANTHROPIC_BASE_URL"])
        connection = http.client.HTTPConnection(url.hostname, url.port, timeout=30)
        body = json.dumps(
            {
                "model": "sonnet",
                "max_tokens": 10,
                "messages": [{"role": "user", "content": "hi"}],
            }
        ).encode()
        connection.request(
            "POST",
            "/v1/messages",
            body=body,
            headers={
                "x-api-key": os.environ["ANTHROPIC_API_KEY"],
                "Content-Type": "application/json",
            },
        )
        sent.set()
        connection.getresponse()


    # A daemon thread: the process below exits once the request is on the
    # wire, without waiting for a reply.
    threading.Thread(target=fire, daemon=True).start()
    sent.wait(timeout=10)
    """
)


@unittest.skipUnless(os.name == "posix", "pgrep-based process checks are POSIX-only")
class OrphanedTurnTests(unittest.TestCase):
    """A child of `subbridge run` can fire a request and exit long before the
    turn finishes; `close()` must still find and kill that turn's CLI (round
    2 fix for C1), not leave it running once the child (and `run`) are gone.
    """

    def assert_no_fake_claude_running(self, cli_path: Path) -> None:
        deadline = time.monotonic() + 5
        result = None
        while time.monotonic() < deadline:
            result = subprocess.run(
                ["pgrep", "-f", str(cli_path)],
                capture_output=True,
                text=True,
                check=False,
            )
            if result.returncode != 0:  # no matching process
                return
            time.sleep(0.1)
        self.fail(f"a fake claude process is still running: {result.stdout!r}")

    def kill_any_matching(self, cli_path: Path) -> None:
        """Cleanup run whether the test passes or fails, so a regression
        here (a process this test's own fix should have killed) cannot
        leave a fake claude running on the machine past this test.
        """
        result = subprocess.run(
            ["pgrep", "-f", str(cli_path)], capture_output=True, text=True, check=False
        )
        for pid in result.stdout.split():
            with suppress(ProcessLookupError, ValueError):
                os.kill(int(pid), signal.SIGKILL)

    def test_run_leaves_nothing_running_after_a_child_fires_and_exits(self) -> None:
        with TemporaryDirectory() as tempdir:
            bin_dir = Path(tempdir)
            cli_path = _fake_claude_that_stalls(
                bin_dir, sleep_seconds=20, auth_sleep_seconds=3
            )
            self.addCleanup(self.kill_any_matching, cli_path)
            child_path = bin_dir / "child.py"
            child_path.write_text(_FIRE_AND_FORGET_CHILD, encoding="utf-8")
            path = f"{bin_dir}{os.pathsep}{os.environ.get('PATH', '')}"
            with mock.patch.dict(os.environ, {"PATH": path}):
                exit_code = cli.main(["run", "--", sys.executable, str(child_path)])
            self.assertEqual(exit_code, 0)
            self.assert_no_fake_claude_running(cli_path)


if __name__ == "__main__":
    unittest.main()
