import io
import json
import os
import socket
import sys
import tomllib
import unittest
from contextlib import redirect_stdout
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


if __name__ == "__main__":
    unittest.main()
