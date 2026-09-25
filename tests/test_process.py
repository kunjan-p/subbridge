import os
import subprocess
import sys
import time
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from subbridge._process import (
    describe_turn_failure,
    process_error_message,
    process_group_options,
    terminate_process_tree,
)


class ProcessErrorMessageTests(unittest.TestCase):
    def test_model_plan_restriction_is_explained_without_raw_diagnostics(self) -> None:
        message = process_error_message(
            "Claude Code",
            1,
            "model is not available on your plan",
            include_raw_diagnostics=False,
        )
        self.assertIn("unavailable on this account plan", message)
        self.assertNotIn("model is not available on your plan", message)

    def test_usage_limit_is_distinguished(self) -> None:
        message = process_error_message(
            "Codex", 1, "usage limit reached", include_raw_diagnostics=False
        )
        self.assertIn("usage or rate limit", message)


class TurnFailureMessageTests(unittest.TestCase):
    def test_keeps_cli_message_next_to_hint(self) -> None:
        message = describe_turn_failure(
            "Codex", "You've hit your usage limit. Try again at Sep 28th, 7:14 PM."
        )
        self.assertIn("usage or rate limit", message)
        self.assertIn("Try again at Sep 28th, 7:14 PM.", message)

    def test_unclassified_error_is_returned_unchanged(self) -> None:
        self.assertEqual(describe_turn_failure("Codex", "boom"), "boom")

    def test_context_classifies_but_is_never_shown(self) -> None:
        message = describe_turn_failure(
            "Codex", "turn failed", context="rate limit account=/private/secret"
        )
        self.assertIn("usage or rate limit", message)
        self.assertNotIn("/private/secret", message)


@unittest.skipUnless(os.name == "posix", "process-group assertion is POSIX-specific")
class ProcessTreeTests(unittest.TestCase):
    def test_terminate_process_tree_stops_descendant(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            pid_file = root / "child.pid"
            counter_file = root / "child.counter"
            child_script = root / "child.py"
            parent_script = root / "parent.py"
            child_script.write_text(
                "import time\n"
                f"path = {str(counter_file)!r}\n"
                "while True:\n"
                "    with open(path, 'a') as counter:\n"
                "        counter.write('x')\n"
                "    time.sleep(0.02)\n",
                encoding="utf-8",
            )
            parent_script.write_text(
                "import subprocess, sys, time\n"
                f"child = subprocess.Popen([sys.executable, {str(child_script)!r}])\n"
                f"open({str(pid_file)!r}, 'w').write(str(child.pid))\n"
                "time.sleep(60)\n",
                encoding="utf-8",
            )
            parent = subprocess.Popen(
                [sys.executable, str(parent_script)],
                **process_group_options(),
            )
            try:
                deadline = time.monotonic() + 3
                while not pid_file.exists() and time.monotonic() < deadline:
                    time.sleep(0.01)
                self.assertTrue(pid_file.exists(), "child process did not start")
                deadline = time.monotonic() + 3
                while (
                    not counter_file.exists() or not counter_file.stat().st_size
                ) and time.monotonic() < deadline:
                    time.sleep(0.01)
                self.assertTrue(counter_file.exists() and counter_file.stat().st_size)
                terminate_process_tree(parent)
                self.assertIsNotNone(parent.poll())
                stopped_size = counter_file.stat().st_size
                time.sleep(0.1)
                self.assertEqual(
                    counter_file.stat().st_size,
                    stopped_size,
                    "child process remained active",
                )
            finally:
                if parent.poll() is None:
                    terminate_process_tree(parent)


if __name__ == "__main__":
    unittest.main()
