import subprocess
import unittest

from subbridge.errors import (
    ClaudeNotInstalledError,
    ClaudeProtocolError,
    ClaudeTurnError,
    CodexNotAuthenticatedError,
    CodexProcessError,
    CodexTurnError,
    SubBridgeError,
)
from subbridge.gateway._errors import GatewayError, error_body, from_exception


def timed_out() -> CodexProcessError:
    try:
        try:
            raise subprocess.TimeoutExpired(["codex"], 1)
        except subprocess.TimeoutExpired as exc:
            raise CodexProcessError("Codex timed out after 1 seconds.") from exc
    except CodexProcessError as error:
        return error


class StatusTests(unittest.TestCase):
    def test_exceptions_map_to_documented_statuses(self) -> None:
        cases = [
            (ClaudeTurnError("API Error: usage limit reached"), 429),
            (CodexTurnError("You've hit your usage limit."), 429),
            (CodexTurnError("model_not_found: gpt-nope"), 404),
            (ClaudeTurnError("model is not available on your plan"), 403),
            (ClaudeTurnError("boom"), 502),
            (ClaudeNotInstalledError("missing"), 503),
            (CodexNotAuthenticatedError("signed out"), 503),
            (CodexProcessError("Codex CLI exited with code 7."), 502),
            (timed_out(), 504),
            (ClaudeProtocolError("bad JSONL"), 502),
        ]
        for exc, status in cases:
            with self.subTest(exc=exc):
                self.assertEqual(from_exception(exc).status, status)

    def test_unexpected_errors_do_not_leak_details(self) -> None:
        error = from_exception(SubBridgeError("/private/secret"))
        self.assertEqual(error.status, 500)
        self.assertNotIn("/private/secret", error.message)

    def test_process_error_with_rate_limit_text_is_a_429(self) -> None:
        error = from_exception(
            CodexProcessError("Codex CLI exited with code 1: quota exceeded")
        )
        self.assertEqual(error.status, 429)


class BodyTests(unittest.TestCase):
    def test_each_api_gets_its_own_error_shape(self) -> None:
        error = GatewayError(429, "slow down", code="rate_limit_exceeded")
        self.assertEqual(
            error_body("anthropic", error),
            {
                "type": "error",
                "error": {"type": "rate_limit_error", "message": "slow down"},
            },
        )
        self.assertEqual(
            error_body("openai", error),
            {
                "error": {
                    "message": "slow down",
                    "type": "rate_limit_exceeded",
                    "param": None,
                    "code": "rate_limit_exceeded",
                }
            },
        )


if __name__ == "__main__":
    unittest.main()
