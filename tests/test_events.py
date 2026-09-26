import unittest

from subbridge._events import normalize_event


class EventNormalizationTests(unittest.TestCase):
    def test_claude_message_normalizes_without_raw_payload_by_default(self) -> None:
        raw = {
            "type": "assistant",
            "session_id": "claude-session",
            "message": {"content": [{"type": "text", "text": "hello"}]},
            "prompt": "private input",
        }
        event = normalize_event("claude", raw)
        self.assertEqual(event.kind, "message")
        self.assertEqual(event.provider, "claude")
        self.assertEqual(event.thread_id, "claude-session")
        self.assertEqual(event.text, "hello")
        self.assertIsNone(event.raw)

    def test_codex_completion_normalizes_usage_and_raw_is_explicit(self) -> None:
        raw = {
            "type": "turn.completed",
            "usage": {"input_tokens": 10, "cached_input_tokens": 3},
        }
        event = normalize_event(
            "codex", raw, include_raw=True, thread_id="codex-thread"
        )
        self.assertEqual(event.kind, "turn_completed")
        self.assertEqual(event.provider, "codex")
        self.assertEqual(event.thread_id, "codex-thread")
        self.assertEqual(event.usage.input_tokens, 10)
        self.assertEqual(event.usage.cached_input_tokens, 3)
        self.assertIs(event.raw, raw)

    def test_claude_is_error_result_normalizes_as_turn_error(self) -> None:
        raw = {"type": "result", "subtype": "success", "is_error": True}
        self.assertEqual(normalize_event("claude", raw).kind, "turn_error")


if __name__ == "__main__":
    unittest.main()
