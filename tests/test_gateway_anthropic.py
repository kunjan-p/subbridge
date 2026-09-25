"""Drive POST /v1/messages with the official anthropic SDK against fake Claude Code."""

import unittest

import anthropic
from gateway_support import GatewayTestCase

USER_HELLO = [{"role": "user", "content": "hello"}]


class MessagesTests(GatewayTestCase):
    def create(self, **kwargs):
        options = {"model": "sonnet", "max_tokens": 100, "messages": USER_HELLO}
        return self.anthropic.messages.create(**{**options, **kwargs})

    def test_create_returns_the_claude_reply_and_usage(self) -> None:
        message = self.create(
            metadata={"user_id": "u1"},
            stop_sequences=["END"],
            extra_body={"temperature": 0.2, "top_p": 0.9},
        )
        self.assertEqual(message.content[0].text, "answer: hello")
        self.assertEqual(message.stop_reason, "end_turn")
        self.assertEqual(message.model, "sonnet")
        self.assertEqual(message.usage.input_tokens, 12)
        self.assertEqual(message.usage.output_tokens, 7)
        self.assertEqual(message.usage.cache_read_input_tokens, 3)

    def test_model_is_passed_to_the_cli(self) -> None:
        message = self.create(
            model="claude-test-model",
            messages=[{"role": "user", "content": "which-model"}],
        )
        self.assertEqual(message.content[0].text, "answer: claude-test-model")

    def test_system_and_history_become_one_transcript(self) -> None:
        message = self.create(
            system="Be brief.",
            messages=[
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": [{"type": "text", "text": "hello"}]},
                {"role": "user", "content": "again"},
            ],
        )
        self.assertEqual(
            message.content[0].text,
            "answer: <system>\nBe brief.\n</system>\n\n<user>\nhi\n</user>\n\n"
            "<assistant>\nhello\n</assistant>\n\n<user>\nagain\n</user>",
        )

    def test_tools_are_rejected_before_any_cli_runs(self) -> None:
        tool = {"name": "lookup", "input_schema": {"type": "object"}}
        with self.assertRaises(anthropic.BadRequestError) as raised:
            self.create(tools=[tool])
        self.assertIn("`tools`", raised.exception.message)
        self.assertEqual(self.cli_calls("claude"), [])

    def test_image_content_is_rejected(self) -> None:
        image = {
            "type": "image",
            "source": {"type": "base64", "media_type": "image/png", "data": "AAAA"},
        }
        with self.assertRaises(anthropic.BadRequestError) as raised:
            self.create(messages=[{"role": "user", "content": [image]}])
        self.assertIn("messages[0].content[0]", raised.exception.message)

    def test_structured_output_is_rejected(self) -> None:
        output_config = {
            "format": {"type": "json_schema", "schema": {"type": "object"}}
        }
        with self.assertRaises(anthropic.BadRequestError) as raised:
            self.create(output_config=output_config)
        self.assertIn("Structured output", raised.exception.message)

    def test_usage_limit_is_429(self) -> None:
        with self.assertRaises(anthropic.RateLimitError) as raised:
            self.create(messages=[{"role": "user", "content": "is-error"}])
        self.assertIn("usage limit reached", raised.exception.message)


if __name__ == "__main__":
    unittest.main()
