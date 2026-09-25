"""Drive POST /v1/messages with the official anthropic SDK against fake Claude Code."""

import os
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

    def test_stream_helper_yields_text_and_final_message(self) -> None:
        with self.anthropic.messages.stream(
            model="sonnet", max_tokens=100, messages=USER_HELLO
        ) as stream:
            texts = list(stream.text_stream)
            final = stream.get_final_message()
        self.assertEqual("".join(texts), "answer: hello")
        self.assertEqual(final.content[0].text, "answer: hello")
        self.assertEqual(final.stop_reason, "end_turn")
        self.assertEqual(final.usage.output_tokens, 7)

    def test_raw_stream_sends_the_documented_event_order(self) -> None:
        events = list(self.create(stream=True))
        self.assertEqual(
            [event.type for event in events],
            [
                "message_start",
                "content_block_start",
                "content_block_delta",
                "content_block_stop",
                "message_delta",
                "message_stop",
            ],
        )

    def test_usage_limit_is_429_when_streaming_too(self) -> None:
        with self.assertRaises(anthropic.RateLimitError) as raised:
            self.create(messages=[{"role": "user", "content": "is-error"}], stream=True)
        self.assertIn("usage limit reached", raised.exception.message)

    def test_failure_after_text_arrives_as_an_sse_error_event(self) -> None:
        texts = []
        with self.assertRaises(anthropic.APIStatusError) as raised:
            for event in self.create(
                messages=[{"role": "user", "content": "partial-then-error"}],
                stream=True,
            ):
                if event.type == "content_block_delta":
                    texts.append(event.delta.text)
        self.assertEqual(texts, ["partial"])
        self.assertEqual(raised.exception.body["error"]["type"], "rate_limit_error")

    @unittest.skipUnless(os.name == "posix", "checks the process by pid")
    def test_client_disconnect_stops_the_cli(self) -> None:
        with self.anthropic.messages.stream(
            model="sonnet",
            max_tokens=100,
            messages=[{"role": "user", "content": "endless"}],
        ) as stream:
            self.assertEqual(next(iter(stream.text_stream)), "tick ")
        self.assert_cli_exits("claude")


if __name__ == "__main__":
    unittest.main()
