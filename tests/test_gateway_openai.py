"""Drive the OpenAI endpoints with the official openai SDK against fake Codex."""

import json
import unittest

import openai
from gateway_support import GatewayTestCase
from pydantic import BaseModel

USER_HELLO = [{"role": "user", "content": "hello"}]


class Answer(BaseModel):
    answer: int


class ChatCompletionsTests(GatewayTestCase):
    def create(self, **kwargs):
        options = {"model": "gpt-test", "messages": USER_HELLO}
        return self.openai.chat.completions.create(**{**options, **kwargs})

    def test_create_returns_the_codex_reply_and_usage(self) -> None:
        completion = self.create(temperature=0.2, max_completion_tokens=50, seed=1)
        self.assertEqual(completion.choices[0].message.content, "answer: hello")
        self.assertEqual(completion.choices[0].finish_reason, "stop")
        self.assertEqual(completion.model, "gpt-test")
        self.assertEqual(completion.usage.prompt_tokens, 11)
        self.assertEqual(completion.usage.completion_tokens, 8)
        self.assertEqual(completion.usage.total_tokens, 19)

    def test_system_and_developer_messages_join_the_transcript(self) -> None:
        completion = self.create(
            messages=[
                {"role": "system", "content": "Be brief."},
                {"role": "developer", "content": "Use French."},
                {"role": "user", "content": [{"type": "text", "text": "hi"}]},
            ]
        )
        self.assertEqual(
            completion.choices[0].message.content,
            "answer: <system>\nBe brief.\n</system>\n\n<system>\nUse French.\n"
            "</system>\n\n<user>\nhi\n</user>",
        )

    def test_stream_yields_chunks_until_done(self) -> None:
        chunks = list(self.create(stream=True, stream_options={"include_usage": True}))
        text = "".join(chunk.choices[0].delta.content or "" for chunk in chunks)
        self.assertEqual(text, "answer: hello")
        self.assertEqual(chunks[0].choices[0].delta.role, "assistant")
        self.assertEqual(chunks[-1].choices[0].finish_reason, "stop")

    def test_streamed_text_matches_the_non_streamed_reply(self) -> None:
        messages = [
            {"role": "system", "content": "Réponds en français ✓"},
            {"role": "user", "content": "line one\n\ndata: line two"},
        ]
        whole = self.create(messages=messages).choices[0].message.content
        chunks = self.create(messages=messages, stream=True)
        streamed = "".join(chunk.choices[0].delta.content or "" for chunk in chunks)
        self.assertEqual(streamed, whole)
        self.assertIn("français ✓", whole)
        self.assertIn("\n\ndata: line two", whole)

    def test_json_schema_response_format_reaches_codex(self) -> None:
        completion = self.openai.chat.completions.parse(
            model="gpt-test", messages=USER_HELLO, response_format=Answer
        )
        self.assertEqual(completion.choices[0].message.parsed, Answer(answer=42))
        self.assertIn("--output-schema", self.turn_calls("codex")[0])

    def test_model_reaches_the_codex_cli(self) -> None:
        self.create()
        call = self.turn_calls("codex")[0]
        self.assertEqual(call[call.index("--model") + 1], "gpt-test")

    def test_explicit_null_n_is_accepted(self) -> None:
        body = json.dumps({"model": "gpt-test", "messages": USER_HELLO, "n": None})
        status, payload = self.post(
            "/v1/chat/completions", body.encode(), self.key_headers()
        )
        self.assertEqual(status, 200)
        self.assertEqual(payload["choices"][0]["message"]["content"], "answer: hello")

    def test_non_streamed_reply_keeps_only_the_final_message(self) -> None:
        completion = self.create(messages=[{"role": "user", "content": "two-messages"}])
        self.assertEqual(completion.choices[0].message.content, "answer: two-messages")

    def test_stream_joins_two_agent_messages_with_a_blank_line(self) -> None:
        chunks = list(
            self.create(
                messages=[{"role": "user", "content": "two-messages"}], stream=True
            )
        )
        text = "".join(chunk.choices[0].delta.content or "" for chunk in chunks)
        self.assertEqual(text, "Let me check the file.\n\nanswer: two-messages")

    def test_streamed_structured_output_drops_the_preamble(self) -> None:
        with self.openai.chat.completions.stream(
            model="gpt-test",
            messages=[{"role": "user", "content": "two-messages"}],
            response_format=Answer,
        ) as stream:
            stream.until_done()
            completion = stream.get_final_completion()
        self.assertEqual(completion.choices[0].message.parsed, Answer(answer=42))

    def test_tools_and_multiple_choices_are_rejected_before_any_cli_runs(self) -> None:
        tool = {"type": "function", "function": {"name": "lookup", "parameters": {}}}
        function = {"name": "lookup", "parameters": {}}
        audio_part = {
            "type": "input_audio",
            "input_audio": {"data": "AAAA", "format": "wav"},
        }
        cases = (
            ({"tools": [tool]}, "tools"),
            ({"n": 2}, "n"),
            ({"tool_choice": "auto"}, "tool_choice"),
            ({"functions": [function]}, "functions"),
            ({"function_call": "auto"}, "function_call"),
            ({"audio": {"voice": "alloy", "format": "wav"}}, "audio"),
            ({"response_format": {"type": "json_object"}}, "response_format"),
            (
                {"messages": [{"role": "user", "content": [audio_part]}]},
                "messages[0].content[0]",
            ),
        )
        for options, param in cases:
            with self.subTest(param=param):
                with self.assertRaises(openai.BadRequestError) as raised:
                    self.create(**options)
                self.assertEqual(raised.exception.param, param)
        self.assertEqual(self.cli_calls("codex"), [])

    def test_image_parts_are_rejected(self) -> None:
        image = {
            "type": "image_url",
            "image_url": {"url": "https://example.test/a.png"},
        }
        with self.assertRaises(openai.BadRequestError) as raised:
            self.create(messages=[{"role": "user", "content": [image]}])
        self.assertEqual(raised.exception.param, "messages[0].content[0]")

    def test_usage_limit_is_429_with_and_without_streaming(self) -> None:
        limited = [{"role": "user", "content": "usage-limit"}]
        for stream in (False, True):
            with self.subTest(stream=stream):
                with self.assertRaises(openai.RateLimitError) as raised:
                    self.create(messages=limited, stream=stream)
                self.assertEqual(raised.exception.code, "rate_limit_exceeded")

    def test_failure_after_text_arrives_as_a_stream_error(self) -> None:
        texts = []
        with self.assertRaises(openai.APIError) as raised:
            for chunk in self.create(
                messages=[{"role": "user", "content": "partial-then-error"}],
                stream=True,
            ):
                texts.append(chunk.choices[0].delta.content)
        self.assertIn("partial", texts)
        self.assertIn("usage limit", raised.exception.message)


if __name__ == "__main__":
    unittest.main()
