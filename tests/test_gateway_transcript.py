import unittest

from subbridge.gateway._errors import GatewayError
from subbridge.gateway._transcript import (
    content_text,
    flatten,
    json_schema_format,
    message_turns,
    model_name,
    reject_params,
)

ROLES = {"user": "user", "assistant": "assistant"}


class FlattenTests(unittest.TestCase):
    def test_single_user_turn_is_sent_unchanged(self) -> None:
        self.assertEqual(flatten([("user", "hi")]), "hi")

    def test_longer_conversations_become_tagged_turns(self) -> None:
        self.assertEqual(
            flatten([("system", "Be brief."), ("user", "hi")]),
            "<system>\nBe brief.\n</system>\n\n<user>\nhi\n</user>",
        )


class ContentTests(unittest.TestCase):
    def test_text_parts_are_joined_by_newlines(self) -> None:
        parts = [{"type": "text", "text": "a"}, {"type": "input_text", "text": "b"}]
        self.assertEqual(content_text(parts, "content"), "a\nb")

    def test_non_text_part_names_its_position(self) -> None:
        with self.assertRaises(GatewayError) as raised:
            content_text([{"type": "image"}], "messages[0].content")
        self.assertEqual(raised.exception.status, 400)
        self.assertEqual(raised.exception.param, "messages[0].content[0]")

    def test_missing_content_is_rejected(self) -> None:
        with self.assertRaises(GatewayError):
            content_text(None, "messages[0].content")


class MessageTests(unittest.TestCase):
    def test_unknown_role_is_rejected(self) -> None:
        with self.assertRaises(GatewayError) as raised:
            message_turns([{"role": "tool", "content": "x"}], ROLES, "messages")
        self.assertEqual(raised.exception.param, "messages[0]")

    def test_empty_list_is_rejected(self) -> None:
        with self.assertRaises(GatewayError):
            message_turns([], ROLES, "messages")


class ParameterTests(unittest.TestCase):
    def test_model_names_that_look_like_flags_are_rejected(self) -> None:
        self.assertEqual(model_name({"model": "opus[1m]"}), "opus[1m]")
        self.assertIsNone(model_name({}))
        for bad in ("-x", "--model", "", "a b", 7):
            with self.subTest(model=bad), self.assertRaises(GatewayError):
                model_name({"model": bad})

    def test_only_non_empty_unsupported_parameters_are_rejected(self) -> None:
        reject_params({"tools": [], "tool_choice": None}, ("tools", "tool_choice"))
        with self.assertRaises(GatewayError) as raised:
            reject_params({"tools": [{"name": "x"}]}, ("tools",))
        self.assertEqual(raised.exception.code, "unsupported_parameter")

    def test_json_schema_formats(self) -> None:
        schema = {"type": "object"}
        chat = {"type": "json_schema", "json_schema": {"name": "a", "schema": schema}}
        responses = {"type": "json_schema", "name": "a", "schema": schema}
        self.assertEqual(json_schema_format(chat, "response_format"), schema)
        self.assertEqual(json_schema_format(responses, "text.format"), schema)
        self.assertIsNone(json_schema_format({"type": "text"}, "response_format"))
        self.assertIsNone(json_schema_format(None, "response_format"))
        with self.assertRaises(GatewayError):
            json_schema_format({"type": "json_object"}, "response_format")


if __name__ == "__main__":
    unittest.main()
