"""subbridge.use_subscription(): the one-line in-process entry point."""

import importlib.util
import os
import unittest
import warnings
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

# The gateway tests need the (dev-only) anthropic and openai SDKs; the
# release workflow installs only requirements-release.txt, so skip cleanly
# there instead of failing to collect this module.
if (
    importlib.util.find_spec("anthropic") is None
    or importlib.util.find_spec("openai") is None
):
    raise unittest.SkipTest(
        "anthropic and openai are not installed; skipping gateway tests."
    )

import anthropic
import openai
from test_claude import fake_claude
from test_codex import fake_codex

import subbridge
import subbridge.gateway as gateway_module

GATEWAY_VARIABLES = (
    "ANTHROPIC_BASE_URL",
    "ANTHROPIC_API_KEY",
    "OPENAI_BASE_URL",
    "OPENAI_API_KEY",
)

_PROXY_ENV_NAMES = (
    "HTTP_PROXY",
    "http_proxy",
    "HTTPS_PROXY",
    "https_proxy",
    "ALL_PROXY",
    "all_proxy",
    "NO_PROXY",
    "no_proxy",
)


class UseSubscriptionTestCase(unittest.TestCase):
    """Fake `claude`/`codex` on PATH, plus a proxy-free, gateway-var-free
    environment so a leftover shell setting can't make a test flaky."""

    def setUp(self) -> None:
        tempdir = TemporaryDirectory()
        self.addCleanup(tempdir.cleanup)
        bin_dir = Path(tempdir.name)
        fake_claude(bin_dir)
        fake_codex(bin_dir)
        path = f"{bin_dir}{os.pathsep}{os.environ.get('PATH', '')}"
        env_patch = mock.patch.dict(os.environ, {"PATH": path}, clear=False)
        env_patch.start()
        self.addCleanup(env_patch.stop)
        for name in _PROXY_ENV_NAMES:
            os.environ.pop(name, None)
        # Whatever this test starts, close it -- close() is idempotent, and
        # this also clears the module-level singleton so the next test in
        # the suite always starts from "no gateway running".
        self.addCleanup(self._close_any_running_gateway)

    def _close_any_running_gateway(self) -> None:
        gateway = gateway_module._use_subscription_singleton
        if gateway is not None:
            gateway.close()


class EnvironmentAndRequestsTests(UseSubscriptionTestCase):
    def test_sets_exactly_the_four_sdk_variables(self) -> None:
        gateway = subbridge.use_subscription()
        for name in GATEWAY_VARIABLES:
            self.assertIn(name, os.environ)
        self.assertEqual(os.environ["ANTHROPIC_BASE_URL"], gateway.anthropic_base_url)
        self.assertEqual(os.environ["ANTHROPIC_API_KEY"], gateway.api_key)
        self.assertEqual(os.environ["OPENAI_BASE_URL"], gateway.openai_base_url)
        self.assertEqual(os.environ["OPENAI_API_KEY"], gateway.api_key)

    def test_anthropic_client_with_no_arguments_completes_a_request(self) -> None:
        subbridge.use_subscription()
        client = anthropic.Anthropic()
        message = client.messages.create(
            model="sonnet",
            max_tokens=100,
            messages=[{"role": "user", "content": "hello"}],
        )
        self.assertEqual(message.content[0].text, "answer: hello")

    def test_openai_client_with_no_arguments_completes_a_request(self) -> None:
        subbridge.use_subscription()
        client = openai.OpenAI()
        completion = client.chat.completions.create(
            model="gpt-test", messages=[{"role": "user", "content": "hello"}]
        )
        self.assertEqual(completion.choices[0].message.content, "answer: hello")


class IdempotencyTests(UseSubscriptionTestCase):
    def test_second_call_returns_the_same_gateway_and_changes_nothing(self) -> None:
        first = subbridge.use_subscription()
        env_after_first = {name: os.environ[name] for name in GATEWAY_VARIABLES}
        second = subbridge.use_subscription(port=1234, max_concurrency=99, timeout=1)
        self.assertIs(first, second)
        for name in GATEWAY_VARIABLES:
            self.assertEqual(os.environ[name], env_after_first[name])


class CloseRestoresEnvironmentTests(UseSubscriptionTestCase):
    def test_close_restores_a_preset_value_and_removes_an_unset_one(self) -> None:
        os.environ["ANTHROPIC_BASE_URL"] = "https://pre-existing.example"
        os.environ.pop("OPENAI_API_KEY", None)

        gateway = subbridge.use_subscription()
        self.assertEqual(os.environ["ANTHROPIC_BASE_URL"], gateway.anthropic_base_url)
        self.assertIn("OPENAI_API_KEY", os.environ)

        gateway.close()
        self.assertEqual(
            os.environ["ANTHROPIC_BASE_URL"], "https://pre-existing.example"
        )
        self.assertNotIn("OPENAI_API_KEY", os.environ)

    def test_a_call_after_close_starts_a_fresh_gateway_with_a_new_key(self) -> None:
        first = subbridge.use_subscription()
        first_key = first.api_key
        first.close()

        second = subbridge.use_subscription()
        self.assertIsNot(first, second)
        self.assertNotEqual(second.api_key, first_key)

    def test_close_twice_is_safe(self) -> None:
        gateway = subbridge.use_subscription()
        gateway.close()
        gateway.close()  # must not raise


class AtexitRegistrationTests(UseSubscriptionTestCase):
    def test_atexit_registers_the_cleanup_callback_exactly_once(self) -> None:
        with (
            mock.patch.object(
                gateway_module, "_use_subscription_atexit_registered", False
            ),
            mock.patch("subbridge.gateway.atexit.register") as register,
        ):
            first = subbridge.use_subscription()
            self.assertEqual(register.call_count, 1)
            # A second call while the gateway is still running must not
            # register a second cleanup callback.
            second = subbridge.use_subscription()
            self.assertIs(first, second)
            self.assertEqual(register.call_count, 1)

            first.close()
            subbridge.use_subscription()
            # Registration happens once ever, not once per gateway.
            self.assertEqual(register.call_count, 1)


class ProxyWarningTests(UseSubscriptionTestCase):
    def test_warns_when_a_proxy_lacks_a_no_proxy_exemption(self) -> None:
        os.environ["HTTP_PROXY"] = "http://proxy.example:8080"
        with self.assertWarns(UserWarning) as caught:
            subbridge.use_subscription()
        self.assertIn("NO_PROXY", str(caught.warning))

    def test_no_warning_without_any_proxy_variable(self) -> None:
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            subbridge.use_subscription()  # must not raise/warn

    def test_no_warning_once_127_0_0_1_is_exempted(self) -> None:
        os.environ["ALL_PROXY"] = "http://proxy.example:8080"
        os.environ["NO_PROXY"] = "localhost,127.0.0.1"
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            subbridge.use_subscription()  # must not raise/warn


if __name__ == "__main__":
    unittest.main()
