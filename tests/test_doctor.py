import unittest
from unittest.mock import patch

from subbridge.doctor import collect_doctor_report
from subbridge.models import ProviderCapabilities


class DoctorTests(unittest.TestCase):
    def test_report_exposes_unknown_entitlements_and_usage_explicitly(self) -> None:
        fake = ProviderCapabilities(
            provider="claude", installed=False, authenticated=False
        )
        with (
            patch("subbridge.doctor.ClaudeClient.capabilities", return_value=fake),
            patch("subbridge.doctor.CodexClient.capabilities", return_value=fake),
        ):
            report = collect_doctor_report()
        self.assertEqual(set(report), {"claude", "codex"})
        for provider in report.values():
            self.assertIn("plan_allowed_models", provider)
            self.assertIsNone(provider["plan_allowed_models"])
            self.assertFalse(provider["usage_available"])
            self.assertFalse(provider["model_access_verified"])


if __name__ == "__main__":
    unittest.main()
