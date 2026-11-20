"""D4.4 external stop report contract."""
from __future__ import annotations

from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from d4_headless_stop_acceptance import validate_stopped_report
from verify_relocated_package import SmokeFailure


class ExternalStopReportTests(unittest.TestCase):
    def setUp(self) -> None:
        self.report = {
            "event": "server.stopped", "role": "server", "reason": "external-stop",
            "requestedTicks": 0, "completedTicks": 29,
            "configuration": "Shipping", "engineVersion": "0.1.0",
        }

    def test_valid_external_stop(self) -> None:
        validate_stopped_report(self.report)

    def test_rejects_tick_limit_instead_of_stop(self) -> None:
        self.report["reason"] = "tick-limit"
        with self.assertRaisesRegex(SmokeFailure, "reason"):
            validate_stopped_report(self.report)

    def test_rejects_invalid_tick_count(self) -> None:
        self.report["completedTicks"] = -1
        with self.assertRaisesRegex(SmokeFailure, "completedTicks"):
            validate_stopped_report(self.report)

    def test_rejects_boolean_tick_count(self) -> None:
        self.report["completedTicks"] = True
        with self.assertRaisesRegex(SmokeFailure, "completedTicks"):
            validate_stopped_report(self.report)

    def test_requires_engine_identity(self) -> None:
        self.report["engineVersion"] = ""
        with self.assertRaisesRegex(SmokeFailure, "engineVersion"):
            validate_stopped_report(self.report)


if __name__ == "__main__":
    unittest.main()
