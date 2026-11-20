"""D4.3 deterministic packaged-server report contract tests."""
from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from d4_headless_shipping_acceptance import verify_server_report
from verify_relocated_package import SmokeFailure


class ServerLifecycleReportTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory(prefix="arc-d4-report-")
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "stdout.txt"
        self.report = {
            "event": "server.stopped", "role": "server", "engineVersion": "0.1.0",
            "configuration": "Shipping", "reason": "tick-limit",
            "requestedTicks": 10, "completedTicks": 10,
        }

    def write(self) -> None:
        self.path.write_text(json.dumps(self.report) + "\n", encoding="utf-8")

    def test_valid_ten_tick_shutdown(self) -> None:
        self.write()
        self.assertEqual(verify_server_report(self.path, ticks=10), self.report)

    def test_wrong_tick_count_is_rejected(self) -> None:
        self.report["completedTicks"] = 9
        self.write()
        with self.assertRaisesRegex(SmokeFailure, "completedTicks"):
            verify_server_report(self.path, ticks=10)

    def test_wrong_stop_reason_is_rejected(self) -> None:
        self.report["reason"] = "runtime-stop"
        self.write()
        with self.assertRaisesRegex(SmokeFailure, "reason"):
            verify_server_report(self.path, ticks=10)

    def test_unstructured_or_multiple_lines_are_rejected(self) -> None:
        self.path.write_text("server started\n" + json.dumps(self.report) + "\n", encoding="utf-8")
        with self.assertRaisesRegex(SmokeFailure, "invalid server shutdown report"):
            verify_server_report(self.path, ticks=10)

    def test_report_requires_engine_identity(self) -> None:
        del self.report["engineVersion"]
        self.write()
        with self.assertRaisesRegex(SmokeFailure, "engineVersion"):
            verify_server_report(self.path, ticks=10)


if __name__ == "__main__":
    unittest.main()
