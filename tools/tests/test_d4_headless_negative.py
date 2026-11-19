"""D4.2 real-server negative acceptance unit tests."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import d4_headless_negative as negative
from verify_relocated_package import SmokeFailure, verify_distribution


class D4ServerNegativeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        inventory = []
        for path, kind, data in (
            ("bin/Server.exe", "executable", b"binary"),
            ("bin/MSVCP140.dll", "library", b"runtime"),
            ("Config/runtime.json", "configuration", b"{}"),
            ("Content/server.arccookmanifest", "content", b"manifest"),
        ):
            file = self.root / path
            file.parent.mkdir(parents=True, exist_ok=True)
            file.write_bytes(data)
            inventory.append({"path": path, "kind": kind, "size": len(data),
                              "sha256": hashlib.sha256(data).hexdigest()})
        self.manifest = {
            "format": "arc.distribution", "formatVersion": 1,
            "projectGuid": "11111111-1111-1111-1111-111111111111",
            "engineVersion": "0.1.0", "sdkVersion": "0.1.0",
            "target": {"id": "windows-x64-headless", "platform": "windows", "architecture": "x86_64",
                       "renderer": "none", "configuration": "Shipping", "cookProfile": "windows-x64-headless"},
            "entrypoint": {"kind": "executable", "value": "bin/Server.exe"},
            "runtimeConfig": "Config/runtime.json",
            "content": {"cookManifests": ["Content/server.arccookmanifest"]},
            "role": "server", "startupScenes": [], "files": inventory,
        }
        (self.root / "arc-distribution.json").write_text(json.dumps(self.manifest), encoding="utf-8")

    def test_selects_native_dependency_and_content_by_inventory_kind(self) -> None:
        self.assertEqual(negative.inventory_file(self.root, self.manifest, "library").name, "MSVCP140.dll")
        self.assertEqual(negative.inventory_file(self.root, self.manifest, "content").suffix, ".arccookmanifest")

    def test_missing_kind_rejected(self) -> None:
        with self.assertRaisesRegex(SmokeFailure, "no plugin"):
            negative.inventory_file(self.root, self.manifest, "plugin")

    def test_all_negative_cases_mutate_independent_copies(self) -> None:
        with tempfile.TemporaryDirectory(prefix="arc-d4-negative-artifacts-") as output:
            artifacts = Path(output)
            with mock.patch.object(negative, "check_expected_failure") as check:
                negative.run(self.root, artifacts, 5)
            self.assertEqual(check.call_count, 5)
            self.assertEqual(len(json.loads((artifacts / "negative-results.json").read_text())["passed"]), 5)
            self.assertEqual(verify_distribution(self.root)["role"], "server")

    def test_wrong_failure_does_not_count_as_rejection(self) -> None:
        with mock.patch.object(negative, "smoke", side_effect=SmokeFailure("unrelated crash")):
            with self.assertRaisesRegex(SmokeFailure, "Expected"):
                negative.check_expected_failure(self.root, [], self.root / "artifacts",
                                                "file hash mismatch", 5, False)


if __name__ == "__main__":
    unittest.main()
