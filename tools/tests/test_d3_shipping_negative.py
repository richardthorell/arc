"""Fast tests for D3.3 mutations; production CI also runs them against real binaries."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import d3_shipping_negative as negative
from verify_relocated_package import SmokeFailure, verify_distribution


class NegativeDistributionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.files = {
            "bin/game.exe": ("executable", b"game"),
            "bin/MSVCP140.dll": ("library", b"redist"),
            "Config/runtime.json": ("configuration", b"{}"),
            "Content/game.arccookmanifest": ("content", b"cooked-manifest"),
        }
        entries = []
        for path, (kind, data) in self.files.items():
            dest = self.root / path
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(data)
            entries.append({"path": path, "kind": kind, "size": len(data),
                            "sha256": hashlib.sha256(data).hexdigest()})
        self.manifest = {
            "format": "arc.distribution", "formatVersion": 1,
            "engineVersion": "0.1.0", "sdkVersion": "0.1.0",
            "projectGuid": "11111111-1111-1111-1111-111111111111",
            "target": {"id": "windows-x64-vulkan", "platform": "windows",
                       "architecture": "x86_64", "renderer": "vulkan",
                       "configuration": "Shipping", "cookProfile": "windows-x64-vulkan"},
            "entrypoint": {"kind": "executable", "value": "bin/game.exe"},
            "role": "client", "runtimeConfig": "Config/runtime.json",
            "content": {"cookManifests": ["Content/game.arccookmanifest"]},
            "startupScenes": ["11111111-2222-4333-8444-555555555555"],
            "files": entries,
        }
        self.write_manifest()

    def write_manifest(self) -> None:
        (self.root / "arc-distribution.json").write_text(
            json.dumps(self.manifest), encoding="utf-8")

    def test_baseline_distribution_is_valid(self) -> None:
        self.assertEqual(verify_distribution(self.root)["role"], "client")

    def test_missing_ms_vc_runtime_is_detected(self) -> None:
        negative.remove_native_dependency(self.root, self.manifest)
        with self.assertRaisesRegex(SmokeFailure, "missing or unsafe inventory file"):
            verify_distribution(self.root)

    def test_missing_cooked_manifest_is_detected(self) -> None:
        negative.remove_cooked_content(self.root, self.manifest)
        with self.assertRaisesRegex(SmokeFailure, "missing or unsafe inventory file"):
            verify_distribution(self.root)

    def test_same_size_corruption_detected_by_sha256(self) -> None:
        negative.corrupt_cooked_content(self.root, self.manifest)
        with self.assertRaisesRegex(SmokeFailure, "file hash mismatch"):
            verify_distribution(self.root)

    def test_incompatible_engine_survives_inventory_checks_for_runtime_rejection(self) -> None:
        negative.incompatible_engine(self.root, self.manifest)
        manifest = verify_distribution(self.root)
        self.assertEqual(manifest["engineVersion"], "999.0.0-d3-negative")

    def test_invalid_startup_scene_survives_inventory_checks_for_runtime_rejection(self) -> None:
        negative.bad_startup_scene(self.root, self.manifest)
        manifest = verify_distribution(self.root)
        self.assertEqual(manifest["startupScenes"], ["aaaaaaaa-bbbb-4ccc-8ddd-eeeeeeeeeeee"])

    def test_missing_native_dependency_fixture_fails_closed(self) -> None:
        (self.root / "bin/MSVCP140.dll").unlink()
        with self.assertRaisesRegex(SmokeFailure, "missing or unsafe inventory file"):
            negative.run(self.root, self.root / "results")


if __name__ == "__main__":
    unittest.main()
