"""D4.1 headless distribution contract tests, no Windows toolchain required."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from d4_headless_shipping_acceptance import require_one, verify_headless_distribution
from verify_relocated_package import SmokeFailure


class HeadlessDistributionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory(prefix="arc-d4-test-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.manifest = {
            "format": "arc.distribution", "formatVersion": 1,
            "projectGuid": "11111111-1111-1111-1111-111111111111",
            "engineVersion": "0.1.0", "sdkVersion": "0.1.0",
            "target": {
                "id": "windows-x64-headless", "platform": "windows",
                "architecture": "x86_64", "renderer": "none",
                "configuration": "Shipping", "cookProfile": "windows-x64-headless",
            },
            "entrypoint": {"kind": "executable", "value": "bin/Server.exe"},
            "runtimeConfig": "Config/runtime.json",
            "content": {"cookManifests": ["Content/windows-x64-headless.arccookmanifest"]},
            "startupScenes": [], "role": "server", "files": [],
        }
        self.add_file("bin/Server.exe", "executable", b"headless")
        self.add_file("Config/runtime.json", "configuration", b'{"backend":"none"}')
        self.add_file("Content/windows-x64-headless.arccookmanifest", "content", b"manifest")
        self.write_manifest()

    def add_file(self, path: str, kind: str, data: bytes) -> None:
        target = self.root / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
        self.manifest["files"].append({
            "path": path, "kind": kind, "size": len(data),
            "sha256": hashlib.sha256(data).hexdigest(),
        })

    def write_manifest(self) -> None:
        (self.root / "arc-distribution.json").write_text(
            json.dumps(self.manifest), encoding="utf-8")

    def test_valid_headless_distribution(self) -> None:
        self.assertEqual(verify_headless_distribution(self.root)["role"], "server")

    def test_rejects_client_role_and_graphical_renderer(self) -> None:
        self.manifest["role"] = "client"
        self.manifest["target"]["renderer"] = "vulkan"
        self.write_manifest()
        with self.assertRaisesRegex(SmokeFailure, "server role"):
            verify_headless_distribution(self.root)

    def test_rejects_unexpected_startup_scene(self) -> None:
        self.manifest["startupScenes"] = ["11111111-2222-4333-8444-555555555555"]
        self.write_manifest()
        with self.assertRaisesRegex(SmokeFailure, "startup scenes"):
            verify_headless_distribution(self.root)

    def test_rejects_graphical_dll_even_with_valid_inventory(self) -> None:
        self.add_file("bin/arc-render-vulkan.dll", "library", b"renderer")
        self.write_manifest()
        with self.assertRaisesRegex(SmokeFailure, "graphical/editor dependency"):
            verify_headless_distribution(self.root)

    def test_rejects_missing_cooked_content(self) -> None:
        (self.root / "Content/windows-x64-headless.arccookmanifest").unlink()
        with self.assertRaisesRegex(SmokeFailure, "missing or unsafe inventory file"):
            verify_headless_distribution(self.root)

    def test_one_expected_file_contract(self) -> None:
        self.assertEqual(require_one([self.root / "bin/Server.exe"], "server"),
                         self.root / "bin/Server.exe")
        with self.assertRaisesRegex(SmokeFailure, "expected one"):
            require_one([], "server")


if __name__ == "__main__":
    unittest.main()
