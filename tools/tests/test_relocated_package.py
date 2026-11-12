"""Regression tests for the D3 relocated production-package smoke runner."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
from unittest import TestCase, main, mock
import tempfile
import zipfile

SCRIPT = Path(__file__).resolve().parents[1] / "verify_relocated_package.py"
spec = importlib.util.spec_from_file_location("verify_relocated_package", SCRIPT)
assert spec is not None and spec.loader is not None
smoke_runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(smoke_runner)


class DistributionFixture(TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory(prefix="arc-d3-test-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.distribution = self.root / "original-distribution"
        self.distribution.mkdir()
        self.files = [
            ("bin/Game.exe", "executable", b"synthetic-executable"),
            ("Config/runtime.json", "configuration", b"{}"),
            ("Content/game.arccookmanifest", "content", b"cooked-manifest"),
            ("Content/game.arcpak", "content", b"cooked-payload"),
        ]
        self.manifest = {
            "format": "arc.distribution",
            "formatVersion": 1,
            "projectGuid": "11111111-1111-1111-1111-111111111111",
            "engineVersion": "0.1.0",
            "sdkVersion": "0.1.0",
            "target": {
                "platform": "windows",
                "architecture": "x86_64",
                "renderer": "vulkan",
                "configuration": "Shipping",
                "cookProfile": "windows-x64-vulkan",
                "id": "windows-x64-vulkan",
            },
            "entrypoint": {"kind": "executable", "value": "bin/Game.exe"},
            "runtimeConfig": "Config/runtime.json",
            "content": {"cookManifests": ["Content/game.arccookmanifest"]},
            "role": "client",
            "files": [],
        }
        self.reset_files()

    def reset_files(self) -> None:
        inventory = []
        for path, kind, data in self.files:
            target = self.distribution / path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
            inventory.append({
                "path": path, "kind": kind, "size": len(data),
                "sha256": hashlib.sha256(data).hexdigest(),
            })
        self.manifest["files"] = inventory
        self.write_manifest()

    def write_manifest(self) -> None:
        (self.distribution / "arc-distribution.json").write_text(
            json.dumps(self.manifest, indent=2), encoding="utf-8"
        )

    def test_validates_real_manifest_inventory_and_rejects_tampering(self) -> None:
        assert smoke_runner.verify_distribution(self.distribution)["role"] == "client"
        (self.distribution / "Content/game.arcpak").write_bytes(b"tampered")
        with self.assertRaisesRegex(smoke_runner.SmokeFailure, "size mismatch"):
            smoke_runner.verify_distribution(self.distribution)
        self.reset_files()
        (self.distribution / "Source").mkdir()
        (self.distribution / "Source/main.cpp").write_text("authoring leak")
        with self.assertRaisesRegex(smoke_runner.SmokeFailure, "undeclared file"):
            smoke_runner.verify_distribution(self.distribution)

    def test_rejects_unsafe_inventory_and_symbol_leaks(self) -> None:
        self.manifest["files"][0]["path"] = "../escape.exe"
        self.write_manifest()
        with self.assertRaisesRegex(smoke_runner.SmokeFailure, "unsafe package path"):
            smoke_runner.verify_distribution(self.distribution)
        self.reset_files()
        (self.distribution / "bin/Game.pdb").write_bytes(b"symbols")
        with self.assertRaisesRegex(smoke_runner.SmokeFailure, "undeclared file"):
            smoke_runner.verify_distribution(self.distribution)

    def test_runner_launches_relocated_entry_from_unrelated_cwd(self) -> None:
        artifacts = self.root / "results"
        with mock.patch.object(smoke_runner.subprocess, "run") as launch:
            launch.return_value = mock.Mock(returncode=0, stdout="boot complete", stderr="")
            smoke_runner.smoke(self.distribution, None, 5, ["--ci-smoke"], artifacts, "boot complete")
        self.assertEqual(launch.call_count, 1)
        args, kwargs = launch.call_args
        self.assertTrue(args[0][0].endswith("Game.exe"))
        self.assertEqual(args[0][1:], ["--ci-smoke"])
        self.assertNotEqual(kwargs["cwd"], self.distribution)
        self.assertNotEqual(kwargs["cwd"], Path(args[0][0]).parent)
        self.assertTrue(Path(args[0][0]).is_absolute())
        self.assertEqual(json.loads((artifacts / "launch.json").read_text())["exitCode"], 0)
        self.assertTrue((artifacts / "arc-distribution.json").is_file())

    def test_archive_extraction_and_server_default_ticks(self) -> None:
        self.manifest["role"] = "server"
        self.manifest["target"]["renderer"] = "none"
        self.write_manifest()
        archive = self.root / "server.zip"
        with zipfile.ZipFile(archive, "w") as output:
            for file in self.distribution.rglob("*"):
                if file.is_file():
                    output.write(file, file.relative_to(self.distribution).as_posix())
        with mock.patch.object(smoke_runner.subprocess, "run") as launch:
            launch.return_value = mock.Mock(returncode=0, stdout="Server stopped after 10 simulation ticks",
                                            stderr="")
            smoke_runner.smoke(None, archive, 5, [], expected_output="10 simulation ticks")
        self.assertEqual(launch.call_args.args[0][1:], ["--ticks", "10", "--no-sleep"])

    def test_launch_failure_and_timeout_are_reported(self) -> None:
        with mock.patch.object(smoke_runner.subprocess, "run") as launch:
            launch.return_value = mock.Mock(returncode=1, stdout="", stderr="missing startup scene")
            with self.assertRaisesRegex(smoke_runner.SmokeFailure, "missing startup scene"):
                smoke_runner.smoke(self.distribution, None, 2, [])
            launch.side_effect = smoke_runner.subprocess.TimeoutExpired(["Game.exe"], 2)
            with self.assertRaisesRegex(smoke_runner.SmokeFailure, "exited 124"):
                smoke_runner.smoke(self.distribution, None, 2, [])

    def test_archive_rejects_path_traversal(self) -> None:
        archive = self.root / "unsafe.zip"
        with zipfile.ZipFile(archive, "w") as output:
            output.writestr("../escape.txt", "unsafe")
        with self.assertRaisesRegex(smoke_runner.SmokeFailure, "unsafe package path"):
            smoke_runner.smoke(None, archive, 5, [])


if __name__ == "__main__":
    main()
