"""Contract tests for exporting the Android Gradle host into an ARC SDK."""
from __future__ import annotations

import json
from pathlib import Path
import struct
import tempfile
import unittest
from zipfile import ZipFile

from tools.export_android_host_sdk import GRADLE_FILES, export_android_host_sdk


def elf(machine: int = 183) -> bytes:
    data = bytearray(64)
    data[:6] = b"\x7fELF\x02\x01"
    struct.pack_into("<H", data, 18, machine)
    return bytes(data)


class AndroidHostSdkExportTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.sdk = self.root / "sdk"
        (self.sdk / "lib/cmake/ARC").mkdir(parents=True)
        (self.sdk / "lib/cmake/ARC/ARCConfig.cmake").write_text("# installed SDK\n")
        (self.sdk / "arc-installation.json").write_text(
            json.dumps({"format": "arc-installation", "engineVersion": "0.1.0"})
        )
        self.source = self.root / "android"
        for relative in GRADLE_FILES:
            path = self.source / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(f"// {relative}\n")
        self.debug = self.root / "debug.apk"
        self.release = self.root / "release.apk"
        self.apk(self.debug)
        self.apk(self.release)

    @staticmethod
    def apk(path: Path, machine: int = 183, *, duplicate: bool = False, missing: bool = False):
        with ZipFile(path, "w") as output:
            output.writestr("AndroidManifest.xml", b"manifest")
            if not missing:
                output.writestr("lib/arm64-v8a/libarc_android_host.so", elf(machine))
                if duplicate:
                    output.writestr("lib/arm64-v8a/libarc_android_host.so", elf(machine))

    def test_exports_reusable_gradle_launcher_and_hashed_host(self):
        output = export_android_host_sdk(self.sdk, self.source, self.debug, self.release)
        manifest = json.loads((output / "host/host.json").read_text())
        self.assertEqual(manifest["engineVersion"], "0.1.0")
        self.assertEqual(set(manifest["configurations"]), {"debug", "release"})
        self.assertTrue((output / "gradle/app/src/main/AndroidManifest.xml").is_file())
        for relative in GRADLE_FILES:
            self.assertEqual(
                (output / "gradle" / relative).read_bytes(),
                (self.source / relative).read_bytes(),
                f"Android SDK missed the host-side HTTP source: {relative}",
            )

        self.assertTrue((output / "android_apk_finalizer.py").is_file())
        self.assertTrue((output / "android_build_workflow.py").is_file())
        self.assertEqual((output / "host/debug/jniLibs/arm64-v8a/libarc_android_host.so").read_bytes(), elf())
        with self.assertRaisesRegex(ValueError, "refusing to overwrite"):
            export_android_host_sdk(self.sdk, self.source, self.debug, self.release)

    def test_invalid_native_abi_does_not_publish(self):
        self.apk(self.debug, machine=62)
        with self.assertRaisesRegex(ValueError, "not an arm64 ELF"):
            export_android_host_sdk(self.sdk, self.source, self.debug, self.release)
        self.assertFalse((self.sdk / "share/arc/android").exists())

    def test_missing_host_does_not_publish(self):
        self.apk(self.release, missing=True)
        with self.assertRaisesRegex(ValueError, "does not contain"):
            export_android_host_sdk(self.sdk, self.source, self.debug, self.release)
        self.assertFalse((self.sdk / "share/arc/android").exists())

    def test_duplicate_member_rejected(self):
        self.apk(self.debug, duplicate=True)
        with self.assertRaisesRegex(ValueError, "Duplicate APK member"):
            export_android_host_sdk(self.sdk, self.source, self.debug, self.release)

    def test_mismatched_sdk_version_rejected(self):
        (self.sdk / "arc-installation.json").write_text(
            json.dumps({"format": "arc-installation", "engineVersion": "9.0.0"})
        )
        with self.assertRaisesRegex(ValueError, "does not match installed ARC SDK"):
            export_android_host_sdk(self.sdk, self.source, self.debug, self.release)
        self.assertFalse((self.sdk / "share/arc/android").exists())

    def test_missing_installed_sdk_rejected(self):
        (self.sdk / "lib/cmake/ARC/ARCConfig.cmake").unlink()
        with self.assertRaisesRegex(ValueError, "CMake package"):
            export_android_host_sdk(self.sdk, self.source, self.debug, self.release)


if __name__ == "__main__":
    unittest.main()
