"""A4 APK finalizer contract: identity, native closure, signatures and atomicity."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import struct
import tempfile
import unittest
from unittest import mock
from zipfile import ZIP_STORED, ZipFile

from tools import android_apk_finalizer as apk

GUID = "a1111111-1111-4111-8111-111111111111"
SCENE = "b80a0000-0000-4000-8000-000000000005"


def arm64(tag: bytes) -> bytes:
    data = bytearray(64)
    data[:4] = b"\x7fELF"
    data[4] = 2
    data[5] = 1
    struct.pack_into("<H", data, 18, 183)
    return bytes(data) + tag


class FinalizerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.project = self.root / "Project"
        self.sdk = self.root / "SDK"
        self.assets = self.root / "Assets"
        for directory in (self.project, self.sdk, self.assets):
            directory.mkdir()

        self.host = arm64(b"host")
        self.module = arm64(GUID.encode() + b" GameAlpha.runtime")
        sdk_library = self.sdk / "share/arc/android/host/debug/jniLibs/arm64-v8a"
        sdk_library.mkdir(parents=True)
        (sdk_library / apk.HOST_NAME).write_bytes(self.host)
        (self.sdk / "share/arc/android/host/host.json").write_text(json.dumps({
            "format": "arc.android-host-sdk", "version": 1, "abi": apk.ABI,
            "engineVersion": "0.1.0",
            "configurations": {"debug": [{"name": apk.HOST_NAME, "sha256": apk.sha(self.host)}]},
        }))
        (self.sdk / "arc-installation.json").write_text(json.dumps({
            "format": "arc-installation", "engineVersion": "0.1.0",
        }))
        config = self.sdk / "lib/cmake/ARC/ARCConfig.cmake"
        config.parent.mkdir(parents=True)
        config.write_text("# fixture\n")
        launcher = self.sdk / "share/arc/android/gradle/app"
        (launcher / "src/main/cpp/project_module").mkdir(parents=True)
        (launcher / "build.gradle.kts").write_text("plugins { id(\"com.android.application\") }\n")
        (launcher / "src/main/cpp/project_module/CMakeLists.txt").write_text("project(fixture)\n")

        self.settings = {
            "format": "arc.android-project-settings", "version": 1,
            "projectGuid": GUID, "applicationId": "com.example.alpha",
            "displayName": "GameAlpha", "versionName": "2.3", "versionCode": 12,
            "minimumSdk": 26, "targetSdk": 35,
        }
        self.settings_path = self.root / "settings.json"
        self.settings_path.write_text(json.dumps(self.settings))
        self.descriptor = {
            "format": "arc-project", "formatVersion": 3, "guid": GUID,
            "name": "GameAlpha", "engineVersion": "0.1.0",
            "platformSettings": {"android": {
                "applicationId": self.settings["applicationId"],
                "versionName": self.settings["versionName"],
                "versionCode": self.settings["versionCode"],
                "minimumSdk": self.settings["minimumSdk"],
                "targetSdk": self.settings["targetSdk"],
            }},
            "package": {"applicationName": self.settings["displayName"]},
            "startupScenes": [{"guid": SCENE, "expectedType": "scene",
                               "pathHint": "Content/Scenes/Startup.arcscene"}],
        }
        self.descriptor_path = self.project / "GameAlpha.arcproject"
        self.descriptor_path.write_text(json.dumps(self.descriptor))
        self.files = {
            "Config/runtime.json": b'{"tickRate":60}\n',
            "Content/android-debug.arccookmanifest": b'{"fixture":true}\n',
            "Content/startup.arcpak": b"cooked package fixture",
        }
        for relative, data in self.files.items():
            destination = self.assets / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(data)
        self.d0 = {
            "format": "arc.distribution", "formatVersion": 1,
            "projectGuid": GUID, "engineVersion": "0.1.0", "sdkVersion": "0.1.0",
            "target": {
                "id": "android-arm64-vulkan", "platform": "android", "architecture": "arm64",
                "renderer": "vulkan", "configuration": "Debug", "cookProfile": "android-debug",
            },
            "role": "client", "finalizer": "android-gradle",
            "entrypoint": {"kind": "android-activity", "value": "android.app.NativeActivity"},
            "startupScenes": [SCENE], "runtimeConfig": "Config/runtime.json",
            "content": {"cookManifests": ["Content/android-debug.arccookmanifest"]},
            "files": [
                {"path": relative, "kind": "configuration" if relative.startswith("Config/") else "content",
                 "size": len(data), "sha256": apk.sha(data)}
                for relative, data in sorted(self.files.items())
            ],
        }
        self.write_d0()

    def write_d0(self) -> None:
        (self.assets / "arc-distribution.json").write_text(json.dumps(self.d0, sort_keys=True))

    def make_apk(self, path: Path, missing: str | None = None, corrupt: str | None = None,
                 foreign_abi: bool = False) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with ZipFile(path, "w") as zipped:
            binary = {
                f"lib/{apk.ABI}/{apk.HOST_NAME}": self.host,
                f"lib/{apk.ABI}/{apk.PROJECT_NAME}": self.module,
            }
            if foreign_abi:
                binary["lib/x86_64/libunexpected.so"] = arm64(b"wrong")
            sources = {
                "AndroidManifest.xml": b"binary XML fixture",
                "assets/arc-distribution.json": (self.assets / "arc-distribution.json").read_bytes(),
                **{"assets/" + name: data for name, data in self.files.items()},
                **binary,
            }
            for member, data in sources.items():
                if member == missing:
                    continue
                zipped.writestr(member, b"changed" if member == corrupt else data,
                                compress_type=ZIP_STORED)

    def tool_result(self, command: list[str], **kwargs):
        if command[0] == "/mock/aapt":
            return mock.Mock(returncode=0, stderr="", stdout=(
                "package: name='com.example.alpha' versionCode='12' versionName='2.3'\n"
                "sdkVersion:'26'\ntargetSdkVersion:'35'\n"
                "application-label:'GameAlpha'\n"
                "launchable-activity: name='android.app.NativeActivity' label='GameAlpha'\n"
                "native-code: 'arm64-v8a'\n"
            ))
        if command[0] == "/mock/apksigner":
            return mock.Mock(returncode=0, stdout="Verified\n", stderr="")
        if ":app:assembleDebug" in command:
            directory = Path(command[command.index("--project-dir") + 1])
            self.make_apk(directory / "app/build/outputs/apk/debug/app-debug.apk")
            return mock.Mock(returncode=0)
        raise AssertionError(f"Unexpected command: {command}")

    def test_valid_inputs_and_complete_signed_apk(self):
        descriptor, document, _, host = apk.validate_inputs(
            self.project, self.sdk, self.assets, self.settings
        )
        self.assertEqual(descriptor["guid"], GUID)
        output = self.root / "Packages/android-debug"
        with mock.patch.object(apk, "android_tools",
                               return_value=(Path("/mock/aapt"), Path("/mock/apksigner"))), \
             mock.patch.object(apk.subprocess, "run", side_effect=self.tool_result):
            result = apk.finalize(self.project, self.sdk, self.assets, output,
                                  None, self.settings_path, "gradle")
        self.assertTrue(result["succeeded"])
        self.assertTrue(Path(result["apk"]).is_file())
        report = json.loads(Path(result["manifest"]).read_text())
        self.assertEqual(report["format"], "arc.android-apk-finalization")
        self.assertEqual(report["formatVersion"], 1)
        self.assertEqual(report["projectGuid"], GUID)
        self.assertTrue(report["signatureVerified"])
        self.assertEqual(len(report["assets"]), len(self.files))
        self.assertEqual(len(report["nativeLibraries"]), 2)
        self.assertEqual(apk.sha(Path(result["apk"]).read_bytes()), report["apk"]["sha256"])
        with self.assertRaisesRegex(apk.FinalizationError, "already exists"):
            apk.finalize(self.project, self.sdk, self.assets, output,
                         None, self.settings_path, "gradle")
        self.assertEqual(list(output.parent.glob("android-debug.pending")), [])

    def test_mismatched_and_corrupted_assets_fail_closed(self):
        self.d0["projectGuid"] = "b2222222-2222-4222-8222-222222222222"
        self.write_d0()
        with self.assertRaisesRegex(apk.FinalizationError, "GUID mismatch"):
            apk.validate_inputs(self.project, self.sdk, self.assets, self.settings)
        self.d0["projectGuid"] = GUID
        self.write_d0()
        (self.assets / "Content/startup.arcpak").write_bytes(b"tampered")
        with self.assertRaisesRegex(apk.FinalizationError, "hash/size mismatch"):
            apk.validate_inputs(self.project, self.sdk, self.assets, self.settings)

    def test_missing_library_and_wrong_abi_rejected(self):
        with mock.patch.object(apk.subprocess, "run", side_effect=self.tool_result):
            for missing, foreign in [(f"lib/{apk.ABI}/{apk.PROJECT_NAME}", False), (None, True)]:
                test_apk = self.root / f"test-{missing is not None}.apk"
                self.make_apk(test_apk, missing=missing, foreign_abi=foreign)
                with self.assertRaises(apk.FinalizationError):
                    apk.verify_apk(test_apk, self.assets, self.d0, self.settings,
                                   {apk.HOST_NAME: apk.sha(self.host)},
                                   Path("/mock/aapt"), Path("/mock/apksigner"))

    def test_corrupted_cooked_entry_and_unsigned_apk_rejected(self):
        test_apk = self.root / "test.apk"
        self.make_apk(test_apk, corrupt="assets/Content/startup.arcpak")
        with mock.patch.object(apk.subprocess, "run", side_effect=self.tool_result):
            with self.assertRaisesRegex(apk.FinalizationError, "cooked asset differs"):
                apk.verify_apk(test_apk, self.assets, self.d0, self.settings,
                               {apk.HOST_NAME: apk.sha(self.host)},
                               Path("/mock/aapt"), Path("/mock/apksigner"))
        self.make_apk(test_apk)

        def unsigned(cmd, **kwargs):
            if cmd[0] == "/mock/apksigner":
                return mock.Mock(returncode=1, stdout="", stderr="unsigned APK")
            return self.tool_result(cmd, **kwargs)

        with mock.patch.object(apk.subprocess, "run", side_effect=unsigned):
            with self.assertRaisesRegex(apk.FinalizationError, "valid Debug signature"):
                apk.verify_apk(test_apk, self.assets, self.d0, self.settings,
                               {apk.HOST_NAME: apk.sha(self.host)},
                               Path("/mock/aapt"), Path("/mock/apksigner"))

    def test_native_activity_uses_compiled_manifest_when_badging_omits_launch(self):
        test_apk = self.root / "native-activity.apk"
        self.make_apk(test_apk)

        def inspect(command, **kwargs):
            if command[0] == "/mock/aapt" and command[1:3] == ["dump", "badging"]:
                return mock.Mock(returncode=0, stderr="", stdout=(
                    "package: name='com.example.alpha' versionCode='12' versionName='2.3'\n"
                    "sdkVersion:'26'\ntargetSdkVersion:'35'\n"
                ))
            if command[0] == "/mock/aapt" and command[1:3] == ["dump", "xmltree"]:
                return mock.Mock(returncode=0, stderr="", stdout=(
                    'E: application\n'
                    '  A: android:label(0x01010001)="GameAlpha" (Raw: "GameAlpha")\n'
                    '  E: activity\n'
                    '    A: android:name(0x01010003)="android.app.NativeActivity"\n'
                    '    A: android:name(0x01010003)="android.intent.action.MAIN"\n'
                    '    A: android:name(0x01010003)="android.intent.category.LAUNCHER"\n'
                ))
            return self.tool_result(command, **kwargs)

        with mock.patch.object(apk.subprocess, "run", side_effect=inspect):
            result = apk.verify_apk(test_apk, self.assets, self.d0, self.settings,
                                    {apk.HOST_NAME: apk.sha(self.host)},
                                    Path("/mock/aapt"), Path("/mock/apksigner"))
        self.assertTrue(result["signatureVerified"])

    def test_failed_assembly_does_not_publish(self):
        output = self.root / "Packages/fail"
        with mock.patch.object(apk, "android_tools",
                               return_value=(Path("/mock/aapt"), Path("/mock/apksigner"))), \
             mock.patch.object(apk.subprocess, "run", return_value=mock.Mock(returncode=5)):
            with self.assertRaisesRegex(apk.FinalizationError, "Gradle Debug build failed"):
                apk.finalize(self.project, self.sdk, self.assets, output,
                             None, self.settings_path, "gradle")
        self.assertFalse(output.exists())
        self.assertFalse(Path(str(output) + ".pending").exists())
        self.assertFalse(list(output.parent.glob(".arc-apk-*")))

    def test_release_cooked_profile_rejected(self):
        self.d0["target"]["configuration"] = "Shipping"
        self.write_d0()
        with self.assertRaisesRegex(apk.FinalizationError, "Debug ARM64"):
            apk.validate_inputs(self.project, self.sdk, self.assets, self.settings)


if __name__ == "__main__":
    unittest.main()
