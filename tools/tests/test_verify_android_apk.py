"""Unit checks for Android APK structure validation."""
from pathlib import Path
from tempfile import TemporaryDirectory
import struct
import sys
import unittest
from zipfile import ZipFile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from verify_android_apk import verify_apk


def elf(machine: int, *, elf_class: int = 2) -> bytes:
    value = bytearray(20)
    value[:4] = b"\x7fELF"
    value[4] = elf_class
    value[5] = 1
    struct.pack_into("<H", value, 18, machine)
    return bytes(value)


class VerifyAndroidApkTests(unittest.TestCase):
    def test_valid_arm64_host(self):
        with TemporaryDirectory() as temp:
            file = Path(temp) / "host.apk"
            with ZipFile(file, "w") as archive:
                archive.writestr("AndroidManifest.xml", b"manifest")
                archive.writestr("lib/arm64-v8a/libarc_android_host.so", elf(183))
            verify_apk(file, "arm64-v8a")

    def test_wrong_abi_fails_closed(self):
        with TemporaryDirectory() as temp:
            file = Path(temp) / "wrong.apk"
            with ZipFile(file, "w") as archive:
                archive.writestr("AndroidManifest.xml", b"manifest")
                archive.writestr("lib/arm64-v8a/libarc_android_host.so", elf(62))
            with self.assertRaisesRegex(ValueError, "ELF machine"):
                verify_apk(file, "arm64-v8a")

    def test_missing_native_host_is_rejected(self):
        with TemporaryDirectory() as temp:
            file = Path(temp) / "empty.apk"
            with ZipFile(file, "w") as archive:
                archive.writestr("AndroidManifest.xml", b"manifest")
            with self.assertRaisesRegex(ValueError, "missing required entry"):
                verify_apk(file, "arm64-v8a")

    def test_duplicate_members_are_rejected(self):
        with TemporaryDirectory() as temp:
            file = Path(temp) / "dup.apk"
            with ZipFile(file, "w") as archive:
                archive.writestr("AndroidManifest.xml", b"manifest")
                archive.writestr("AndroidManifest.xml", b"another")
                archive.writestr("lib/arm64-v8a/libarc_android_host.so", elf(183))
            with self.assertRaisesRegex(ValueError, "duplicate"):
                verify_apk(file, "arm64-v8a")


if __name__ == "__main__":
    unittest.main()
