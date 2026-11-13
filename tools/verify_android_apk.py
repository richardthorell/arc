#!/usr/bin/env python3
"""Verify that a produced ARC Android APK includes the expected native host ABI.

This is a package/layout smoke check, not an emulator/runtime boot test. The
Android build currently produces arm64-v8a only, so an x86-64 runner emulator
cannot launch the APK without first building a matching x86-64 variant.
"""
from __future__ import annotations

import argparse
from pathlib import Path, PurePosixPath
import struct
import sys
from zipfile import BadZipFile, ZipFile

SUPPORTED_ABI = {
    "arm64-v8a": (2, 183),  # ELFCLASS64, EM_AARCH64
    "x86_64": (2, 62),     # ELFCLASS64, EM_X86_64
}


def verify_apk(path: Path, abi: str) -> None:
    if abi not in SUPPORTED_ABI:
        raise ValueError(f"unsupported CI ABI: {abi}")
    if not path.is_file():
        raise ValueError(f"APK not found: {path}")
    with ZipFile(path) as apk:
        names = apk.namelist()
        if len(names) != len(set(names)):
            raise ValueError("APK has duplicate ZIP members")
        for name in names:
            member = PurePosixPath(name)
            if not name or name.startswith("/") or "\\" in name or ".." in member.parts:
                raise ValueError(f"unsafe APK member: {name}")
        for required in ("AndroidManifest.xml", f"lib/{abi}/libarc_android_host.so"):
            if required not in names:
                raise ValueError(f"APK missing required entry: {required}")
        native = apk.read(f"lib/{abi}/libarc_android_host.so")
        if len(native) < 20 or native[:4] != b"\x7fELF":
            raise ValueError("ARC Android native host is not an ELF shared library")
        expected_class, expected_machine = SUPPORTED_ABI[abi]
        if native[4] != expected_class or native[5] != 1:
            raise ValueError(f"ARC Android native host has incorrect ELF class/endianness for {abi}")
        machine = struct.unpack_from("<H", native, 18)[0]
        if machine != expected_machine:
            raise ValueError(f"ARC Android native host has ELF machine {machine}, expected {expected_machine}")
    print(f"Android APK verified: {path} ({abi}, native host ELF)")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("apk", type=Path)
    parser.add_argument("--abi", choices=sorted(SUPPORTED_ABI), default="arm64-v8a")
    args = parser.parse_args()
    try:
        verify_apk(args.apk, args.abi)
    except (ValueError, OSError, BadZipFile) as exc:
        print(f"Android APK validation failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
