#!/usr/bin/env python3
"""Export the built Android host and Gradle launcher into an installed ARC SDK.

This exports the *host*, not project-specific native code or cooked content.
The existing Android Gradle application remains the source of the native binaries.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import struct
import tempfile
from zipfile import ZipFile

ABI = "arm64-v8a"
MAX_LIBRARY_BYTES = 512 * 1024 * 1024
# The current native host CMake target pins both engine and SDK to 0.1.0.
HOST_ENGINE_VERSION = "0.1.0"
GRADLE_FILES = (
    "build.gradle.kts",
    "settings.gradle.kts",
    "gradle.properties",
    "app/build.gradle.kts",
    "app/src/main/AndroidManifest.xml",
    "app/src/main/cpp/project_module/CMakeLists.txt",
)


def _read_sdk_version(sdk_root: Path) -> str:
    if not (sdk_root / "lib/cmake/ARC/ARCConfig.cmake").is_file():
        raise ValueError("ARC SDK CMake package is missing")
    installation = sdk_root / "arc-installation.json"
    if not installation.is_file():
        raise ValueError("ARC SDK installation manifest is missing")
    data = json.loads(installation.read_text(encoding="utf-8"))
    version = data.get("engineVersion")
    if data.get("format") != "arc-installation" or not isinstance(version, str) or not version:
        raise ValueError("Invalid ARC SDK installation manifest")
    return version


def _libraries_from_apk(apk_path: Path) -> dict[str, bytes]:
    libraries: dict[str, bytes] = {}
    with ZipFile(apk_path) as apk:
        names = apk.namelist()
        if len(names) != len(set(names)):
            raise ValueError(f"Duplicate APK member in {apk_path}")
        for item in apk.infolist():
            path = PurePosixPath(item.filename)
            if (not item.filename or item.filename.startswith("/") or
                    "\\" in item.filename or ".." in path.parts):
                raise ValueError(f"Unsafe APK member: {item.filename}")
            if len(path.parts) != 3 or path.parts[:2] != ("lib", ABI):
                continue
            if not path.name.endswith(".so"):
                continue
            if item.file_size > MAX_LIBRARY_BYTES:
                raise ValueError(f"Oversized Android library: {path.name}")
            data = apk.read(item)
            if (len(data) < 20 or data[:4] != b"\x7fELF" or data[4] != 2 or
                    data[5] != 1 or struct.unpack_from("<H", data, 18)[0] != 183):
                raise ValueError(f"Android host library is not an arm64 ELF: {path.name}")
            libraries[path.name] = data
    if "libarc_android_host.so" not in libraries:
        raise ValueError(f"APK does not contain the ARC arm64 native host: {apk_path}")
    return libraries


def export_android_host_sdk(sdk_root: Path, android_source: Path, debug_apk: Path, release_apk: Path) -> Path:
    """Stage and atomically publish the Android host under an existing installed SDK."""
    sdk_root = sdk_root.resolve(strict=True)
    android_source = android_source.resolve(strict=True)
    version = _read_sdk_version(sdk_root)
    if version != HOST_ENGINE_VERSION:
        raise ValueError(f"Android host {HOST_ENGINE_VERSION} does not match installed ARC SDK {version}")
    libraries = {"debug": _libraries_from_apk(debug_apk), "release": _libraries_from_apk(release_apk)}
    finalizer_source = Path(__file__).with_name("android_apk_finalizer.py")
    if not finalizer_source.is_file():
        raise ValueError("Android SDK export requires the A4 APK finalizer")
    for relative in GRADLE_FILES:
        if not (android_source / relative).is_file():
            raise ValueError(f"Missing Android Gradle launcher file: {relative}")
    destination = sdk_root / "share/arc/android"
    if destination.exists():
        raise ValueError(f"Android host SDK already exists, refusing to overwrite: {destination}")

    parent = destination.parent
    parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".android-host-", dir=parent) as temporary:
        staged = Path(temporary) / "android"
        for relative in GRADLE_FILES:
            source = android_source / relative
            target = staged / "gradle" / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
        shutil.copyfile(finalizer_source, staged / "android_apk_finalizer.py")
        configurations: dict[str, list[dict[str, str]]] = {}
        for configuration, entries in libraries.items():
            output = staged / "host" / configuration / "jniLibs" / ABI
            output.mkdir(parents=True)
            configurations[configuration] = []
            for filename, data in sorted(entries.items()):
                (output / filename).write_bytes(data)
                configurations[configuration].append({
                    "name": filename,
                    "sha256": hashlib.sha256(data).hexdigest(),
                })
        descriptor = {
            "format": "arc.android-host-sdk",
            "version": 1,
            "engineVersion": version,
            "abi": ABI,
            "configurations": configurations,
        }
        (staged / "host/host.json").write_text(
            json.dumps(descriptor, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        os.replace(staged, destination)
    return destination


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sdk-root", required=True, type=Path)
    parser.add_argument("--android-source", required=True, type=Path)
    parser.add_argument("--debug-apk", required=True, type=Path)
    parser.add_argument("--release-apk", required=True, type=Path)
    args = parser.parse_args()
    try:
        result = export_android_host_sdk(
            args.sdk_root, args.android_source, args.debug_apk, args.release_apk
        )
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        parser.exit(1, f"ARC Android host SDK export failed: {exc}\n")
    print(f"Exported Android Gradle host and arm64 native libraries to {result}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
