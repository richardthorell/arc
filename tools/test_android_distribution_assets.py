#!/usr/bin/env python3
"""A3 Gradle contract fixture: staged D0 APK assets without a second cooker run.

This fixture verifies Gradle's hash/identity/asset handoff. The native
arc-android-device-smoke CTest separately verifies actual ARC-cooked packages.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from zipfile import ZIP_STORED, ZipFile

SCENE = "b80a0000-0000-4000-8000-000000000005"


def prepare(project: Path, sdk: Path, directory: Path) -> None:
    project = project.resolve(strict=True)
    sdk = sdk.resolve(strict=True)
    directory.mkdir(parents=True, exist_ok=False)
    descriptor = next(project.glob("*.arcproject"))
    metadata = json.loads(descriptor.read_text(encoding="utf-8"))
    version = json.loads((sdk / "arc-installation.json").read_text(encoding="utf-8"))["engineVersion"]
    payloads = {
        "Config/runtime.json": b'{"tickRate":60}\n',
        "Content/android-arm64-vulkan-debug.arccookmanifest": b'{"fixture":"gradle-only"}\n',
        "Content/startup.arcpak": b"ARC Android A3 Gradle asset layout fixture",
    }
    inventory = []
    for relative, data in sorted(payloads.items()):
        path = directory / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        inventory.append({
            "path": relative,
            "kind": "configuration" if relative.startswith("Config/") else "content",
            "size": len(data),
            "sha256": hashlib.sha256(data).hexdigest(),
        })
    document = {
        "format": "arc.distribution",
        "formatVersion": 1,
        "projectGuid": metadata["guid"],
        "engineVersion": version,
        "sdkVersion": version,
        "target": {
            "id": "android-arm64-vulkan",
            "platform": "android",
            "architecture": "arm64",
            "renderer": "vulkan",
            "configuration": "Debug",
            "cookProfile": "android-arm64-vulkan-debug",
        },
        "role": "client",
        "finalizer": "android-gradle",
        "entrypoint": {"kind": "android-activity", "value": "android.app.NativeActivity"},
        "startupScenes": [SCENE],
        "runtimeConfig": "Config/runtime.json",
        "content": {"cookManifests": ["Content/android-arm64-vulkan-debug.arccookmanifest"]},
        "files": inventory,
    }
    (directory / "arc-distribution.json").write_text(
        json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def verify(apk: Path, stage: Path) -> None:
    document = json.loads((stage / "arc-distribution.json").read_text(encoding="utf-8"))
    with ZipFile(apk) as zipped:
        names = set(zipped.namelist())
        assert "assets/arc-distribution.json" in names, "Missing Android D0 manifest in APK"
        assert zipped.read("assets/arc-distribution.json") == (stage / "arc-distribution.json").read_bytes()
        for entry in document["files"]:
            relative = entry["path"]
            name = "assets/" + relative
            assert name in names, f"Missing cooked asset in APK: {name}"
            assert zipped.read(name) == (stage / relative).read_bytes(), f"Corrupt asset in APK: {name}"
            if relative.endswith((".arcpak", ".arccookmanifest")):
                assert zipped.getinfo(name).compress_type == ZIP_STORED, (
                    f"Cooked asset must support uncompressed AAssetManager access: {name}"
                )
    print(f"Verified {len(document['files'])} D0 cooked-content assets in {apk}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_subparsers(dest="action", required=True)
    prep = action.add_parser("prepare")
    prep.add_argument("--project", required=True, type=Path)
    prep.add_argument("--sdk", required=True, type=Path)
    prep.add_argument("--output", required=True, type=Path)
    check = action.add_parser("verify")
    check.add_argument("--apk", required=True, type=Path)
    check.add_argument("--stage", required=True, type=Path)
    args = parser.parse_args()
    if args.action == "prepare":
        prepare(args.project, args.sdk, args.output)
    else:
        verify(args.apk, args.stage)


if __name__ == "__main__":
    main()
