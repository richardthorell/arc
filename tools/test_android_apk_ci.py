#!/usr/bin/env python3
"""Prepare a real external A2 game fixture for A4's signed APK finalizer CI."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from zipfile import ZipFile

APP_ID = "com.arcengine.gamealpha"
SCENE = "b80a0000-0000-4000-8000-000000000005"


def prepare(project_root: Path, settings_path: Path) -> None:
    descriptor_files = list(project_root.glob("*.arcproject"))
    if len(descriptor_files) != 1:
        raise ValueError("Expected one external A2 game descriptor")
    path = descriptor_files[0]
    project = json.loads(path.read_text(encoding="utf-8"))
    android = {
        "applicationId": APP_ID,
        "versionName": "1.4.2",
        "versionCode": 42,
        "minimumSdk": 26,
        "targetSdk": 35,
    }
    project["platformSettings"] = {"android": android}
    project["package"]["applicationName"] = "GameAlpha"
    project["startupScenes"] = [{
        "guid": SCENE, "expectedType": "scene", "pathHint": "Content/Scenes/Startup.arcscene",
    }]
    path.write_text(json.dumps(project, indent=2) + "\n", encoding="utf-8")
    settings_path.write_text(json.dumps({
        "format": "arc.android-project-settings", "version": 1,
        "projectGuid": project["guid"],
        "displayName": project["package"]["applicationName"],
        **android,
    }, indent=2) + "\n", encoding="utf-8")


def verify(output: Path, project_root: Path) -> None:
    report_path = output / "arc-apk-integrity.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    apk = output / (APP_ID + "-debug.apk")
    if not apk.is_file() or report["apk"]["sha256"] != hashlib.sha256(apk.read_bytes()).hexdigest():
        raise ValueError("A4 output APK/hash mismatch")
    if (report.get("format") != "arc.android-apk-finalization" or report.get("formatVersion") != 1 or
            report.get("applicationId") != APP_ID or report.get("versionCode") != 42 or
            not report.get("signed") or not report.get("signatureVerified") or
            report.get("configuration") != "Debug" or report.get("abi") != "arm64-v8a"):
        raise ValueError("A4 output manifest identity/signature mismatch")
    descriptor = next(project_root.glob("*.arcproject"))
    project = json.loads(descriptor.read_text(encoding="utf-8"))
    if report["projectGuid"] != project["guid"]:
        raise ValueError("A4 output project identity mismatch")
    with ZipFile(apk) as zipped:
        native = [entry["path"] for entry in report["nativeLibraries"]]
        if "lib/arm64-v8a/libarc_android_host.so" not in native or \
                "lib/arm64-v8a/libarc_project_runtime.so" not in native:
            raise ValueError("A4 package did not include native host and game module")
        for entry in report["nativeLibraries"]:
            actual = zipped.read(entry["path"])
            if len(actual) != entry["size"] or hashlib.sha256(actual).hexdigest() != entry["sha256"]:
                raise ValueError(f"Native APK inventory mismatch: {entry['path']}")
        for entry in report["assets"]:
            name = "assets/" + entry["path"]
            actual = zipped.read(name)
            if hashlib.sha256(actual).hexdigest() != entry["sha256"]:
                raise ValueError(f"Cooked APK asset inventory mismatch: {name}")
    print(f"A4 signed Android APK verified: {apk}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    start = sub.add_parser("prepare")
    start.add_argument("--project-root", required=True, type=Path)
    start.add_argument("--settings", required=True, type=Path)
    check = sub.add_parser("verify")
    check.add_argument("--output", required=True, type=Path)
    check.add_argument("--project-root", required=True, type=Path)
    args = parser.parse_args()
    if args.action == "prepare":
        prepare(args.project_root, args.settings)
    else:
        verify(args.output, args.project_root)


if __name__ == "__main__":
    main()
