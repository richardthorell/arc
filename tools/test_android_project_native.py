#!/usr/bin/env python3
"""External Blank 3D Android native-module fixture and APK acceptance checks.

This exercises project-specific code in the A1 exported SDK Gradle launcher.
It intentionally does not claim a cooked game or physical device smoke.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import struct
from zipfile import ZipFile

PROJECTS = {
    "GameAlpha": "a1111111-1111-4111-8111-111111111111",
    "GameBeta": "b2222222-2222-4222-8222-222222222222",
}
TEMPLATE_FILES = (
    "CMakeLists.txt",
    "cmake/AndroidRuntime.cmake",
    "Source/__PROJECT__Runtime/Runtime.cpp",
    "Source/__PROJECT__Runtime/AndroidModule.cpp",
    "Source/__PROJECT__Runtime/Components.h",
    "__PROJECT__.arcproject",
)


def prepare(repo: Path, sdk: Path, output: Path) -> None:
    repo = repo.resolve(strict=True)
    sdk = sdk.resolve(strict=True)
    output.mkdir(parents=True, exist_ok=True)
    headers = sdk / "include"
    for directory in sorted((repo / "engine").glob("*/inc")):
        if directory.is_dir():
            shutil.copytree(directory, headers, dirs_exist_ok=True)
    reflection = sdk / "tools/project_reflection.py"
    reflection.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(repo / "tools/project_reflection.py", reflection)
    shutil.copy2(repo / "cmake/ARCProjectReflection.cmake",
                 sdk / "lib/cmake/ARC/ARCProjectReflection.cmake")
    template = repo / "templates/blank-3d"
    for name, guid in PROJECTS.items():
        project = output / name
        project.mkdir(exist_ok=True)
        for relative in TEMPLATE_FILES:
            source = template / relative
            destination = project / relative.replace("__PROJECT__", name)
            destination.parent.mkdir(parents=True, exist_ok=True)
            text = source.read_text(encoding="utf-8")
            for previous, replacement in (
                ("{{PROJECT_TOKEN}}", name),
                ("{{PROJECT_NAME}}", name),
                ("{{PROJECT_GUID}}", guid),
                ("{{ENGINE_VERSION}}", "0.1.0"),
            ):
                text = text.replace(previous, replacement)
            destination.write_text(text, encoding="utf-8")


def verify(apk: Path, expected_name: str) -> str:
    if expected_name not in PROJECTS:
        raise ValueError(f"Unknown fixture: {expected_name}")
    with ZipFile(apk) as zipped:
        name = "lib/arm64-v8a/libarc_project_runtime.so"
        if name not in zipped.namelist():
            raise ValueError("Project runtime native library missing from Android APK")
        if "lib/arm64-v8a/libarc_android_host.so" not in zipped.namelist():
            raise ValueError("Android host native library missing from project APK")
        binary = zipped.read(name)
    if (len(binary) < 20 or binary[:4] != b"\x7fELF" or binary[4] != 2
            or binary[5] != 1 or struct.unpack_from("<H", binary, 18)[0] != 183):
        raise ValueError("Project runtime is not an Android arm64 ELF binary")
    if PROJECTS[expected_name].encode() not in binary or (expected_name + ".runtime").encode() not in binary:
        raise ValueError("Project runtime ABI metadata does not match the requested project")
    return hashlib.sha256(binary).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_subparsers(dest="action", required=True)
    creation = action.add_parser("prepare")
    creation.add_argument("--repo", required=True, type=Path)
    creation.add_argument("--sdk", required=True, type=Path)
    creation.add_argument("--output", required=True, type=Path)
    comparison = action.add_parser("verify")
    comparison.add_argument("--alpha", required=True, type=Path)
    comparison.add_argument("--beta", required=True, type=Path)
    args = parser.parse_args()
    if args.action == "prepare":
        prepare(args.repo, args.sdk, args.output)
        print(f"Prepared {len(PROJECTS)} Android native projects in {args.output}")
    else:
        alpha_hash = verify(args.alpha, "GameAlpha")
        beta_hash = verify(args.beta, "GameBeta")
        if alpha_hash == beta_hash:
            raise ValueError("Two different projects produced identical runtime native modules")
        print(json.dumps({"succeeded": True, "alphaSha256": alpha_hash, "betaSha256": beta_hash}))


if __name__ == "__main__":
    main()
