#!/usr/bin/env python3
"""Check first-party C/C++ formatting with a pinned clang-format executable."""

from __future__ import annotations

import argparse
import pathlib
import subprocess
import sys


EXTENSIONS = {".c", ".cc", ".cpp", ".cxx", ".h", ".hh", ".hpp", ".hxx", ".inl"}
ROOTS = ("engine", "editor/native", "samples", "tools", "benchmarks", "tests")
EXCLUDED_PARTS = {"third_party", "out", "_deps"}
EXCLUDED_FILES = {"engine/render-vulkan/src/builtin_shaders.h"}


def first_party_sources(root: pathlib.Path) -> list[pathlib.Path]:
    result: list[pathlib.Path] = []
    for source_root in ROOTS:
        absolute_root = root / source_root
        if not absolute_root.exists():
            continue
        for absolute_path in absolute_root.rglob("*"):
            if not absolute_path.is_file():
                continue
            path = absolute_path.relative_to(root)
            if path.suffix.lower() not in EXTENSIONS:
                continue
            if path.as_posix() in EXCLUDED_FILES:
                continue
            if any(part in EXCLUDED_PARTS for part in path.parts):
                continue
            result.append(path)
    return sorted(result)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--clang-format", default="clang-format-18")
    parser.add_argument("--fix", action="store_true")
    args = parser.parse_args()
    root = pathlib.Path(__file__).resolve().parents[1]
    sources = first_party_sources(root)
    if not sources:
        print("No first-party C/C++ sources found", file=sys.stderr)
        return 1
    # TEMPORARY L1.5e formatter diagnostics; remove before merge.
    diagnostic = [
        "engine/network/src/dev/live_play_dev_client.cpp",
        "engine/platform/windows/src/windows_live_play_inspection.h",
        "engine/platform/windows/src/windows_entry.cpp",
        "engine/network/test_support/live_play_dev_tests.cpp",
        "engine/platform/windows/tests/windows_live_play_edits_tests.cpp",
    ]
    subprocess.check_call([args.clang_format, "-i", *diagnostic], cwd=root)
    subprocess.check_call(["git", "diff", "--", *diagnostic], cwd=root)
    if args.fix:
        command = [args.clang_format, "-i", *map(str, sources)]
    else:
        command = [args.clang_format, "--dry-run", "--Werror", *map(str, sources)]
    return subprocess.call(command, cwd=root)


if __name__ == "__main__":
    raise SystemExit(main())
