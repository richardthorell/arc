#!/usr/bin/env python3
"""Reject floating dependencies and lockfile drift."""

from __future__ import annotations

import argparse
import json
import pathlib
import re
import sys


EXACT_NPM_VERSION = re.compile(r"^\d+\.\d+\.\d+(?:-[0-9A-Za-z.-]+)?$")
FLOATING_GIT_TAG = re.compile(r"\bGIT_TAG\s+(?:master|main|latest)\b", re.IGNORECASE)
ACTION_USE = re.compile(r"^\s*(?:-\s*)?uses:\s*([^@\s]+)@([^\s#]+)", re.MULTILINE)
FULL_COMMIT = re.compile(r"^[0-9a-fA-F]{40}$")
FLOATING_RUNNER = re.compile(r"^\s*runs-on:\s*(?:ubuntu|windows|macos)-latest\s*$", re.MULTILINE)


def check_npm(root: pathlib.Path, errors: list[str]) -> None:
    package = json.loads((root / "editor/package.json").read_text(encoding="utf-8"))
    lock = json.loads((root / "editor/package-lock.json").read_text(encoding="utf-8"))
    locked_root = lock.get("packages", {}).get("", {})
    for group in ("dependencies", "devDependencies"):
        declared = package.get(group, {})
        locked_declared = locked_root.get(group, {})
        for name, version in declared.items():
            if not EXACT_NPM_VERSION.fullmatch(version):
                errors.append(f"editor/package.json: {name} must use an exact version, found {version!r}")
            if locked_declared.get(name) != version:
                errors.append(f"editor/package-lock.json: {name} does not match package.json")


def check_cmake(root: pathlib.Path, errors: list[str]) -> None:
    for path in list(root.rglob("CMakeLists.txt")) + list(root.rglob("*.cmake")):
        if any(part in {"out", ".git"} for part in path.relative_to(root).parts):
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        if FLOATING_GIT_TAG.search(text):
            errors.append(f"{path.relative_to(root)}: floating FetchContent GIT_TAG")


def check_actions(root: pathlib.Path, errors: list[str]) -> None:
    workflows = root / ".github/workflows"
    for path in [*workflows.glob("*.yml"), *workflows.glob("*.yaml")]:
        text = path.read_text(encoding="utf-8")
        for owner, revision in ACTION_USE.findall(text):
            if owner.startswith("./"):
                continue
            if not FULL_COMMIT.fullmatch(revision):
                errors.append(f"{path.relative_to(root)}: {owner}@{revision} is not pinned to a commit")
        if FLOATING_RUNNER.search(text):
            errors.append(f"{path.relative_to(root)}: hosted runner family must be pinned")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=pathlib.Path, default=pathlib.Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    root = args.root.resolve()
    errors: list[str] = []
    check_npm(root, errors)
    check_cmake(root, errors)
    check_actions(root, errors)
    if errors:
        print("\n".join(errors), file=sys.stderr)
        return 1
    print("Dependency declarations are deterministic")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
