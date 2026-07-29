#!/usr/bin/env python3
"""Validate ARC distribution contents and reject embedded local paths."""

from __future__ import annotations

import argparse
import pathlib
import re
import sys


TEXT_SUFFIXES = {".json", ".txt", ".ini", ".cfg", ".manifest", ".arccookmanifest"}
LOCAL_PATHS = (
    re.compile(rb"[A-Za-z]:[\\/](?:Users|Code|src|build)[\\/]", re.IGNORECASE),
    re.compile(rb"/(?:home|Users|tmp)/[^/\x00]+/", re.IGNORECASE),
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=pathlib.Path)
    parser.add_argument("--require", action="append", default=[])
    args = parser.parse_args()

    root = args.root.resolve()
    if not root.is_dir():
        print(f"Distribution directory does not exist: {root}", file=sys.stderr)
        return 2

    failures: list[str] = []
    for required in args.require:
        if not (root / required).exists():
            failures.append(f"missing required entry: {required}")

    for path in root.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in TEXT_SUFFIXES:
            continue
        data = path.read_bytes()
        for pattern in LOCAL_PATHS:
            match = pattern.search(data)
            if match:
                failures.append(
                    f"{path.relative_to(root).as_posix()}: embedded local path {match.group().decode(errors='replace')}"
                )
                break

    if failures:
        print("\n".join(failures), file=sys.stderr)
        return 1
    print(f"Verified ARC distribution: {root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
