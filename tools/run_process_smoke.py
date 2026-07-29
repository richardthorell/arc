#!/usr/bin/env python3
"""Run a packaged ARC product with a deterministic timeout."""

from __future__ import annotations

import argparse
import os
import pathlib
import subprocess
import sys
import tempfile


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("executable", type=pathlib.Path)
    parser.add_argument("arguments", nargs=argparse.REMAINDER)
    args = parser.parse_args()

    executable = args.executable.resolve()
    if not executable.is_file():
        print(f"Smoke executable does not exist: {executable}", file=sys.stderr)
        return 2

    command = [str(executable), *args.arguments]
    environment = os.environ.copy()
    # Developer shells launched from Electron tooling can inherit this flag.
    # A packaged Electron executable must run as the application, not as Node.
    environment.pop("ELECTRON_RUN_AS_NODE", None)
    with tempfile.TemporaryDirectory(prefix="arc-smoke-") as temporary_directory:
        diagnostic_path = pathlib.Path(temporary_directory) / "diagnostic.txt"
        environment["ARC_CI_SMOKE_LOG"] = str(diagnostic_path)
        try:
            completed = subprocess.run(command, env=environment, timeout=args.timeout)
        except subprocess.TimeoutExpired:
            if diagnostic_path.is_file():
                diagnostic = diagnostic_path.read_text(encoding="utf-8").strip()
                if diagnostic:
                    print(f"Smoke diagnostic: {diagnostic}", file=sys.stderr)
            print(f"Smoke process exceeded {args.timeout:.1f}s: {executable}", file=sys.stderr)
            return 124
        if completed.returncode and diagnostic_path.is_file():
            diagnostic = diagnostic_path.read_text(encoding="utf-8").strip()
            if diagnostic:
                print(f"Smoke diagnostic: {diagnostic}", file=sys.stderr)
    if completed.returncode:
        print(f"Smoke process exited with {completed.returncode}: {executable}", file=sys.stderr)
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
