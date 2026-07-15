#!/usr/bin/env python3
"""Run ARC's instrumented tests and emit LLVM text and LCOV reports."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys


def run(command: list[str], *, env: dict[str, str] | None = None, output=None) -> None:
    subprocess.run(command, check=True, env=env, stdout=output)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--build-dir", type=Path, default=Path("out/build/coverage-clang"))
    parser.add_argument("--output-dir", type=Path, default=Path("out/coverage"))
    args = parser.parse_args()
    build_dir = args.build_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    for tool in ("ctest", "llvm-profdata", "llvm-cov"):
        if shutil.which(tool) is None:
            raise RuntimeError(f"required coverage tool is unavailable: {tool}")

    raw_pattern = output_dir / "arc-%p-%m.profraw"
    environment = os.environ.copy()
    environment["LLVM_PROFILE_FILE"] = str(raw_pattern)
    run(["ctest", "--test-dir", str(build_dir), "--output-on-failure"], env=environment)

    raw_profiles = sorted(output_dir.glob("*.profraw"))
    if not raw_profiles:
        raise RuntimeError("tests produced no LLVM raw profiles")
    profile = output_dir / "arc.profdata"
    run(["llvm-profdata", "merge", "-sparse", *map(str, raw_profiles), "-o", str(profile)])

    suffix = ".exe" if os.name == "nt" else ""
    test_binaries = sorted(path for path in build_dir.rglob(f"*tests{suffix}") if path.is_file())
    if not test_binaries:
        raise RuntimeError("no ARC test binaries were found in the coverage build")
    objects: list[str] = []
    for binary in test_binaries:
        objects.extend(["-object", str(binary)])

    report = output_dir / "summary.txt"
    with report.open("w", encoding="utf-8") as stream:
        run(["llvm-cov", "report", *objects, f"-instr-profile={profile}"], output=stream)
    lcov = output_dir / "coverage.lcov"
    with lcov.open("w", encoding="utf-8") as stream:
        run([
            "llvm-cov", "export", *objects, f"-instr-profile={profile}", "-format=lcov",
            "-ignore-filename-regex=(third_party|out/build)",
        ], output=stream)
    print(f"coverage reports written to {output_dir}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (RuntimeError, subprocess.CalledProcessError) as error:
        print(f"coverage generation failed: {error}", file=sys.stderr)
        raise SystemExit(1)
