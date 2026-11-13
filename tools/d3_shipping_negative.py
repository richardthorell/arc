#!/usr/bin/env python3
"""D3.3 real Shipping distribution negative acceptance suite.

Mutate fresh copies of the D3.2 production package. Missing/corrupt inventory
must be rejected before process launch; validly staged but incompatible
manifests must be rejected by the actual packaged runtime bootstrap.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path, PurePosixPath
import shutil
import sys
import tempfile
from collections.abc import Callable

from verify_relocated_package import SmokeFailure, smoke, verify_distribution


def inventory_path(stage: Path, manifest: dict, *, kind: str, suffix: str = "") -> Path:
    matches = [
        stage.joinpath(*PurePosixPath(item["path"]).parts)
        for item in manifest["files"]
        if item["kind"] == kind and item["path"].casefold().endswith(suffix.casefold())
    ]
    if not matches:
        raise SmokeFailure(f"D3.3 fixture lacks a {kind} inventory entry {suffix!r}")
    return matches[0]


def remove_native_dependency(stage: Path, manifest: dict) -> None:
    # The installed MSVC /MD Shipping client must stage its C++ runtime.
    inventory_path(stage, manifest, kind="library", suffix="msvcp140.dll").unlink()


def remove_cooked_content(stage: Path, manifest: dict) -> None:
    path = PurePosixPath(manifest["content"]["cookManifests"][0])
    stage.joinpath(*path.parts).unlink()


def corrupt_cooked_content(stage: Path, manifest: dict) -> None:
    path = stage.joinpath(*PurePosixPath(manifest["content"]["cookManifests"][0]).parts)
    data = bytearray(path.read_bytes())
    if not data:
        raise SmokeFailure("cannot corrupt an empty cook manifest")
    data[len(data) // 2] ^= 1
    path.write_bytes(data)  # Preserve length; SHA-256 must catch this.


def replace_manifest(stage: Path, change: Callable[[dict], None]) -> None:
    path = stage / "arc-distribution.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    change(data)
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def incompatible_engine(stage: Path, manifest: dict) -> None:
    def mutate(document: dict) -> None:
        document["engineVersion"] = "999.0.0-d3-negative"
    replace_manifest(stage, mutate)


def bad_startup_scene(stage: Path, manifest: dict) -> None:
    if not manifest.get("startupScenes"):
        raise SmokeFailure("D3.3 client fixture has no startupScenes")
    def mutate(document: dict) -> None:
        document["startupScenes"][0] = "aaaaaaaa-bbbb-4ccc-8ddd-eeeeeeeeeeee"
    replace_manifest(stage, mutate)


# Failure messages are checked so a timeout or generic crash cannot pass an
# acceptance case. Runtime failures are checked after Python inventory passes.
CASES: tuple[tuple[str, Callable[[Path, dict], None], str, bool], ...] = (
    ("missing-native-dll", remove_native_dependency, "missing or unsafe inventory file", False),
    ("missing-cooked-manifest", remove_cooked_content, "missing or unsafe inventory file", False),
    ("corrupt-cooked-manifest", corrupt_cooked_content, "file hash mismatch", False),
    ("incompatible-engine", incompatible_engine, "engineVersion", True),
    ("invalid-startup-scene", bad_startup_scene, "startupScenes", True),
)


def run(distribution: Path, artifacts: Path, timeout: float = 20.0) -> None:
    base = verify_distribution(distribution)
    if base["role"] != "client" or base["target"]["configuration"] != "Shipping":
        raise SmokeFailure("D3.3 requires a verified Shipping client package")
    artifacts.mkdir(parents=True, exist_ok=True)
    results: list[dict] = []
    try:
        for name, mutate, expected, runtime_check in CASES:
            with tempfile.TemporaryDirectory(prefix=f"arc-d3-negative-{name}-") as temporary:
                stage = Path(temporary) / "shipping"
                shutil.copytree(distribution, stage)
                mutate(stage, base)
                case_artifacts = artifacts / name
                case_artifacts.mkdir(parents=True, exist_ok=True)
                if runtime_check:
                    # These changes do not violate file inventory. The runtime
                    # must reject incompatible D0 metadata or startup scene.
                    verify_distribution(stage)
                try:
                    smoke(stage, None, timeout, ["--ci-smoke"], case_artifacts)
                except SmokeFailure as failure:
                    detail = str(failure)
                    (case_artifacts / "expected-failure.txt").write_text(detail + "\n", encoding="utf-8")
                    if expected.casefold() not in detail.casefold():
                        raise SmokeFailure(f"{name} failed for the wrong reason: {detail}") from failure
                    # The runtime must genuinely execute and reject the bad
                    # metadata, not just fail Python package verification.
                    if runtime_check:
                        report = case_artifacts / "launch.json"
                        if not report.exists():
                            raise SmokeFailure(f"{name} never reached packaged runtime launch")
                        exit_code = json.loads(report.read_text(encoding="utf-8"))["exitCode"]
                        if exit_code in (0, 124, 127):
                            raise SmokeFailure(f"{name} ended with unexpected exit code {exit_code}")
                    results.append({"case": name, "status": "passed", "expected": expected})
                    print(f"[D3.3] {name}: expected rejection ({expected})", flush=True)
                else:
                    raise SmokeFailure(f"{name}: mutated package unexpectedly launched")
    finally:
        (artifacts / "negative-results.json").write_text(
            json.dumps({"cases": results, "expectedCount": len(CASES)}, indent=2) + "\n",
            encoding="utf-8")
    if len(results) != len(CASES):
        raise SmokeFailure(f"only {len(results)} of {len(CASES)} negative cases passed")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--distribution", required=True, type=Path)
    parser.add_argument("--artifacts", required=True, type=Path)
    parser.add_argument("--timeout", type=float, default=20.0)
    args = parser.parse_args()
    try:
        run(args.distribution.resolve(), args.artifacts.resolve(), args.timeout)
    except (SmokeFailure, OSError, ValueError, KeyError) as exc:
        print(f"D3.3 negative acceptance failed: {exc}", file=sys.stderr)
        return 1
    print("D3.3 production Shipping negative acceptance passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
