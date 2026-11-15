#!/usr/bin/env python3
"""D4.2: negative acceptance for actual packaged, relocated headless servers.

Tests corruption, missing payload and native dependencies, incompatible
runtime identity, and invalid CLI arguments. Each mutation uses a fresh
copy of the D4.1 distribution; production verification remains enabled.
"""
from __future__ import annotations

import json
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import tempfile

from verify_relocated_package import SmokeFailure, smoke, verify_distribution


def inventory_file(root: Path, manifest: dict, kind: str) -> Path:
    matches = [root.joinpath(*PurePosixPath(item["path"]).parts)
               for item in manifest["files"] if item["kind"] == kind]
    if not matches:
        raise SmokeFailure(f"D4.2 fixture has no {kind} inventory file")
    return matches[0]


def check_expected_failure(root: Path, arguments: list[str], artifacts: Path,
                           expected: str, timeout: float, runtime: bool) -> None:
    if runtime:
        verify_distribution(root)
    try:
        smoke(root, None, timeout, arguments, artifacts)
    except SmokeFailure as error:
        message = str(error)
        (artifacts / "expected-failure.txt").write_text(message + "\n", encoding="utf-8")
        if expected.casefold() not in message.casefold():
            raise SmokeFailure(f"Expected {expected!r} failure; received: {message}") from error
        if runtime:
            report = artifacts / "launch.json"
            if not report.is_file():
                raise SmokeFailure(f"Server did not reach runtime launch: {message}")
            exit_code = json.loads(report.read_text(encoding="utf-8"))["exitCode"]
            if exit_code in (0, 124, 127):
                raise SmokeFailure(f"Server failed with unexpected exit code {exit_code}")
    else:
        raise SmokeFailure(f"Mutated server package unexpectedly succeeded ({expected})")


def run(distribution: Path, artifacts: Path, timeout: float) -> None:
    baseline = verify_distribution(distribution)
    if baseline["role"] != "server" or baseline["target"]["renderer"] != "none":
        raise SmokeFailure("D4.2 requires a verified headless server distribution")
    artifacts.mkdir(parents=True, exist_ok=True)
    cases = ("missing-cook-manifest", "corrupt-cook-manifest", "missing-native-library",
             "incompatible-engine", "invalid-ticks")
    passed: list[str] = []
    try:
        for case in cases:
            with tempfile.TemporaryDirectory(prefix="arc-d4-negative-") as temporary:
                root = Path(temporary) / "server"
                shutil.copytree(distribution, root)
                manifest = json.loads((root / "arc-distribution.json").read_text(encoding="utf-8"))
                case_artifacts = artifacts / case
                case_artifacts.mkdir(parents=True, exist_ok=True)
                if case == "missing-cook-manifest":
                    inventory_file(root, manifest, "content").unlink()
                    arguments, expected, runtime = [], "missing or unsafe inventory file", False
                elif case == "corrupt-cook-manifest":
                    target = inventory_file(root, manifest, "content")
                    data = bytearray(target.read_bytes())
                    if not data:
                        raise SmokeFailure("Cannot corrupt an empty manifest")
                    data[len(data) // 2] ^= 1
                    target.write_bytes(data)
                    arguments, expected, runtime = [], "file hash mismatch", False
                elif case == "missing-native-library":
                    inventory_file(root, manifest, "library").unlink()
                    arguments, expected, runtime = [], "missing or unsafe inventory file", False
                elif case == "incompatible-engine":
                    manifest["engineVersion"] = "999.0.0-d4-negative"
                    (root / "arc-distribution.json").write_text(json.dumps(manifest), encoding="utf-8")
                    arguments, expected, runtime = ["--ticks", "10", "--no-sleep"], "engineVersion", True
                else:
                    arguments, expected, runtime = ["--ticks", "invalid"], "--ticks requires", True
                check_expected_failure(root, arguments, case_artifacts, expected, timeout, runtime)
                passed.append(case)
                print(f"[D4.2] {case}: correctly rejected", flush=True)
    finally:
        (artifacts / "negative-results.json").write_text(
            json.dumps({"passed": passed, "expected": list(cases)}, indent=2) + "\n",
            encoding="utf-8")
    if len(passed) != len(cases):
        raise SmokeFailure(f"Only {len(passed)} of {len(cases)} D4.2 cases passed")
