#!/usr/bin/env python3
"""D4.4 graceful external-stop acceptance against the real relocated server."""
from __future__ import annotations

import json
import os
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import tempfile
import time

from verify_relocated_package import SmokeFailure, verify_distribution


def validate_stopped_report(report: dict) -> None:
    expected = {"event": "server.stopped", "role": "server", "reason": "external-stop",
                "requestedTicks": 0, "configuration": "Shipping"}
    for key, value in expected.items():
        if report.get(key) != value:
            raise SmokeFailure(f"invalid external stop report {key}: {report.get(key)!r}")
    ticks = report.get("completedTicks")
    if type(ticks) is not int or ticks < 0:
        raise SmokeFailure("external stop report contains invalid completedTicks")
    if not isinstance(report.get("engineVersion"), str) or not report["engineVersion"]:
        raise SmokeFailure("external stop report missing engineVersion")


def verify_external_stop(distribution: Path, artifacts: Path, timeout: float = 20.0) -> None:
    manifest = verify_distribution(distribution)
    if manifest["role"] != "server":
        raise SmokeFailure("external stop requires a packaged server")
    artifacts.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="arc-d4-stop-") as temp:
        workspace = Path(temp)
        stage = workspace / "relocated"
        shutil.copytree(distribution, stage)
        working = workspace / "unrelated"
        working.mkdir()
        writable = workspace / "writable"
        writable.mkdir()
        stop_file = workspace / "stop-request"
        entry = stage.joinpath(*PurePosixPath(manifest["entrypoint"]["value"]).parts)
        env = os.environ.copy()
        env.pop("ELECTRON_RUN_AS_NODE", None)
        env["ARC_SERVER_DATA_ROOT"] = str(writable)
        command = [str(entry), "--report-json", "--stop-file", str(stop_file)]
        process = subprocess.Popen(command, cwd=working, env=env, stdout=subprocess.PIPE,
                                   stderr=subprocess.PIPE, text=True, errors="replace")
        try:
            # Start a real unbounded server and request stop only after it has
            # had time to initialize. An early failure is never a passing test.
            time.sleep(0.5)
            if process.poll() is not None:
                stdout, stderr = process.communicate()
                raise SmokeFailure(f"server exited before stop request ({process.returncode}): {stdout} {stderr}")
            stop_file.write_text("stop\n", encoding="utf-8")
            stdout, stderr = process.communicate(timeout=timeout)
        except subprocess.TimeoutExpired as exc:
            process.kill()
            process.communicate()
            raise SmokeFailure(f"server did not stop within {timeout}s") from exc
        finally:
            if process.poll() is None:
                process.kill()
                process.communicate()
        (artifacts / "stdout.txt").write_text(stdout, encoding="utf-8")
        (artifacts / "stderr.txt").write_text(stderr, encoding="utf-8")
        (artifacts / "launch.json").write_text(json.dumps({
            "exitCode": process.returncode, "requestedStop": "file",
            "stopFile": str(stop_file), "timeoutSeconds": timeout,
        }, indent=2) + "\n", encoding="utf-8")
        if process.returncode != 0:
            raise SmokeFailure(f"server rejected external stop with exit {process.returncode}: {stderr}")
        try:
            lines = stdout.splitlines()
            if len(lines) != 1:
                raise ValueError("expected one JSON shutdown event")
            report = json.loads(lines[0])
        except (ValueError, json.JSONDecodeError) as exc:
            raise SmokeFailure(f"invalid external stop output: {stdout!r}") from exc
        validate_stopped_report(report)
