#!/usr/bin/env python3
"""D4.1: installed ARC SDK -> external Blank Headless -> Shipping server package.

Reuse the exact production cooker, arc-project package assembler, and D0
relocation validator exercised by the D3 client acceptance suite.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

from d3_shipping_acceptance import command
from d4_headless_negative import run as run_negative
from d4_headless_stop_acceptance import verify_external_stop
from verify_relocated_package import SmokeFailure, smoke, verify_distribution


def require_one(paths: list[Path], description: str) -> Path:
    if len(paths) != 1:
        raise SmokeFailure(f"expected one {description}, found {paths}")
    return paths[0]


def verify_headless_distribution(root: Path) -> dict:
    manifest = verify_distribution(root)
    if manifest["role"] != "server" or manifest["target"]["renderer"] != "none":
        raise SmokeFailure("D4.1 package must declare server role and renderer none")
    if manifest["target"]["platform"] != "windows" or manifest["target"]["configuration"] != "Shipping":
        raise SmokeFailure("D4.1 package must be Windows Shipping")
    if manifest.get("startupScenes"):
        raise SmokeFailure("blank headless package unexpectedly declares startup scenes")
    for item in manifest["files"]:
        name = item["path"].lower()
        if any(token in name for token in ("vulkan", "glfw", "sdl", "electron", "editor", "render-d3d")):
            raise SmokeFailure(f"unexpected graphical/editor dependency: {item['path']}")
    return manifest


def verify_server_report(path: Path, *, ticks: int, configuration: str = "Shipping") -> dict:
    """Validate the actual packaged server's machine-readable shutdown event."""
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
        if len(lines) != 1:
            raise ValueError("expected one JSON event line")
        report = json.loads(lines[0])
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise SmokeFailure(f"invalid server shutdown report: {exc}") from exc
    required = {
        "event": "server.stopped", "role": "server",
        "configuration": configuration, "reason": "tick-limit",
        "requestedTicks": ticks, "completedTicks": ticks,
    }
    for name, expected in required.items():
        if report.get(name) != expected:
            raise SmokeFailure(f"invalid server shutdown report {name}: expected {expected!r}, got {report.get(name)!r}")
    if not report.get("engineVersion"):
        raise SmokeFailure("server shutdown report is missing engineVersion")
    return report


def run(args: argparse.Namespace) -> None:
    workspace = args.workspace.resolve()
    artifacts = args.artifacts.resolve()
    workspace.mkdir(parents=True, exist_ok=True)
    artifacts.mkdir(parents=True, exist_ok=True)
    sdk = args.sdk.resolve()
    cooker = args.cooker.resolve()
    project_tool = sdk / "bin" / "arc-project.exe"
    templates = sdk / "share" / "arc" / "templates"
    if not project_tool.is_file() or not cooker.is_file() or not templates.is_dir():
        raise SmokeFailure("installed SDK, project tool, cooker, or templates missing")

    project = workspace / "external-server"
    build = project / "Build" / "ci-shipping"
    cooked = project / "Build" / "Cooked" / "windows-x64-headless"
    release = workspace / "dist" / "ServerShipping"
    archive = workspace / "dist" / "ServerShipping.zip"

    command("server-create", [str(project_tool), "create", "--name", "ExternalServer",
                              "--destination", str(project), "--template", "blank-headless",
                              "--templates", str(templates), "--engine", args.engine], workspace, artifacts)
    command("server-configure", ["cmake", "-S", str(project), "-B", str(build), "-G", "Ninja",
                                 "-DCMAKE_BUILD_TYPE=Shipping", f"-DCMAKE_PREFIX_PATH={sdk}"],
            workspace, artifacts)
    command("server-compile", ["cmake", "--build", str(build), "--parallel"], workspace, artifacts)
    command("server-cook", [str(cooker), "package", "--project", str(project),
                            "--profile", "windows-x64-headless", "--output", str(cooked)],
            workspace, artifacts)
    command("server-cook-verify", [str(cooker), "verify", "--project", str(project),
                                   "--profile", "windows-x64-headless", "--output", str(cooked)],
            workspace, artifacts)

    executable = build / "ExternalServerServerApp.exe"
    manifest = cooked / "windows-x64-headless.arccookmanifest"
    if not executable.is_file() or not manifest.is_file():
        raise SmokeFailure(f"server build or cooked manifest missing: {executable}, {manifest}")

    # Production binaries import MSVC redistributable libraries on clean hosts.
    # Supply the narrow redistributable directory for recursive PE closure.
    redist = os.environ.get("VCToolsRedistDir")
    if not redist:
        raise SmokeFailure("VCToolsRedistDir missing for headless dependency closure")
    crt = require_one(sorted(path for path in Path(redist).resolve().glob("x64/Microsoft.VC*.CRT")
                             if path.is_dir() and (path / "msvcp140.dll").is_file()),
                      "x64 MSVC CRT redistributable directory")
    command("server-package", [
        str(project_tool), "package", "--server", "--project", str(project / "ExternalServer.arcproject"),
        "--profile", "windows-x64-headless", "--executable", str(executable),
        "--runtime-config", str(project / "Config" / "Renderer.json"),
        "--cook-manifest", str(manifest), "--output", str(release),
        "--zip", str(archive), "--dll-dir", str(build), "--dll-dir", str(sdk / "bin"),
        "--dll-dir", str(crt),
    ], workspace, artifacts)
    staged = verify_headless_distribution(release)

    # No checkout, SDK, or graphical window: launch from another directory,
    # execute exactly 10 fixed simulation ticks, then shut down cleanly.
    for label, distribution, zipped in (("directory", release, None), ("zip", None, archive)):
        smoke(distribution, zipped, args.timeout, [],
              artifacts / label, expected_output="Server stopped after 10 simulation ticks")

    # D4.3: verify compiled version reporting and deterministic lifecycle
    # metadata through the relocated production executable, not a mock host.
    smoke(release, None, args.timeout, ["--version"], artifacts / "version",
          expected_output=f"ARC server engine {args.engine} (Shipping)")
    smoke(release, None, args.timeout, ["--ticks", "10", "--no-sleep", "--report-json"],
          artifacts / "json-report", expected_output='"event":"server.stopped"')
    report = verify_server_report(artifacts / "json-report" / "stdout.txt", ticks=10)
    if report["engineVersion"] != args.engine:
        raise SmokeFailure(f"server reported incompatible engineVersion: {report['engineVersion']}")

    # D4.4: exercise a long-running production process and request graceful
    # termination from outside the packaged distribution.
    verify_external_stop(release, artifacts / "external-stop", timeout=args.timeout)

    # D4.2: the real relocated server must reject corrupt/missing content,
    # missing native dependencies, incompatible engine identity and bad flags.
    run_negative(release, artifacts / "negative", args.timeout)

    shutil.copyfile(release / "arc-distribution.json", artifacts / "server-distribution.json")
    (artifacts / "server-result.json").write_text(json.dumps({
        "status": "passed", "role": "server", "renderer": "none",
        "target": staged["target"], "ticks": 10, "relocation": ["directory", "zip"],
        "package": str(release), "archive": str(archive)
    }, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sdk", required=True, type=Path)
    parser.add_argument("--cooker", required=True, type=Path)
    parser.add_argument("--workspace", required=True, type=Path)
    parser.add_argument("--artifacts", required=True, type=Path)
    parser.add_argument("--engine", default="0.1.0")
    parser.add_argument("--timeout", type=float, default=30.0)
    args = parser.parse_args()
    try:
        run(args)
    except (SmokeFailure, OSError, ValueError, subprocess.SubprocessError) as exc:
        args.artifacts.mkdir(parents=True, exist_ok=True)
        (args.artifacts / "server-failure.txt").write_text(str(exc) + "\n", encoding="utf-8")
        print(f"D4.1 headless Shipping failure: {exc}", file=sys.stderr)
        return 1
    print("D4.1 external headless server package and relocation passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
