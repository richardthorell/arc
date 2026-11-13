#!/usr/bin/env python3
"""D3.2 installed-SDK -> external Shipping client -> cook -> package -> relocated boot.

Runs normal ARC commands. No synthetic binary, test-only cooker or editor host is
used. All diagnostics and the final manifests are retained under --artifacts.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import traceback

from verify_relocated_package import SmokeFailure, smoke, verify_distribution
from d3_shipping_negative import run as run_negative_acceptance


def command(label: str, argv: list[str], cwd: Path, artifacts: Path) -> str:
    print(f"[D3.2] {label}: {' '.join(argv)}", flush=True)
    result = subprocess.run(argv, cwd=cwd, capture_output=True, text=True,
                            errors="replace", check=False)
    (artifacts / f"{label}.stdout.txt").write_text(result.stdout, encoding="utf-8")
    (artifacts / f"{label}.stderr.txt").write_text(result.stderr, encoding="utf-8")
    if result.returncode:
        raise SmokeFailure(f"{label} exited {result.returncode}: {result.stderr[-3000:]}")
    return result.stdout


def prepare_minimal_startup_scene(project: Path) -> None:
    """Use a strictly supported authored scene for the boot-only D3 fixture.

    rendering-sample supplies a real startup-scene identity, but its full
    showcase includes legacy authoring fields that are outside the current
    Shipping component schema. Keep the scene's persistent identity and its
    authored entities; restrict this bootstrap test to Name/Transform/Active.
    Full showcase schema compatibility is a separate rendering/content test.
    """
    source = project / "Content" / "Scenes" / "Startup.arcscene"
    document = json.loads(source.read_text(encoding="utf-8"))
    if document.get("format") != "arc.scene" or not document.get("entities"):
        raise SmokeFailure("generated rendering-sample has no valid startup scene")
    for entity in document["entities"]:
        components = entity.get("components", {})
        entity["components"] = {
            name: component for name, component in components.items()
            if name in {"Name", "Transform"}
        }
    # ARC persistence seals the canonical JSON payload (without the integrity
    # envelope). Re-seal after editing; otherwise arc-cook correctly rejects
    # the generated scene as corrupt.
    document.pop("integrity", None)
    canonical = json.dumps(document, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    document["integrity"] = {
        "algorithm": "sha256",
        "payloadHash": hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
    }
    source.write_text(json.dumps(document, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> None:
    root = args.workspace.resolve()
    artifacts = args.artifacts.resolve()
    artifacts.mkdir(parents=True, exist_ok=True)
    sdk = args.sdk.resolve()
    project_tool = sdk / "bin" / "arc-project.exe"
    cooker = args.cooker.resolve()
    templates = sdk / "share" / "arc" / "templates"
    if not project_tool.is_file() or not cooker.is_file() or not templates.is_dir():
        raise SmokeFailure("installed SDK/project tool, cooker, or installed templates missing")

    project = root / "external-game"
    build = project / "Build" / "ci-shipping"
    cooked = project / "Build" / "Cooked" / "windows-x64-vulkan"
    release = root / "dist" / "GameShipping"
    archive = root / "dist" / "GameShipping.zip"
    root.mkdir(parents=True, exist_ok=True)

    command("create", [str(project_tool), "create", "--name", "ExternalGame",
                      "--destination", str(project), "--template", "rendering-sample",
                      "--templates", str(templates), "--engine", args.engine], root, artifacts)
    descriptor = project / "ExternalGame.arcproject"
    prepare_minimal_startup_scene(project)
    command("configure", ["cmake", "-S", str(project), "-B", str(build), "-G", "Ninja",
                           "-DCMAKE_BUILD_TYPE=Shipping",
                           "-DARC_WORKFLOW_PROFILE=windows-x64-vulkan",
                           f"-DCMAKE_PREFIX_PATH={sdk}"], root, artifacts)
    command("compile", ["cmake", "--build", str(build), "--target", "arc-runtime-output", "--parallel"], root, artifacts)
    command("cook", [str(cooker), "cook", "--project", str(project),
                     "--profile", "windows-x64-vulkan", "--output", str(cooked)], root, artifacts)
    command("cook-verify", [str(cooker), "verify", "--project", str(project),
                            "--profile", "windows-x64-vulkan", "--output", str(cooked)], root, artifacts)

    build_manifest = build / "arc-runtime-Shipping.json"
    command("build-verify", [str(project_tool), "package", "--json", "--verify-build-only",
                             "--project", str(descriptor), "--profile", "windows-x64-vulkan",
                             "--build-manifest", str(build_manifest)], root, artifacts)
    manifest = cooked / "windows-x64-vulkan.arccookmanifest"
    if not manifest.is_file():
        raise SmokeFailure(f"cooker did not publish expected manifest: {manifest}")
    packaged = json.loads(command("content-package", [str(cooker), "package", "--json",
                                 "--project", str(project), "--profile", "windows-x64-vulkan",
                                 "--manifest", str(manifest), "--output", str(cooked / "packaged")], root, artifacts))
    if packaged.get("event") != "package.complete" or not packaged.get("manifest"):
        raise SmokeFailure("content packaging did not return an exact manifest")
    manifest = Path(packaged["manifest"])
    command("content-verify", [str(cooker), "verify", "--project", str(project),
                              "--manifest", str(manifest)], root, artifacts)

    # Only the project binary, its import closure and explicitly selected
    # native backends enter the package; never copy the whole build directory.
    package_args = [str(project_tool), "package", "--project", str(descriptor),
                    "--profile", "windows-x64-vulkan", "--build-manifest", str(build_manifest),
                    "--runtime-config", str(project / "Config" / "Renderer.json"),
                    "--cook-manifest", str(manifest), "--output", str(release),
                    "--zip", str(archive)]
    command("package", package_args, root, artifacts)
    staged = verify_distribution(release)
    if staged["target"]["platform"] != "windows" or staged["role"] != "client":
        raise SmokeFailure("incorrect client/Windows identity in production package")

    # Check both relocated directory and the independently extracted ZIP.
    smoke(release, None, args.timeout, ["--ci-smoke"], artifacts / "directory",
          expected_output="ARC packaged client bootstrap smoke passed")
    smoke(None, archive, args.timeout, ["--ci-smoke"], artifacts / "archive",
          expected_output="ARC packaged client bootstrap smoke passed")
    # D3.3: mutate isolated copies of this actual production distribution and
    # prove inventory and runtime boot fail closed with actionable diagnostics.
    run_negative_acceptance(release, artifacts / "negative", timeout=args.timeout)
    shutil.copyfile(release / "arc-distribution.json", artifacts / "arc-distribution.json")
    (artifacts / "result.json").write_text(json.dumps({
        "status": "passed", "role": "client", "target": staged["target"],
        "relocation": ["directory", "zip"], "package": str(release), "archive": str(archive),
    }, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sdk", type=Path, required=True)
    parser.add_argument("--cooker", type=Path, required=True)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--artifacts", type=Path, required=True)
    parser.add_argument("--engine", default="0.1.0")
    parser.add_argument("--timeout", type=float, default=30.0)
    args = parser.parse_args()
    try:
        run(args)
    except (OSError, ValueError, SmokeFailure, subprocess.SubprocessError) as exc:
        args.artifacts.mkdir(parents=True, exist_ok=True)
        (args.artifacts / "failure.txt").write_text(str(exc) + "\n", encoding="utf-8")
        print(f"D3.2 Shipping failure: {exc}", file=sys.stderr)
        return 1
    except Exception:
        args.artifacts.mkdir(parents=True, exist_ok=True)
        (args.artifacts / "failure.txt").write_text(traceback.format_exc(), encoding="utf-8")
        raise
    print("D3.2 installed-SDK Shipping package and relocation smoke passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
