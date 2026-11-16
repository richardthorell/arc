#!/usr/bin/env python3
"""A6 acceptance: fresh external ARC projects through the public Android build CLI.

Unlike A1-A5's separate native/cook/Gradle fixtures, this executes two entire
real workflows and inspects their published, signed APKs. Requires an already
exported Android host SDK and real host arc-project/arc-cook executables.
No mock cooker, synthetic D0 manifests or repackaged APKs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import struct
import subprocess
import tempfile
from zipfile import ZIP_STORED, ZipFile

PROFILE = "android-arm64-vulkan-debug"
TARGET = "android-arm64-vulkan"
SCENE_GUID = "b80a0000-0000-4000-8000-000000000005"
PROJECTS = {
    "A6Alpha": {"applicationId": "com.arcengine.a6alpha", "versionCode": 601},
    "A6Beta": {"applicationId": "com.arcengine.a6beta", "versionCode": 602},
}
CONTENT = (
    ("tests/fixtures/android-device-smoke/Content/Scenes/Startup.arcscene",
     "Content/Scenes/Startup.arcscene"),
    ("tests/fixtures/android-device-smoke/Content/Scenes/Startup.arcscene.arcmeta",
     "Content/Scenes/Startup.arcscene.arcmeta"),
    ("assets/materials/baseline/unlit_texture.arcmat",
     "Content/materials/unlit_texture.arcmat"),
    ("assets/materials/baseline/unlit_texture.arcmat.arcmeta",
     "Content/materials/unlit_texture.arcmat.arcmeta"),
    ("assets/textures/terrain/aerial_grass_rock/aerial_grass_rock_diff_1k.jpg",
     "Content/textures/terrain/aerial_grass_rock/aerial_grass_rock_diff_1k.jpg"),
    ("assets/textures/terrain/aerial_grass_rock/aerial_grass_rock_diff_1k.jpg.arcmeta",
     "Content/textures/terrain/aerial_grass_rock/aerial_grass_rock_diff_1k.jpg.arcmeta"),
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def read_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"Expected JSON object: {path}")
    return value


def command(argv: list[str], cwd: Path, *, timeout: int = 1200,
            expect_success: bool = True) -> tuple[int, list[dict]]:
    result = subprocess.run(argv, cwd=cwd, capture_output=True, text=True, timeout=timeout,
                            check=False)
    events = []
    for line in result.stdout.splitlines():
        if line.startswith("{"):
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(event, dict):
                events.append(event)
    if expect_success != (result.returncode == 0):
        print("Command:", argv, flush=True)
        print("stdout:\n", result.stdout[-6000:], flush=True)
        print("stderr:\n", result.stderr[-6000:], flush=True)
        # A5 intentionally sends compiler/cooker details into persistent
        # per-run build.log files. Surface the real failure in CI, rather
        # than leaving only an exit code from the public build command.
        records = cwd / "Saved" / "Builds" / "android"
        if records.is_dir():
            for log in sorted(records.glob("*/build.log")):
                data = log.read_text(encoding="utf-8", errors="replace")
                print(f"A5 build log ({log}), last 16000 chars:\n{data[-16000:]}",
                      flush=True)
        raise ValueError(f"Unexpected command status {result.returncode} (success expected: {expect_success})")
    return result.returncode, events


def snapshot(root: Path) -> dict[str, str]:
    # Only source/configuration files. A5 is allowed to write Saved/ and Build/.
    tracked = {}
    for name in ("Content", "Source", "Config", "cmake"):
        folder = root / name
        if folder.is_dir():
            for file in folder.rglob("*"):
                if file.is_file():
                    tracked[file.relative_to(root).as_posix()] = digest(file.read_bytes())
    for file in root.glob("*.arcproject"):
        tracked[file.name] = digest(file.read_bytes())
    for file in (root / "CMakeLists.txt",):
        if file.is_file():
            tracked[file.name] = digest(file.read_bytes())
    return tracked


def create_project(repo: Path, project_tool: Path, root: Path, name: str) -> Path:
    project = root / name
    command([str(project_tool), "create", "--name", name,
             "--destination", str(project), "--template", "blank-3d",
             "--templates", str(repo / "templates"), "--engine", "0.1.0",
             "--json"], cwd=root)
    descriptor_path = project / f"{name}.arcproject"
    require(descriptor_path.is_file(), "arc-project create did not produce a project descriptor")
    doc = read_json(descriptor_path)
    require(doc["guid"] and doc["engineVersion"] == "0.1.0",
            "Invalid generated external project identity")

    for src, dst in CONTENT:
        destination = project / dst
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(repo / src, destination)
    # The template itself supplies Content/models/UAL2_Standard.glb + metadata.
    require((project / "Content/models/UAL2_Standard.glb").is_file(),
            "Independent template is missing the startup-scene model")

    scene_file = project / "Content/Scenes/Startup.arcscene"
    scene_document = read_json(scene_file)
    scene_document["scene"]["name"] = f"{name} Android acceptance"
    # Distinguish the cooked chunks, not just the APK native module.
    scene_document["entities"][0]["components"]["Name"]["value"] = name + " Camera"
    scene_file.write_text(json.dumps(scene_document, indent=2) + "\n", encoding="utf-8")
    startup_scene = {
        "guid": SCENE_GUID,
        "expectedType": "scene",
        "pathHint": "Content/Scenes/Startup.arcscene",
    }
    doc["startupScenes"] = [startup_scene]
    doc["defaultScene"] = startup_scene
    doc["package"]["applicationName"] = name
    settings = doc["platformSettings"]["android"]
    settings.update({
        "applicationId": PROJECTS[name]["applicationId"],
        "versionName": f"6.0.{PROJECTS[name]['versionCode']}",
        "versionCode": PROJECTS[name]["versionCode"],
        "minimumSdk": 26,
        "targetSdk": 35,
    })
    descriptor_path.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")
    command([str(project_tool), "validate", "--project", str(descriptor_path), "--json"],
            cwd=project)
    return descriptor_path


def build(project_tool: Path, cooker: Path, sdk: Path, descriptor: Path,
          output: Path, run_id: str, *, expect_success: bool) -> dict:
    argv = [str(project_tool), "build", "--platform", "android",
            "--project", str(descriptor), "--sdk-root", str(sdk),
            "--cooker", str(cooker), "--profile", PROFILE, "--target", TARGET,
            "--run-id", run_id, "--output", str(output), "--json"]
    code, events = command(argv, cwd=descriptor.parent, timeout=1500,
                           expect_success=expect_success)
    state_file = descriptor.parent / "Saved/Builds/android" / run_id / "status.json"
    require(state_file.is_file(), f"No persistent A5 status for {run_id}")
    status = read_json(state_file)
    require(status.get("runId") == run_id and status.get("format") == "arc.workflow-run" and
            status.get("project") == str(descriptor), "A5 persisted invalid identity")
    expected_status = "succeeded" if expect_success else "failed"
    require(status["status"] == expected_status, f"Expected {expected_status} build: {status}")
    require([step["name"] for step in status["stages"]] ==
            ["validate", "configure", "compile", "cook", "package"], "A5 stage sequence changed")
    require((descriptor.parent / "Saved/Builds/android" / run_id / "build.log").is_file(),
            "No A5 build log")
    require(any(event.get("event") == ("workflow.complete" if expect_success else "workflow.failure")
                for event in events), "A5 emitted no terminal event")
    require((code == 0) == expect_success, "A5 process code disagrees with state")
    print(f"A6 {descriptor.parent.name} {run_id}: {status['status']}, stage={status['stage']}",
          flush=True)
    return status


def verify_result(project: Path, sdk: Path, cooker: Path, output: Path, status: dict) -> dict:
    descriptor = read_json(next(project.glob("*.arcproject")))
    package = project.name
    info = PROJECTS[package]
    apk = output / (info["applicationId"] + "-debug.apk")
    report_file = output / "arc-apk-integrity.json"
    require(apk.is_file() and report_file.is_file(), f"Missing A4 APK or sidecar: {output}")
    require(set(item.name for item in output.iterdir()) ==
            {apk.name, report_file.name}, "Published A4 output contains untracked files")
    report = read_json(report_file)
    require(report.get("format") == "arc.android-apk-finalization" and
            report.get("formatVersion") == 1 and
            report.get("signed") is True and report.get("signatureVerified") is True,
            "A4 finalizer did not certify a signed APK")
    require(report["projectGuid"] == descriptor["guid"] ==
            status["projectGuid"] and report["applicationId"] == info["applicationId"] and
            report["versionCode"] == info["versionCode"] and
            report["versionName"] == f"6.0.{info['versionCode']}",
            "APK identity differs from the authored project")
    require(report["configuration"] == "Debug" and report["abi"] == "arm64-v8a",
            "A6 only accepts Debug ARM64 APKs")
    data = apk.read_bytes()
    require(report["apk"]["path"] == apk.name and
            report["apk"]["size"] == len(data) and
            report["apk"]["sha256"] == digest(data), "APK sidecar hash differs from published APK")
    require(status.get("apk") == str(apk) and
            status.get("integrity") == str(report_file), "A5 status does not reference published A4 output")

    with ZipFile(apk) as archive:
        names = archive.namelist()
        require(len(names) == len(set(names)), "APK has duplicate ZIP entries")
        d0 = json.loads(archive.read("assets/arc-distribution.json"))
        require(d0.get("format") == "arc.distribution" and
                d0.get("formatVersion") == 1 and
                d0["projectGuid"] == descriptor["guid"] and
                d0["target"]["configuration"] == "Debug" and
                d0["target"]["id"] == TARGET and
                d0["target"]["cookProfile"] == PROFILE and
                d0["startupScenes"] == [SCENE_GUID],
                "Incorrect cooked D0 project, platform or startup scene")
        require(report["distributionSha256"] == digest(archive.read("assets/arc-distribution.json")),
                "APK D0 manifest changed after A4 signing")
        files = d0["files"]
        require(len(files) >= 3, "A6 cooked distribution lacks startup-scene dependencies")
        expected_assets = {"assets/arc-distribution.json"}
        for entry in files:
            name = "assets/" + entry["path"]
            expected_assets.add(name)
            content = archive.read(name)
            require(digest(content) == entry["sha256"] and len(content) == entry["size"],
                    f"Corrupt A6 APK content: {name}")
            if name.endswith((".arcpak", ".arccookmanifest")):
                require(archive.getinfo(name).compress_type == ZIP_STORED,
                        f"Cook package is compressed: {name}")
        require({name for name in names if name.startswith("assets/") and not name.endswith("/")} ==
                expected_assets, "APK assets differ from the D0 inventory")
        require(any(name.endswith(".arcpak") for name in expected_assets),
                "No cooked ARCPAK shipped in A6 APK")
        require(len(d0["content"]["cookManifests"]) == 1, "Expected one Android ASTC cook manifest")

        native = {item["path"]: item for item in report["nativeLibraries"]}
        runtime_path = "lib/arm64-v8a/libarc_project_runtime.so"
        host_path = "lib/arm64-v8a/libarc_android_host.so"
        require(runtime_path in native and host_path in native, "APK lacks A1 host or A2 project module")
        for name, record in native.items():
            binary = archive.read(name)
            require(binary.startswith(b"\x7fELF") and binary[4] == 2 and binary[5] == 1 and
                    struct.unpack_from("<H", binary, 18)[0] == 183, f"Not ARM64 ELF: {name}")
            require(len(binary) == record["size"] and digest(binary) == record["sha256"],
                    f"A4 native inventory mismatch: {name}")
        runtime = archive.read(runtime_path)
        require(descriptor["guid"].encode() in runtime and
                (package + ".runtime").encode() in runtime, "Wrong project C++ module packaged")
        host_record = read_json(sdk / "share/arc/android/host/host.json")
        host_hash = {item["name"]: item["sha256"]
                     for item in host_record["configurations"]["debug"]}
        require(digest(archive.read(host_path)) == host_hash["libarc_android_host.so"],
                "APK native host differs from the installed SDK")

        with tempfile.TemporaryDirectory(prefix="arc-a6-cook-verify-") as directory:
            root = Path(directory)
            for entry in files:
                if entry["path"].endswith((".arccookmanifest", ".arcpak")):
                    dest = root / entry["path"]
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    dest.write_bytes(archive.read("assets/" + entry["path"]))
            cook_manifest = root / d0["content"]["cookManifests"][0]
            require(cook_manifest.is_file(), "No cooked manifest extracted from finished APK")
            command([str(cooker), "verify", "--project", str(project),
                     "--manifest", str(cook_manifest), "--json"], cwd=project)
    print(f"A6 accepted signed, cooked, module-specific APK {apk.name}", flush=True)
    return {
        "projectGuid": descriptor["guid"],
        "projectModuleSha256": native[runtime_path]["sha256"],
        "apkSha256": report["apk"]["sha256"],
        "d0Sha256": report["distributionSha256"],
    }


def run(repo: Path, sdk: Path, project_tool: Path, cooker: Path, output: Path) -> None:
    repo = repo.resolve(strict=True)
    sdk = sdk.resolve(strict=True)
    project_tool = project_tool.resolve(strict=True)
    cooker = cooker.resolve(strict=True)
    require(not output.exists(), f"A6 acceptance output must be new: {output}")
    output.mkdir(parents=True)
    projects = output / "Projects"
    packages = output / "Packages"
    projects.mkdir()
    packages.mkdir()
    results = []
    for name in PROJECTS:
        desc = create_project(repo, project_tool, projects, name)
        before = snapshot(desc.parent)
        packaged = packages / name
        status = build(project_tool, cooker, sdk, desc, packaged,
                       f"a6-{name.lower()}", expect_success=True)
        require(snapshot(desc.parent) == before, "A5 unexpectedly modified authored game content")
        require(all(step["status"] == "succeeded" for step in status["stages"]),
                "Not every A5 stage actually completed")
        results.append(verify_result(desc.parent, sdk, cooker, packaged, status))

    require(results[0]["projectGuid"] != results[1]["projectGuid"] and
            results[0]["projectModuleSha256"] != results[1]["projectModuleSha256"] and
            results[0]["apkSha256"] != results[1]["apkSha256"] and
            results[0]["d0Sha256"] != results[1]["d0Sha256"],
            "Two fresh independent projects produced indistinguishable Android artifacts")

    # An actual late cook failure after a successful native compile must not
    # publish a partial or stale APK. Copy external authored files only, then
    # break one required scene dependency (without invalidating the descriptor).
    source = projects / "A6Alpha"
    broken = projects / "A6Broken"
    shutil.copytree(source, broken, ignore=shutil.ignore_patterns("Build", "Saved", "Intermediate"))
    broken_descriptor = broken / "A6Alpha.arcproject"
    scene_file = broken / "Content/Scenes/Startup.arcscene"
    scene = read_json(scene_file)
    scene["dependencies"][0]["guid"] = "c1234567-1111-4111-8111-123456789abc"
    scene_file.write_text(json.dumps(scene, indent=2) + "\n", encoding="utf-8")
    failed_output = packages / "Broken"
    failed = build(project_tool, cooker, sdk, broken_descriptor, failed_output,
                   "a6-broken-cook", expect_success=False)
    require(failed["stage"] == "cook" and
            [part["status"] for part in failed["stages"][:3]] ==
            ["succeeded", "succeeded", "succeeded"] and
            failed["stages"][3]["status"] == "failed" and
            failed["stages"][4]["status"] == "pending",
            "Missing required content dependency did not fail at the cooker stage")
    require(not failed_output.exists() and not Path(str(failed_output) + ".pending").exists(),
            "A6 failed cook leaked a partially published Android APK")
    require(not list((broken / "Build").rglob(".a5-*")),
            "A6 failed workflow left temporary private workspaces")
    print(json.dumps({"succeeded": True, "projects": results,
                      "lateCookFailurePublishedApk": False}, sort_keys=True), flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--sdk", type=Path, required=True)
    parser.add_argument("--project-tool", type=Path, required=True)
    parser.add_argument("--cooker", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        run(args.repo, args.sdk, args.project_tool, args.cooker, args.output.absolute())
    except (OSError, ValueError, subprocess.SubprocessError) as exc:
        print(f"Android A6 acceptance FAILED: {exc}", flush=True)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
