#!/usr/bin/env python3
"""ARC Android A4 APK finalizer.

Consumes an A3 D0 cooked-content stage, an installed A1/A2 Android host SDK,
project native code, and a validated ARC project-settings export. Invokes the
SDK's standalone Gradle launcher in a private workspace, validates the signed
Debug APK, and atomically publishes an APK and native/content integrity report.

Release signing, AAB and Play distribution are deliberately separate work.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import struct
import subprocess
import sys
import tempfile
from zipfile import BadZipFile, ZIP_STORED, ZipFile

ABI = "arm64-v8a"
HOST_NAME = "libarc_android_host.so"
PROJECT_NAME = "libarc_project_runtime.so"
MAX_MANIFEST = 16 * 1024 * 1024
MAX_LIBRARY = 512 * 1024 * 1024
SAFE_LIBRARY = re.compile(r"lib[A-Za-z0-9_.+-]+\.so\Z")
SAFE_SHA = re.compile(r"[0-9a-f]{64}\Z")
PACKAGE_LINE = re.compile(r"package: name='([^']+)' versionCode='([^']+)' versionName='([^']+)'")
SDK_LINE = re.compile(r"^sdkVersion:'(\d+)'$", re.MULTILINE)
TARGET_LINE = re.compile(r"^targetSdkVersion:'(\d+)'$", re.MULTILINE)
LABEL_LINE = re.compile(r"^application-label:'([^']*)'$", re.MULTILINE)
LAUNCH_LINE = re.compile(r"^launchable-activity: name='([^']+)'", re.MULTILINE)


class FinalizationError(ValueError):
    """An invalid project, staged distribution, APK or build tool."""


def expect(condition: bool, message: str) -> None:
    if not condition:
        raise FinalizationError(message)


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def read_json(path: Path, maximum: int = MAX_MANIFEST) -> dict:
    expect(path.is_file() and not path.is_symlink(), f"Missing or unsafe JSON file: {path}")
    expect(0 < path.stat().st_size <= maximum, f"Invalid JSON file size: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise FinalizationError(f"Invalid JSON in {path.name}: {exc}") from exc
    expect(isinstance(value, dict), f"Expected object in {path.name}")
    return value


def exact_directory(path: Path, label: str) -> Path:
    expect(path.is_absolute() and path.is_dir() and not path.is_symlink(),
           f"{label} must be an existing absolute directory")
    return path.resolve(strict=True)


def safe_relative(value: object) -> str:
    expect(isinstance(value, str) and value and "\\" not in value and ":" not in value,
           "Unsafe distribution-relative filename")
    path = PurePosixPath(value)
    expect(not path.is_absolute() and all(part not in ("", ".", "..") for part in value.split("/")) and
           all(ord(char) >= 32 and ord(char) != 127 for char in value),
           f"Unsafe distribution-relative path: {value}")
    return value


def project_descriptor(root: Path) -> dict:
    descriptors = list(root.glob("*.arcproject"))
    expect(len(descriptors) == 1, "Android project root must contain exactly one .arcproject")
    document = read_json(descriptors[0])
    expect(document.get("format") == "arc-project" and isinstance(document.get("guid"), str),
           "Invalid ARC project descriptor")
    return document


def project_settings(project_root: Path, explicit: Path | None, tool: str | None) -> dict:
    if explicit is not None:
        settings = read_json(explicit, 64 * 1024)
    else:
        expect(bool(tool), "A4 requires either --settings or the active --project-tool")
        result = subprocess.run(
            [tool, "settings", "export-android", "--project", str(project_root), "--json"],
            capture_output=True, text=True, check=False,
        )
        expect(result.returncode == 0,
               f"ARC project settings export failed: {result.stderr.strip() or result.stdout.strip()}")
        try:
            settings = json.loads(result.stdout)
        except json.JSONDecodeError as exc:
            raise FinalizationError("ARC project settings tool returned invalid JSON") from exc
    expect(isinstance(settings, dict) and settings.get("format") == "arc.android-project-settings" and
           settings.get("version") == 1, "Unsupported ARC Android project settings export")
    application = settings.get("applicationId")
    expect(isinstance(application, str) and re.fullmatch(
        r"[A-Za-z][A-Za-z0-9_]*(\.[A-Za-z][A-Za-z0-9_]*)+", application),
        "Invalid Android application ID")
    expect(isinstance(settings.get("displayName"), str) and bool(settings["displayName"]),
           "Android display name is required")
    expect(isinstance(settings.get("versionName"), str) and bool(settings["versionName"]),
           "Android versionName is required")
    for field in ("versionCode", "minimumSdk", "targetSdk"):
        expect(type(settings.get(field)) is int, f"Android {field} must be an integer")
    expect(1 <= settings["versionCode"] <= 2100000000 and
           26 <= settings["minimumSdk"] <= settings["targetSdk"] <= 35,
           "Unsupported Android version or SDK settings")
    return settings


def validate_inputs(project_root: Path, sdk_root: Path, assets_root: Path,
                    settings: dict) -> tuple[dict, dict, dict, dict]:
    descriptor = project_descriptor(project_root)
    sdk = read_json(sdk_root / "arc-installation.json")
    expect(sdk.get("format") == "arc-installation" and isinstance(sdk.get("engineVersion"), str),
           "Selected ARC SDK is missing installation identity")
    expect((sdk_root / "lib/cmake/ARC/ARCConfig.cmake").is_file(),
           "Selected ARC SDK lacks its CMake package")
    host = read_json(sdk_root / "share/arc/android/host/host.json")
    expect(host.get("format") == "arc.android-host-sdk" and host.get("version") == 1 and
           host.get("abi") == ABI and host.get("engineVersion") == sdk["engineVersion"],
           "Installed Android host is incompatible with selected ARC SDK")

    document = read_json(assets_root / "arc-distribution.json")
    target = document.get("target")
    expect(document.get("format") == "arc.distribution" and document.get("formatVersion") == 1 and
           isinstance(target, dict), "A3 content stage must contain a versioned D0 distribution")
    expect(target.get("platform") == "android" and target.get("architecture") == "arm64" and
           target.get("renderer") == "vulkan" and target.get("id") == "android-arm64-vulkan" and
           target.get("configuration") == "Debug", "A4 currently finalizes Android Debug ARM64 Vulkan only")
    expect(document.get("projectGuid") == descriptor.get("guid") == settings.get("projectGuid"),
           "Project GUID mismatch between A3 assets, project and exported settings")
    expect(document.get("engineVersion") == sdk["engineVersion"] ==
           document.get("sdkVersion") == descriptor.get("engineVersion"),
           "ARC engine/SDK version mismatch across project and cooked content")
    expect(document.get("role") == "client" and document.get("finalizer") == "android-gradle",
           "D0 distribution role/finalizer mismatch")
    expect(document.get("entrypoint") == {
        "kind": "android-activity", "value": "android.app.NativeActivity",
    }, "A4 requires an android.app.NativeActivity entrypoint")
    expect(isinstance(document.get("startupScenes"), list) and bool(document["startupScenes"]),
           "Android content stage must declare startup scenes")
    expect(isinstance(document.get("content"), dict), "Android cooked-content metadata missing")
    cooks = document["content"].get("cookManifests")
    expect(isinstance(cooks, list) and bool(cooks), "Cook manifests are required in A3 assets")
    expect(isinstance(document.get("files"), list) and bool(document["files"]),
           "A3 distribution has no file inventory")

    inventory: dict[str, dict] = {}
    for entry in document["files"]:
        expect(isinstance(entry, dict), "Invalid staged file entry")
        relative = safe_relative(entry.get("path"))
        expect(relative.startswith(("Content/", "Config/")), "A3 files must be Content or Config")
        expect(entry.get("kind") == ("content" if relative.startswith("Content/") else "configuration"),
               f"Incorrect staged file kind: {relative}")
        expect(relative.lower() not in inventory, "Duplicate staged file path")
        actual = assets_root / relative
        expect(actual.is_file() and not actual.is_symlink() and actual.resolve().is_relative_to(assets_root),
               f"Missing or escaping staged file: {relative}")
        data = actual.read_bytes()
        expect(type(entry.get("size")) is int and entry["size"] == len(data) and
               isinstance(entry.get("sha256"), str) and
               SAFE_SHA.fullmatch(entry["sha256"]) and sha(data) == entry["sha256"],
               f"Staged cooked-content hash/size mismatch: {relative}")
        inventory[relative.lower()] = entry

    config = safe_relative(document.get("runtimeConfig"))
    expect(config.lower() in inventory and inventory[config.lower()]["kind"] == "configuration",
           "D0 runtime configuration missing from inventory")
    for cook in cooks:
        relative = safe_relative(cook)
        expect(relative.endswith(".arccookmanifest") and relative.lower() in inventory,
               f"Missing D0 cook manifest: {relative}")

    seen: set[str] = set()
    for item in assets_root.rglob("*"):
        expect(not item.is_symlink(), "A3 staged assets must not contain symbolic links")
        if item.is_file():
            relative = item.relative_to(assets_root).as_posix()
            if relative != "arc-distribution.json":
                seen.add(relative.lower())
        else:
            expect(item.is_dir(), "Non-regular A3 staged asset")
    expect(seen == set(inventory), "A3 stage has missing or untracked cooked assets")

    libraries = host.get("configurations", {}).get("debug")
    expect(isinstance(libraries, list) and bool(libraries), "Installed SDK has no Debug ARM64 host")
    expected_host: dict[str, str] = {}
    for entry in libraries:
        expect(isinstance(entry, dict) and isinstance(entry.get("name"), str) and
               bool(SAFE_LIBRARY.fullmatch(entry["name"])) and
               isinstance(entry.get("sha256"), str) and SAFE_SHA.fullmatch(entry["sha256"]),
               "Installed Android SDK host manifest is invalid")
        expect(entry["name"] not in expected_host, "Duplicate Android SDK library")
        expected_host[entry["name"]] = entry["sha256"]
        source = sdk_root / "share/arc/android/host/debug/jniLibs" / ABI / entry["name"]
        expect(source.is_file() and sha(source.read_bytes()) == entry["sha256"],
               f"Installed ARC SDK native host has changed: {entry['name']}")
    expect(HOST_NAME in expected_host, "Android host SDK is missing ARC native host")
    return descriptor, document, host, expected_host


def android_tools(sdk_root: Path) -> tuple[Path, Path]:
    android_home = os.environ.get("ANDROID_HOME") or os.environ.get("ANDROID_SDK_ROOT")
    expect(bool(android_home), "ANDROID_HOME/ANDROID_SDK_ROOT must point to the Android SDK")
    build_tools = Path(android_home) / "build-tools"
    expect(build_tools.is_dir(), "Android SDK build-tools are not installed")
    versions = sorted((path for path in build_tools.iterdir() if path.is_dir()),
                      key=lambda x: tuple(int(n) for n in re.findall(r"\d+", x.name)), reverse=True)
    for directory in versions:
        aapt = directory / ("aapt.exe" if os.name == "nt" else "aapt")
        signer = directory / ("apksigner.bat" if os.name == "nt" else "apksigner")
        if aapt.is_file() and signer.is_file():
            return aapt, signer
    raise FinalizationError("Android build tools must provide both aapt and apksigner")


def verify_binary(data: bytes, label: str) -> None:
    expect(len(data) >= 20 and len(data) <= MAX_LIBRARY and data[:4] == b"\x7fELF" and
           data[4] == 2 and data[5] == 1 and struct.unpack_from("<H", data, 18)[0] == 183,
           f"APK native library is not an ARM64 ELF: {label}")


def verify_apk(apk: Path, assets_root: Path, document: dict, settings: dict,
               expected_host: dict[str, str], aapt: Path, signer: Path) -> dict:
    expect(apk.is_file(), "Gradle did not produce the expected Debug APK")
    with ZipFile(apk) as zipped:
        names = zipped.namelist()
        expect(len(names) == len(set(names)), "APK contains duplicate ZIP members")
        native: dict[str, dict] = {}
        for member in zipped.infolist():
            path = PurePosixPath(member.filename)
            expect(bool(member.filename) and not member.filename.startswith("/") and
                   "\\" not in member.filename and ".." not in path.parts and
                   not member.filename.startswith("./"), f"Unsafe APK ZIP member: {member.filename}")
            if member.is_dir():
                continue
            if path.parts and path.parts[0] == "lib":
                expect(len(path.parts) == 3 and path.parts[1] == ABI and
                       bool(SAFE_LIBRARY.fullmatch(path.name)), "APK contains an unexpected native ABI or library")
                data = zipped.read(member)
                verify_binary(data, member.filename)
                native[path.name] = {"path": member.filename, "size": len(data), "sha256": sha(data)}
        expect(HOST_NAME in native and PROJECT_NAME in native,
               "APK is missing ARC Android host or project runtime module")
        for filename, digest in expected_host.items():
            expect(filename in native and native[filename]["sha256"] == digest,
                   f"APK native host differs from installed ARC SDK: {filename}")
        project_library = zipped.read(f"lib/{ABI}/{PROJECT_NAME}")
        expect(document["projectGuid"].encode("ascii") in project_library,
               "Compiled native project module GUID differs from D0 content identity")
        expected_files = {"assets/arc-distribution.json"}
        for entry in document["files"]:
            member = "assets/" + entry["path"]
            expected_files.add(member)
            expect(member in names, f"APK lacks cooked asset: {member}")
            info = zipped.getinfo(member)
            expect(info.file_size == entry["size"] and sha(zipped.read(info)) == entry["sha256"],
                   f"APK cooked asset differs from staged D0 inventory: {member}")
            if member.endswith((".arcpak", ".arccookmanifest")):
                expect(info.compress_type == ZIP_STORED,
                       f"APK cooked content must be uncompressed for AAssetManager: {member}")
        expect(zipped.read("assets/arc-distribution.json") ==
               (assets_root / "arc-distribution.json").read_bytes(),
               "Embedded D0 distribution manifest differs from verified stage")
        present_assets = {name for name in names if name.startswith("assets/") and not name.endswith("/")}
        expect(present_assets == expected_files, "APK contains unexpected or missing assets")
        expect("AndroidManifest.xml" in names, "APK has no Android application manifest")

    result = subprocess.run([str(aapt), "dump", "badging", str(apk)],
                            capture_output=True, text=True, check=False)
    expect(result.returncode == 0, f"Android aapt manifest inspection failed: {result.stderr.strip()}")
    package = PACKAGE_LINE.search(result.stdout)
    expect(bool(package) and package.group(1) == settings["applicationId"] and
           package.group(2) == str(settings["versionCode"]) and
           package.group(3) == settings["versionName"],
           "APK application ID/version does not match exported ARC settings")
    minimum, target = SDK_LINE.search(result.stdout), TARGET_LINE.search(result.stdout)
    expect(bool(minimum) and bool(target) and int(minimum.group(1)) == settings["minimumSdk"] and
           int(target.group(1)) == settings["targetSdk"], "APK SDK limits differ from project settings")
    label, launch = LABEL_LINE.search(result.stdout), LAUNCH_LINE.search(result.stdout)
    expect(bool(label) and label.group(1) == settings["displayName"] and
           bool(launch) and launch.group(1) == "android.app.NativeActivity",
           "APK label or launch activity differs from project settings")

    signed = subprocess.run([str(signer), "verify", "--verbose", str(apk)],
                            capture_output=True, text=True, check=False)
    expect(signed.returncode == 0,
           f"Android APK must have a valid Debug signature: {signed.stderr.strip() or signed.stdout.strip()}")
    return {"nativeLibraries": [native[name] for name in sorted(native)],
            "nativeHost": HOST_NAME, "projectModule": PROJECT_NAME,
            "signed": True, "signatureVerified": True}


def finalize(project_root: Path, sdk_root: Path, assets_root: Path, output: Path,
             tool: str | None, settings_path: Path | None, gradle: str) -> dict:
    project_root = exact_directory(project_root, "project root")
    sdk_root = exact_directory(sdk_root, "ARC SDK root")
    assets_root = exact_directory(assets_root, "A3 distribution asset root")
    expect(output.is_absolute(), "A4 output must be an absolute package directory")
    output = output.absolute()
    expect(not output.exists() and not Path(str(output) + ".pending").exists(),
           "Android APK output or reserved pending directory already exists")
    for root in (sdk_root, assets_root):
        expect(not output.is_relative_to(root) and not root.is_relative_to(output),
               "APK output must not overlap SDK or staged content roots")
    expect(output != project_root and not project_root.is_relative_to(output),
           "APK output must not replace the project root")

    settings = project_settings(project_root, settings_path, tool)
    descriptor, document, _, expected_host = validate_inputs(project_root, sdk_root, assets_root, settings)
    aapt, signer = android_tools(sdk_root)
    launcher = sdk_root / "share/arc/android/gradle"
    expect((launcher / "app/build.gradle.kts").is_file() and
           (launcher / "app/src/main/cpp/project_module/CMakeLists.txt").is_file(),
           "Installed ARC SDK has no A2 Android Gradle project launcher")
    expect(bool(gradle), "Gradle executable is required")
    output.parent.mkdir(parents=True, exist_ok=True)

    # A4 owns an isolated Gradle workspace. No stale project build or SDK files
    # can become implicit inputs; failed builds never publish an output APK.
    with tempfile.TemporaryDirectory(prefix=".arc-apk-", dir=output.parent) as temporary:
        scratch = Path(temporary)
        gradle_root = scratch / "gradle"
        shutil.copytree(launcher, gradle_root, ignore=shutil.ignore_patterns(".gradle", "build", ".cxx"))
        settings_file = scratch / "android-project-settings.json"
        settings_file.write_text(json.dumps(settings, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        command = [gradle, "--no-daemon", "--project-dir", str(gradle_root),
                   f"-ParcSdkRoot={sdk_root}", f"-ParcProjectRoot={project_root}",
                   f"-ParcDistributionAssets={assets_root}", f"-ParcProjectSettings={settings_file}",
                   ":app:assembleDebug", "--stacktrace"]
        result = subprocess.run(command, cwd=gradle_root, stdout=sys.stderr,
                                stderr=sys.stderr, check=False)
        expect(result.returncode == 0, f"Android Gradle Debug build failed (exit {result.returncode})")
        apk = gradle_root / "app/build/outputs/apk/debug/app-debug.apk"
        inspected = verify_apk(apk, assets_root, document, settings, expected_host, aapt, signer)

        pending = Path(str(output) + ".pending")
        pending.mkdir()
        try:
            apk_name = settings["applicationId"] + "-debug.apk"
            staged_apk = pending / apk_name
            shutil.copyfile(apk, staged_apk)
            digest = sha(staged_apk.read_bytes())
            report = {
                "format": "arc.android-apk-finalization",
                "formatVersion": 1,
                "projectGuid": descriptor["guid"],
                "engineVersion": document["engineVersion"],
                "applicationId": settings["applicationId"],
                "versionCode": settings["versionCode"],
                "versionName": settings["versionName"],
                "configuration": "Debug",
                "abi": ABI,
                "entrypoint": "android.app.NativeActivity",
                "distributionSha256": sha((assets_root / "arc-distribution.json").read_bytes()),
                "apk": {"path": apk_name, "size": staged_apk.stat().st_size, "sha256": digest},
                "assets": [{"path": entry["path"], "size": entry["size"], "sha256": entry["sha256"]}
                           for entry in document["files"]],
                **inspected,
            }
            report_file = pending / "arc-apk-integrity.json"
            report_file.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            expect(not output.exists(), "Android APK output appeared during finalization")
            os.replace(pending, output)
        finally:
            if pending.exists():
                shutil.rmtree(pending)
    return {"succeeded": True, "apk": str(output / apk_name),
            "manifest": str(output / "arc-apk-integrity.json"),
            "sha256": digest, "applicationId": settings["applicationId"],
            "nativeLibraries": len(inspected["nativeLibraries"])}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", required=True, type=Path)
    parser.add_argument("--sdk-root", required=True, type=Path)
    parser.add_argument("--assets", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--project-tool", help="ARC CLI for validated Android settings export")
    parser.add_argument("--settings", type=Path, help="Existing validated ARC settings export")
    parser.add_argument("--gradle", default="gradle")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    try:
        result = finalize(args.project_root, args.sdk_root, args.assets, args.output,
                          args.project_tool, args.settings, args.gradle)
    except (FinalizationError, ValueError, OSError, BadZipFile, subprocess.SubprocessError) as exc:
        if args.json:
            print(json.dumps({"succeeded": False, "error": str(exc)}))
        else:
            print(f"ARC Android APK finalization failed: {exc}", file=sys.stderr)
        return 1
    if args.json:
        print(json.dumps(result))
    else:
        print("Packaged signed ARC Android Debug APK:", result["apk"])
        print("Integrity manifest:", result["manifest"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
