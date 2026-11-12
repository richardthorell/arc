"""Exercise the CMake producer and native consumer without requiring a GPU/SDK."""
import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile

tool, templates, version, cmake, module = sys.argv[1:]


def run(*args):
    result = subprocess.run(args, capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stdout + result.stderr
    return result


with tempfile.TemporaryDirectory(prefix="arc runtime metadata ") as temporary:
    root = Path(temporary)
    run(tool, "create", "--name", "Game", "--destination", str(root),
        "--template", "blank-3d", "--templates", templates, "--engine", version)
    descriptor = root / "Game.arcproject"
    project = json.loads(descriptor.read_text(encoding="utf-8"))
    project["cookProfiles"][0]["configuration"] = "Shipping"
    descriptor.write_text(json.dumps(project), encoding="utf-8")
    executable = root / "Game.exe"
    library = root / "render.dll"
    executable.write_bytes(b"deterministic executable fixture")
    library.write_bytes(b"deterministic library fixture")
    source = root / "fixture"
    source.mkdir()
    (source / "CMakeLists.txt").write_text(f'''
cmake_minimum_required(VERSION 3.20)
project(RuntimeMetadata NONE)
set(CMAKE_CONFIGURATION_TYPES Debug Shipping CACHE STRING "" FORCE)
set(CMAKE_SYSTEM_NAME Windows)
set(CMAKE_SYSTEM_PROCESSOR x86_64)
set(CMAKE_SIZEOF_VOID_P 8)
include("{Path(module).as_posix()}")
add_executable(Game IMPORTED GLOBAL)
set_target_properties(Game PROPERTIES IMPORTED_LOCATION "{executable.as_posix()}")
add_library(Render SHARED IMPORTED GLOBAL)
set_target_properties(Render PROPERTIES IMPORTED_LOCATION "{library.as_posix()}")
arc_register_runtime_output(TARGET Game PROJECT_GUID "{project['guid']}"
 ENGINE_VERSION "{version}" PROFILE windows-x64-vulkan RENDERER vulkan ROLE client LIBRARIES Render)
''', encoding="utf-8")
    build = root / "build"
    run(cmake, "-S", str(source), "-B", str(build), "-DCMAKE_BUILD_TYPE=Shipping")
    manifest = build / "arc-runtime-Shipping.json"
    assert not manifest.exists(), "Configure must not advertise completed runtime output"
    build_args = (cmake, "--build", str(build), "--config", "Shipping", "--target", "arc-runtime-output")
    run(*build_args)
    original = manifest.read_bytes()
    data = json.loads(original)
    assert data["targetId"] == "windows-x64-vulkan"
    assert data["executable"]["path"] == executable.as_posix()
    assert len(data["libraries"]) == 1

    def verify(expected=True):
        result = subprocess.run([tool, "package", "--json", "--project", str(descriptor),
                                 "--profile", "windows-x64-vulkan", "--build-manifest", str(manifest),
                                 "--verify-build-only"], capture_output=True, text=True)
        payload = json.loads(result.stdout)
        assert payload["succeeded"] == expected, payload
        assert (result.returncode == 0) == expected, result

    verify()
    mixed = subprocess.run([tool, "package", "--json", "--project", str(descriptor),
                            "--profile", "windows-x64-vulkan", "--build-manifest", str(manifest),
                            "--executable", str(executable), "--verify-build-only"],
                           capture_output=True, text=True)
    assert mixed.returncode != 0
    assert json.loads(mixed.stdout)["field"] == "arguments"
    # Exercise the packaging path (not only standalone verification): validated
    # build selection must get as far as cook validation without manual binaries.
    package = subprocess.run([tool, "package", "--json", "--project", str(descriptor),
                              "--profile", "windows-x64-vulkan", "--build-manifest", str(manifest),
                              "--runtime-config", "missing.json", "--cook-manifest", "missing.arccookmanifest"],
                             capture_output=True, text=True)
    assert package.returncode != 0
    assert json.loads(package.stdout)["field"] == "cookManifest", package.stdout
    run(*build_args)
    assert manifest.read_bytes() == original, "Repeated metadata emission must be deterministic"
    for field, value in (("formatVersion", 9), ("projectGuid", "different"),
                         ("engineVersion", "9.9.9"), ("profile", "other"),
                         ("targetId", "android-arm64-vulkan"), ("role", "server"),
                         ("configuration", "Debug")):
        changed = copy.deepcopy(data)
        changed[field] = value
        manifest.write_text(json.dumps(changed), encoding="utf-8")
        verify(False)
    changed = copy.deepcopy(data)
    changed["libraries"].append(changed["libraries"][0])
    manifest.write_text(json.dumps(changed), encoding="utf-8")
    verify(False)
    for record in ({"path": "relative.exe", "size": 1, "sha256": "bad"},
                   {**data["executable"], "size": -1},
                   {**data["executable"], "sha256": "0" * 64}):
        changed = copy.deepcopy(data)
        changed["executable"] = record
        manifest.write_text(json.dumps(changed), encoding="utf-8")
        verify(False)
    manifest.write_text('{"format":', encoding="utf-8")
    verify(False)
    manifest.write_bytes(original)
    library.write_bytes(b"changed library")
    verify(False)
    run(*build_args)
    verify()
    last_good = manifest.read_bytes()
    library.unlink()
    result = subprocess.run(build_args, capture_output=True, text=True)
    assert result.returncode != 0
    assert manifest.read_bytes() == last_good, "Failed emission must preserve last successful metadata"
    verify(False)

print("Runtime build manifest producer/consumer checks passed")
