"""Regression: fresh project Build must configure CMake and surface useful failures.

Run against the real arc-project executable with a deliberately tiny external
CMake project. No installed ARC SDK, Vulkan, GPU, Android NDK or VS build is
needed: this specifically exercises the native CLI's configure/build lifecycle.
"""
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

tool, templates, version, cmake = sys.argv[1:5]


def run(*arguments, success=True):
    result = subprocess.run(
        [tool, *arguments, "--json"], capture_output=True, text=True, check=False,
        timeout=90,
    )
    # CMake must never corrupt the machine-readable JSON stream. The caller's
    # redirected stderr instead contains compiler/configure details.
    try:
        payload = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise AssertionError(
            f"arc-project returned non-JSON output: {result.stdout!r}\nstderr:\n{result.stderr}"
        ) from exc
    assert (result.returncode == 0) == success, (arguments, payload, result.stderr)
    assert payload["succeeded"] is success, (arguments, payload)
    return payload, result.stderr


with tempfile.TemporaryDirectory(prefix="arc first build with spaces ") as directory:
    root = Path(directory)
    run("create", "--name", "FirstBuild", "--destination", str(root),
        "--template", "blank-3d", "--templates", templates, "--engine", version)
    descriptor = root / "FirstBuild.arcproject"
    source = root / "CMakeLists.txt"
    build_root = root / "Build"
    project = ["--project", str(descriptor), "--cmake", cmake]
    generator = ["--generator", "Ninja"] if shutil.which("ninja") else []
    if len(sys.argv) > 5:
        generator = ["--generator", sys.argv[5]]

    # Fresh Clean is a no-op, including for Rebuild's initial clean phase.
    payload, _ = run("build", *project, "--target", "clean")
    assert payload["nothingToClean"] is True
    assert not (build_root / "default").exists()

    source.write_text(r'''
cmake_minimum_required(VERSION 3.20)
project(FirstBuildSmoke NONE)
add_custom_target(FirstBuildSmoke ALL
    COMMAND "${CMAKE_COMMAND}" -E touch "${CMAKE_BINARY_DIR}/built.txt")
''', encoding="utf-8")

    sdk = root / "Example ARC SDK"
    sdk.mkdir()
    payload, stderr = run(
        "build", *project, "--build-dir", "default",
        "--config", "RelWithDebInfo", *generator, "--sdk", str(sdk),
    )
    target = build_root / "default"
    assert (target / "CMakeCache.txt").is_file(), (payload, stderr)
    assert (target / "built.txt").is_file(), (payload, stderr)
    assert payload["buildDirectory"] == target.as_posix(), payload
    cache = (target / "CMakeCache.txt").read_text(encoding="utf-8")
    # CMake may preserve native backslashes when caching a -D path on
    # Windows, while paths generated elsewhere use forward slashes. Compare
    # the resolved SDK directories, not their textual separators.
    prefix_entries = [
        line.partition("=")[2]
        for line in cache.splitlines()
        if line.startswith("CMAKE_PREFIX_PATH:")
    ]
    assert len(prefix_entries) == 1, "Missing CMAKE_PREFIX_PATH in CMakeCache.txt"
    configured_sdk = Path(prefix_entries[0].replace("\\", "/")).resolve()
    assert configured_sdk == sdk.resolve(), (
        f"Configured SDK differs from requested SDK: {configured_sdk!s} != {sdk!s}"
    )
    assert any(line.startswith("CMAKE_BUILD_TYPE:") and line.endswith("=RelWithDebInfo")
               for line in cache.splitlines()), "The first build lost its selected configuration"

    # Subsequent builds are incremental; the cache and generated build system
    # are reused rather than demanding manual Configure.
    run("build", *project, "--config", "RelWithDebInfo")
    payload, _ = run("build", *project, "--target", "clean")
    assert payload["succeeded"], payload
    run("build", *project)
    assert (target / "built.txt").exists()

    # Even when a CMake configure fails, surface its detailed diagnostics
    # through stderr and an actionable CLI error, not "CMake command failed".
    source.write_text(r'''
cmake_minimum_required(VERSION 3.20)
project(FirstBuildSmoke NONE)
message(FATAL_ERROR "Deliberate configure failure from test")
''', encoding="utf-8")
    payload, stderr = run("build", *project, "--build-dir", "configure-fails",
                          *generator, success=False)
    assert "CMake configuration failed" in payload["error"], payload
    assert "--sdk" in payload["error"] and "configure-fails" in payload["error"]
    assert "Deliberate configure failure from test" in stderr, stderr

    # Failed configuration leaves a cache but no usable native build system.
    # Repeated Build must retry configuration, not invoke a missing ALL_BUILD
    # project (MSVC) or build.ninja (Ninja).
    assert (build_root / "configure-fails" / "CMakeCache.txt").is_file()
    payload, stderr = run("build", *project, "--build-dir", "configure-fails",
                          *generator, success=False)
    assert "CMake configuration failed" in payload["error"], payload
    assert "Deliberate configure failure from test" in stderr, stderr

    # Correcting the source/dependency must recover the same directory on
    # Build without manual deletion of the cache or a separate Configure.
    source.write_text(r'''
cmake_minimum_required(VERSION 3.20)
project(FirstBuildSmoke NONE)
add_custom_target(Recovered ALL
    COMMAND "${CMAKE_COMMAND}" -E touch "${CMAKE_BINARY_DIR}/recovered.txt")
''', encoding="utf-8")
    run("build", *project, "--build-dir", "configure-fails", *generator)
    assert (build_root / "configure-fails" / "recovered.txt").is_file()

    # BM7.2: opt-in telemetry is forwarded only for the Visual Studio
    # generator. A distinct log must be produced without disturbing JSON
    # output or normal native build behavior.
    if len(generator) == 2 and generator[1].startswith("Visual Studio "):
        log_folder = build_root / "configure-fails" / ".arc-monitor"
        log_folder.mkdir(parents=True, exist_ok=True)
        log_path = log_folder / "arc-msbuild-cli-smoke.log"
        run("build", *project, "--build-dir", "configure-fails", *generator,
            "--msbuild-timing-log", str(log_path))
        assert log_path.is_file() and log_path.stat().st_size > 0, "Missing MSBuild diagnostic log"
        unsafe = root / "outside-build.log"
        payload, _ = run("build", *project, "--build-dir", "configure-fails", *generator,
                         "--msbuild-timing-log", str(unsafe), success=False)
        assert "Invalid MSBuild timing log path" in payload["error"], payload
        assert not unsafe.exists(), "Native logger wrote outside the build output"

    # A compile error after successful configure must be distinguished from
    # a configure failure and must preserve CMake's actual error output.
    source.write_text(r'''
cmake_minimum_required(VERSION 3.20)
project(FirstBuildSmoke NONE)
add_custom_target(Failing ALL
    COMMAND "${CMAKE_COMMAND}" -E compare_files
            "${CMAKE_BINARY_DIR}/missing-a" "${CMAKE_BINARY_DIR}/missing-b")
''', encoding="utf-8")
    payload, stderr = run("build", *project, "--build-dir", "compile-fails",
                          *generator, success=False)
    assert "CMake build failed" in payload["error"], payload
    assert "compile-fails" in payload["error"] and "configuration failed" not in payload["error"]
    assert stderr.strip(), "CMake stderr must be preserved on build failures"

print("ARC first-build auto-configure, clean, and CMake diagnostics passed")
