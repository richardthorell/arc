"""Regression: fresh project Build must configure CMake and surface useful failures.

Run against the real arc-project executable with a deliberately tiny external
CMake project. No installed ARC SDK, Vulkan, GPU, Android NDK or VS build is
needed: this specifically exercises the native CLI's configure/build lifecycle.
"""
import json
import os
import re
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

tool, templates, version, cmake = sys.argv[1:5]


def run(*arguments, success=True, environment=None):
    env = os.environ.copy()
    env.pop("ARC_DEVELOPMENT_SDK", None)
    if environment:
        env.update(environment)
    result = subprocess.run(
        [tool, *arguments, "--json"], capture_output=True, text=True, check=False,
        timeout=90, env=env,
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
    # BM7.3: Ninja can only join measured object outputs to source files when
    # the configure step writes CMake's native compilation database.
    assert any(line.startswith("CMAKE_EXPORT_COMPILE_COMMANDS:") and line.endswith("=ON")
               for line in cache.splitlines()), "Source drill-down compilation database is disabled"

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
        log_text = log_path.read_text(encoding="utf-8", errors="replace")
        assert re.search(r"\b\d{1,2}:\d{2}:\d{2}\b", log_text), "MSBuild logger omitted native timestamps"
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

    # The source editor supplies an explicit, versioned SDK staging root.
    # Exercise the real CLI/CMake handoff without rebuilding the whole engine.
    engine_source = root / "engine-source"
    engine_source.mkdir()
    engine_build = root / "engine-build"
    engine_source.joinpath("CMakeLists.txt").write_text('''
cmake_minimum_required(VERSION 3.20)
project(DevelopmentSDK NONE)
add_custom_target(arc-development-sdk
    COMMAND "${CMAKE_COMMAND}" -E touch "${CMAKE_BINARY_DIR}/prepared.txt")
''', encoding="utf-8")
    subprocess.run([cmake, "-S", str(engine_source), "-B", str(engine_build)], check=True,
                   capture_output=True, timeout=90)
    staging = root / "development SDK"
    prefix = staging / "RelWithDebInfo"
    package = prefix / "lib" / "cmake" / "ARC"
    package.mkdir(parents=True)
    package.joinpath("ARCConfig.cmake").write_text('set(ARC_FOUND TRUE)\n', encoding="utf-8")
    metadata = {"version": 1, "engineVersion": version, "configuration": "RelWithDebInfo",
                "buildDirectory": engine_build.as_posix()}
    manifest = prefix / "arc-development-sdk.json"
    manifest.write_text(json.dumps(metadata), encoding="utf-8")
    source.write_text('''
cmake_minimum_required(VERSION 3.20)
project(DevelopmentConsumer NONE)
find_package(ARC CONFIG REQUIRED)
add_custom_target(Consumer ALL
    COMMAND "${CMAKE_COMMAND}" -E touch "${CMAKE_BINARY_DIR}/consumer.txt")
''', encoding="utf-8")
    environment = {"ARC_DEVELOPMENT_SDK": str(staging)}
    run("build", *project, "--build-dir", "development-consumer", environment=environment)
    assert (engine_build / "prepared.txt").is_file()
    assert (build_root / "development-consumer" / "consumer.txt").is_file()
    metadata["engineVersion"] = "999.0.0"
    manifest.write_text(json.dumps(metadata), encoding="utf-8")
    payload, _ = run("build", *project, environment=environment, success=False)
    assert "does not support this engine version" in payload["error"]
    payload, _ = run("build", *project, "--config", "Missing", environment=environment, success=False)
    assert "configuration" in payload["error"]
    # Explicit SDK choice takes precedence over the launcher's development SDK.
    run("build", *project, "--build-dir", "explicit-sdk", "--sdk", str(prefix), environment=environment)

print("ARC first-build auto-configure, clean, and CMake diagnostics passed")
