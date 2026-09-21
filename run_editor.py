#!/usr/bin/env python
"""Build the native host and run the ARC Electron editor."""

from __future__ import print_function

import argparse
import io
import json
import os
import platform
import subprocess
import sys

from tools import arc_build


DEFAULT_BUILD_DIR = "out/build/editor-vulkan"
DEFAULT_NO_VULKAN_BUILD_DIR = "out/build/editor-no-vulkan"
DEFAULT_QUICK_START_PROJECT = os.path.join("out", "editor-quick-start-project")


def parse_args():
    parser = argparse.ArgumentParser(description="Build and run the ARC editor.")
    prerequisites = parser.add_mutually_exclusive_group()
    prerequisites.add_argument(
        "--check-prerequisites",
        action="store_true",
        help="Check the editor development prerequisites and exit.",
    )
    prerequisites.add_argument(
        "--install-prerequisites",
        action="store_true",
        help="Install supported missing prerequisites, then report anything that still needs manual setup.",
    )
    parser.add_argument(
        "--no-install",
        action="store_true",
        help="Skip prerequisite detection and install prompts.",
    )
    parser.add_argument("--editor-dir", default="editor", help="Electron editor directory.")
    parser.add_argument("--npm", default="npm", help="npm executable to invoke.")
    parser.add_argument("--npm-script", default="dev", help="npm script used to launch the editor.")
    parser.add_argument(
        "--skip-npm-install",
        action="store_true",
        help="Do not install or repair Electron dependencies when node_modules is missing or stale.",
    )
    parser.add_argument("--build-dir", default=DEFAULT_BUILD_DIR, help="CMake build directory for the native host.")
    parser.add_argument("--config", default="Release", help="Native host build configuration.")
    parser.add_argument("--cmake", default="cmake", help="CMake executable to invoke.")
    parser.add_argument("--parallel", default=None, help="Native build job count. Defaults to the host CPU count.")
    parser.add_argument(
        "--no-vulkan-render",
        action="store_false",
        dest="vulkan_render",
        default=True,
        help="Build the native host without the Vulkan viewport backend.",
    )
    parser.add_argument(
        "--force-build",
        action="store_true",
        help="Discard the native CMake build tree and rerun native/npm preparation.",
    )
    parser.add_argument("--build-only", action="store_true", help="Prepare and validate the editor without launching it.")
    parser.add_argument(
        "--quick-start",
        action="store_true",
        help="Open a persistent Blank 3D development project and bypass the project browser.",
    )
    parser.add_argument(
        "--clear-asset-db",
        nargs="?",
        const="",
        default=None,
        metavar="PROJECT",
        help=(
            "Delete the rebuildable .arc/cache/assets.db registry before launch. "
            "Defaults to the quick-start project; optionally pass a project root or .arcproject path."
        ),
    )
    parser.add_argument(
        "--ui-lab",
        action="store_true",
        help="Launch the standalone editor UI control lab without building or starting the native engine host.",
    )
    parser.add_argument(
        "--perf",
        action="store_true",
        help="Enable ARC editor performance diagnostics ([ARC PERF] startup and slow-operation timings).",
    )
    parser.add_argument(
        "--perf-slow-ms",
        type=float,
        default=None,
        metavar="MS",
        help="Set the slow-operation threshold in milliseconds and enable performance diagnostics.",
    )
    args = parser.parse_args()
    if args.ui_lab and args.quick_start:
        parser.error("--ui-lab and --quick-start cannot be used together")
    if args.no_install and (args.check_prerequisites or args.install_prerequisites):
        parser.error("--no-install cannot be combined with prerequisite check/install commands")
    if args.perf_slow_ms is not None and args.perf_slow_ms < 0:
        parser.error("--perf-slow-ms must be zero or greater")
    return args


def clear_asset_database(repo_root, project_argument):
    project_path = project_argument or os.environ.get("ARC_EDITOR_QUICK_START_PROJECT") or DEFAULT_QUICK_START_PROJECT
    project_path = os.path.abspath(os.path.join(repo_root, os.path.expanduser(project_path)))
    if project_path.lower().endswith(".arcproject"):
        project_path = os.path.dirname(project_path)

    cache_dir = os.path.join(project_path, ".arc", "cache")
    removed = []
    if os.path.isdir(cache_dir):
        for name in os.listdir(cache_dir):
            if not name.startswith("assets.db"):
                continue
            candidate = os.path.join(cache_dir, name)
            if os.path.isfile(candidate):
                os.remove(candidate)
                removed.append(candidate)

    if removed:
        print("Cleared ARC asset database for {}".format(project_path))
        for candidate in removed:
            print("  removed {}".format(candidate))
    else:
        print("No ARC asset database found for {}".format(project_path))


def host_executable_candidates(build_dir, config):
    executable = "arc_host_process.exe" if platform.system() == "Windows" else "arc_host_process"
    return [
        os.path.join(build_dir, "editor", "native", config, executable),
        os.path.join(build_dir, "editor", "native", executable),
    ]


def find_host_executable(build_dir, config):
    for candidate in host_executable_candidates(build_dir, config):
        if os.path.exists(candidate):
            return candidate
    return None


def project_tool_executable_candidates(build_dir, config):
    executable = "arc-project.exe" if platform.system() == "Windows" else "arc-project"
    return [
        os.path.join(build_dir, "tools", "project_cli", config, executable),
        os.path.join(build_dir, "tools", "project_cli", executable),
    ]


def find_project_tool_executable(build_dir, config):
    for candidate in project_tool_executable_candidates(build_dir, config):
        if os.path.exists(candidate):
            return candidate
    return None


def prepare_native_editor(args, repo_root, env=None):
    build_dir_name = args.build_dir
    if build_dir_name == DEFAULT_BUILD_DIR and not args.vulkan_render:
        build_dir_name = DEFAULT_NO_VULKAN_BUILD_DIR
    build_dir = os.path.abspath(os.path.join(repo_root, build_dir_name))
    cmake = arc_build.find_executable(args.cmake)
    if cmake is None:
        raise RuntimeError("could not find CMake executable '{}'".format(args.cmake))

    if args.force_build:
        arc_build.reset_cmake_build_directory(build_dir, cmake=cmake)

    generator = arc_build.resolve_visual_studio_generator(cmake)
    existing_generator = arc_build.cmake_cache_generator(build_dir)
    if generator and existing_generator and existing_generator != generator:
        raise RuntimeError(
            "CMake build directory '{}' uses generator '{}'; rerun with --force-build to discard it and reconfigure with '{}'".format(
                build_dir, existing_generator, generator
            )
        )

    if generator:
        existing_platform = arc_build.cmake_cache_generator_platform(build_dir)
        if existing_platform is not None and existing_platform.lower() != "x64":
            platform_label = existing_platform or "<default>"
            raise RuntimeError(
                "CMake build directory '{}' uses generator platform '{}'; rerun with --force-build to discard it and reconfigure for 'x64'".format(
                    build_dir, platform_label
                )
            )

    # Always run CMake configure before building. It is incremental for a
    # healthy tree and repairs partially generated trees (for example a cache
    # that exists while the Visual Studio project files do not).
    configure_command = [
        cmake,
        "-B",
        build_dir,
        "-S",
        repo_root,
        "-DCMAKE_BUILD_TYPE={}".format(args.config),
        "-DARC_BUILD_EDITOR=ON",
        "-DARC_BUILD_RENDER_VULKAN={}".format("ON" if args.vulkan_render else "OFF"),
        "-DFETCHCONTENT_FULLY_DISCONNECTED=OFF",
    ]
    if generator:
        # Visual Studio generators initialize the MSVC environment themselves,
        # so Windows builds do not depend on nmake.exe or a Developer Command Prompt.
        configure_command.extend(["-G", generator, "-A", "x64"])
    arc_build.run(configure_command, repo_root, env)

    # Always ask the build system for the host. CMake/MSBuild/Ninja perform an
    # incremental no-op when it is current, while checking timestamps prevents
    # Electron from speaking a newer protocol to a stale executable.
    arc_build.build_cmake_target(
        cmake,
        build_dir,
        "arc_host_process",
        args.config,
        repo_root,
        env,
        args.parallel,
    )
    arc_build.build_cmake_target(
        cmake,
        build_dir,
        "arc-project-cli",
        args.config,
        repo_root,
        env,
        args.parallel,
    )
    host = find_host_executable(build_dir, args.config)
    project_tool = find_project_tool_executable(build_dir, args.config)

    if host is None:
        raise RuntimeError("arc_host_process was not found after the native build")
    if project_tool is None:
        raise RuntimeError("arc-project was not found after the native build")
    return host, project_tool


def editor_dependency_names(editor_dir):
    package_json = os.path.join(editor_dir, "package.json")
    try:
        with io.open(package_json, "r", encoding="utf-8") as handle:
            manifest = json.load(handle)
    except (IOError, OSError, ValueError):
        return None

    names = set()
    for section in ("dependencies", "devDependencies"):
        dependencies = manifest.get(section, {})
        if not isinstance(dependencies, dict):
            return None
        names.update(dependencies.keys())
    return sorted(names)


def npm_dependency_path(node_modules, dependency):
    return os.path.join(node_modules, *dependency.split("/"))


def dependencies_ready(editor_dir):
    node_modules = os.path.join(editor_dir, "node_modules")
    package_json = os.path.join(editor_dir, "package.json")
    package_lock = os.path.join(editor_dir, "package-lock.json")
    installed_lock = os.path.join(node_modules, ".package-lock.json")

    if not os.path.isdir(node_modules):
        return False
    if not os.path.isfile(package_json) or not os.path.isfile(package_lock) or not os.path.isfile(installed_lock):
        return False

    dependencies = editor_dependency_names(editor_dir)
    if dependencies is None:
        return False
    for dependency in dependencies:
        installed_package = npm_dependency_path(node_modules, dependency)
        if not os.path.isfile(os.path.join(installed_package, "package.json")):
            return False

    try:
        source_mtime = max(os.path.getmtime(package_json), os.path.getmtime(package_lock))
        if os.path.getmtime(installed_lock) < source_mtime:
            return False
    except OSError:
        return False

    return True


def install_editor_dependencies(npm, editor_dir):
    package_lock = os.path.join(editor_dir, "package-lock.json")
    if not os.path.isfile(package_lock):
        raise RuntimeError("editor/package-lock.json is required for deterministic npm setup")

    print("Installing ARC editor npm dependencies...")
    arc_build.run([npm, "ci"], editor_dir)
    if not dependencies_ready(editor_dir):
        raise RuntimeError("npm ci completed but the ARC editor dependency tree is still incomplete")


def main():
    args = parse_args()
    repo_root = os.path.dirname(os.path.abspath(__file__))

    if args.check_prerequisites:
        checks = arc_build.check_editor_prerequisites(cmake=args.cmake, npm=args.npm)
        arc_build.print_prerequisite_report(checks)
        return 0 if arc_build.prerequisites_ready(checks) else 1

    if args.install_prerequisites:
        checks = arc_build.check_editor_prerequisites(cmake=args.cmake, npm=args.npm)
        arc_build.print_prerequisite_report(checks, show_install_hint=False)
        if arc_build.prerequisites_ready(checks):
            return 0

        try:
            _, installed = arc_build.install_editor_prerequisites(cmake=args.cmake, npm=args.npm)
        except (RuntimeError, OSError, subprocess.CalledProcessError) as error:
            print("error: {}".format(error), file=sys.stderr)
            return 1

        if installed:
            print("")
            print("Selected prerequisites were installed.")
            print("Open a new terminal so PATH updates are visible, then run:")
            print("  python run_editor.py --check-prerequisites")
        return 1
    if args.clear_asset_db is not None:
        try:
            clear_asset_database(repo_root, args.clear_asset_db)
        except OSError as error:
            print("error: could not clear ARC asset database: {}".format(error), file=sys.stderr)
            return 1

    editor_dir = os.path.abspath(os.path.join(repo_root, args.editor_dir))
    if not os.path.isdir(editor_dir):
        print("error: editor directory was not found: {}".format(editor_dir), file=sys.stderr)
        return 1

    if not args.no_install:
        prerequisite_checks = arc_build.check_editor_prerequisites(
            cmake=args.cmake,
            npm=args.npm,
            require_native=not args.ui_lab,
        )
        if not arc_build.prerequisites_ready(prerequisite_checks):
            arc_build.print_prerequisite_report(prerequisite_checks, show_install_hint=False)
            try:
                _, installed = arc_build.install_editor_prerequisites(
                    cmake=args.cmake,
                    npm=args.npm,
                    require_native=not args.ui_lab,
                )
            except (RuntimeError, OSError, subprocess.CalledProcessError) as error:
                print("error: {}".format(error), file=sys.stderr)
                return 1

            if installed:
                print("")
                print("Selected prerequisites were installed.")
                print("Open a new terminal so PATH updates are visible, then rerun the editor.")
            return 1

    host = None
    project_tool = None
    tool_env = os.environ.copy()
    if args.perf or args.perf_slow_ms is not None:
        tool_env["ARC_EDITOR_PERF"] = "1"
    if args.perf_slow_ms is not None:
        tool_env["ARC_EDITOR_PERF_SLOW_MS"] = str(args.perf_slow_ms)
    if not args.ui_lab:
        try:
            slangc = arc_build.resolve_slangc(repo_root)
            arc_build.add_slang_to_environment(tool_env, slangc)
            host, project_tool = prepare_native_editor(args, repo_root, tool_env)
        except (RuntimeError, OSError, subprocess.CalledProcessError) as error:
            print("error: {}".format(error), file=sys.stderr)
            return 1

    npm = arc_build.find_executable(args.npm)
    if npm is None:
        print("error: could not find npm executable '{}'".format(args.npm), file=sys.stderr)
        return 1

    try:
        if not args.skip_npm_install and (args.force_build or not dependencies_ready(editor_dir)):
            install_editor_dependencies(npm, editor_dir)

        editor_env = tool_env.copy()
        if args.ui_lab:
            editor_env["VITE_ARC_UI_LAB"] = "1"
        if host is not None and project_tool is not None:
            editor_env["ARC_HOST_PROCESS_PATH"] = host
            editor_env["ARC_PROJECT_TOOL_PATH"] = project_tool
            editor_env["ARC_PROJECT_TEMPLATES_PATH"] = os.path.join(repo_root, "templates")
        if args.quick_start:
            editor_env["ARC_EDITOR_QUICK_START_PROJECT"] = os.path.join(repo_root, DEFAULT_QUICK_START_PROJECT)
        if args.build_only:
            arc_build.run([npm, "run", "typecheck"], editor_dir, editor_env)
            print("ARC Editor is ready: {}".format(editor_dir))
            if host is not None:
                print("Native host: {}".format(host))
            return 0

        command = [npm, "run", args.npm_script]
        if args.ui_lab:
            # npm consumes the first separator and Electron Forge consumes the
            # second before forwarding the switch to the Electron process.
            command.extend(["--", "--", "--ui-lab"])
        elif args.quick_start:
            # npm consumes the first separator and Electron Forge consumes the
            # second before forwarding the switch to the Electron process.
            command.extend(["--", "--", "--quick-start"])
        print("+ " + " ".join(command))
        sys.stdout.flush()
        return subprocess.call(command, cwd=editor_dir, env=editor_env)
    except subprocess.CalledProcessError as error:
        return error.returncode


if __name__ == "__main__":
    sys.exit(main())
