#!/usr/bin/env python
"""Build the native host and run the ARC Electron editor."""

from __future__ import print_function

import argparse
import io
import multiprocessing
import os
import platform
import subprocess
import sys


DEFAULT_BUILD_DIR = "out/build/editor-vulkan"
DEFAULT_NO_VULKAN_BUILD_DIR = "out/build/editor-no-vulkan"


def find_executable(name):
    if os.path.isabs(name) and os.path.exists(name):
        return name

    path = os.environ.get("PATH", "")
    extensions = [""]
    if platform.system() == "Windows":
        extensions = os.environ.get("PATHEXT", ".EXE;.BAT;.CMD").split(os.pathsep)

    for directory in path.split(os.pathsep):
        for extension in extensions:
            candidate = os.path.join(directory, name + extension)
            if os.path.exists(candidate):
                return candidate
    return None


def cpu_count():
    try:
        # Unbounded MSBuild node creation is counterproductive on high-core
        # workstations and can exhaust Windows process resources.
        return min(multiprocessing.cpu_count(), 16)
    except NotImplementedError:
        return 1


def parse_args():
    parser = argparse.ArgumentParser(description="Build and run the ARC editor.")
    parser.add_argument("--editor-dir", default="editor", help="Electron editor directory.")
    parser.add_argument("--npm", default="npm", help="npm executable to invoke.")
    parser.add_argument("--npm-script", default="dev", help="npm script used to launch the editor.")
    parser.add_argument(
        "--skip-npm-install",
        action="store_true",
        help="Do not install Electron dependencies when node_modules is missing.",
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
    parser.add_argument("--force-build", action="store_true", help="Force native and npm preparation work.")
    parser.add_argument("--build-only", action="store_true", help="Prepare and validate the editor without launching it.")
    return parser.parse_args()


def run(command, cwd, env=None):
    print("+ " + " ".join(command))
    sys.stdout.flush()
    subprocess.check_call(command, cwd=cwd, env=env)


def cmake_cache_matches(build_dir, vulkan_render):
    cache = os.path.join(build_dir, "CMakeCache.txt")
    if not os.path.exists(cache):
        return False

    try:
        with io.open(cache, "r", encoding="utf-8", errors="replace") as handle:
            text = handle.read()
    except IOError:
        return False

    expected_vulkan = "ON" if vulkan_render else "OFF"
    return (
        "ARC_BUILD_EDITOR:BOOL=ON" in text
        and "ARC_BUILD_RENDER_VULKAN:BOOL={}".format(expected_vulkan) in text
    )


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


def prepare_native_host(args, repo_root):
    build_dir_name = args.build_dir
    if build_dir_name == DEFAULT_BUILD_DIR and not args.vulkan_render:
        build_dir_name = DEFAULT_NO_VULKAN_BUILD_DIR
    build_dir = os.path.abspath(os.path.join(repo_root, build_dir_name))
    cmake = find_executable(args.cmake)
    if cmake is None:
        raise RuntimeError("could not find CMake executable '{}'".format(args.cmake))

    host = find_host_executable(build_dir, args.config)
    cache_matches = cmake_cache_matches(build_dir, args.vulkan_render)
    if args.force_build or host is None or not cache_matches:
        if not cache_matches:
            run(
                [
                    cmake,
                    "-B",
                    build_dir,
                    "-S",
                    repo_root,
                    "-DCMAKE_BUILD_TYPE={}".format(args.config),
                    "-DARC_BUILD_EDITOR=ON",
                    "-DARC_BUILD_RENDER_VULKAN={}".format("ON" if args.vulkan_render else "OFF"),
                ],
                repo_root,
            )
        run(
            [
                cmake,
                "--build",
                build_dir,
                "--config",
                args.config,
                "--target",
                "arc_host_process",
                "--parallel",
                args.parallel or str(cpu_count()),
            ],
            repo_root,
        )
        host = find_host_executable(build_dir, args.config)

    if host is None:
        raise RuntimeError("arc_host_process was not found after the native build")
    return host


def dependencies_ready(editor_dir):
    return os.path.isdir(os.path.join(editor_dir, "node_modules"))


def main():
    args = parse_args()
    repo_root = os.path.dirname(os.path.abspath(__file__))
    editor_dir = os.path.abspath(os.path.join(repo_root, args.editor_dir))
    if not os.path.isdir(editor_dir):
        print("error: editor directory was not found: {}".format(editor_dir), file=sys.stderr)
        return 1

    try:
        host = prepare_native_host(args, repo_root)
    except (RuntimeError, subprocess.CalledProcessError) as error:
        print("error: {}".format(error), file=sys.stderr)
        return 1

    npm = find_executable(args.npm)
    if npm is None:
        print("error: could not find npm executable '{}'".format(args.npm), file=sys.stderr)
        return 1

    try:
        if not args.skip_npm_install and (args.force_build or not dependencies_ready(editor_dir)):
            run([npm, "install"], editor_dir)

        editor_env = os.environ.copy()
        editor_env["ARC_HOST_PROCESS_PATH"] = host
        if args.build_only:
            run([npm, "run", "typecheck"], editor_dir, editor_env)
            print("ARC Editor is ready: {}".format(editor_dir))
            print("Native host: {}".format(host))
            return 0

        command = [npm, "run", args.npm_script]
        print("+ " + " ".join(command))
        sys.stdout.flush()
        return subprocess.call(command, cwd=editor_dir, env=editor_env)
    except subprocess.CalledProcessError as error:
        return error.returncode


if __name__ == "__main__":
    sys.exit(main())
