#!/usr/bin/env python
"""Build and run the arc editor.

By default this launches the Electron-based editor2 shell. Use --native to run
legacy ImGui/SDL editor targets.
"""

from __future__ import print_function

import argparse
import io
import multiprocessing
import os
import platform
import subprocess
import sys

DEFAULT_NATIVE_BUILD_DIR = "out/build/editor-vulkan"
DEFAULT_NATIVE_NO_VULKAN_BUILD_DIR = "out/build/editor-no-vulkan"


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
        return multiprocessing.cpu_count()
    except NotImplementedError:
        return 1


def parse_args():
    parser = argparse.ArgumentParser(description="Build and run the arc editor.")
    parser.add_argument(
        "--native",
        action="store_true",
        help="Run the legacy native ImGui/SDL editor instead of the Electron editor2 shell.",
    )
    parser.add_argument("--editor2-dir", default="editor2", help="Electron editor2 directory to use.")
    parser.add_argument("--npm", default="npm", help="npm executable to invoke for editor2.")
    parser.add_argument(
        "--editor2-script",
        default="dev",
        help="npm script to run when launching editor2. Defaults to dev.",
    )
    parser.add_argument(
        "--skip-npm-install",
        action="store_true",
        help="Do not automatically run npm install when editor2 dependencies are missing.",
    )
    parser.add_argument(
        "--build-dir",
        default=DEFAULT_NATIVE_BUILD_DIR,
        help="CMake build directory to use for the native editor.",
    )
    parser.add_argument("--config", default="Release", help="CMake configuration to build and run for the native editor.")
    parser.add_argument("--cmake", default="cmake", help="CMake executable to invoke for the native editor.")
    parser.add_argument("--parallel", default=None, help="Parallel native build job count. Defaults to the host CPU count.")
    parser.add_argument(
        "--no-vulkan-render",
        action="store_false",
        dest="vulkan_render",
        default=True,
        help="Run the native editor without building the Vulkan render backend.",
    )
    parser.add_argument("--force-build", action="store_true", help="Force dependency install/build work before launching.")
    parser.add_argument("--build-only", action="store_true", help="Prepare the editor without launching it.")
    return parser.parse_args()


def run(command, cwd):
    print("+ " + " ".join(command))
    sys.stdout.flush()
    subprocess.check_call(command, cwd=cwd)


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
        "ARC_BUILD_EDITOR:BOOL=ON" in text and
        "ARC_BUILD_RENDER_VULKAN:BOOL={}".format(expected_vulkan) in text
    )


def native_editor_executable_candidates(build_dir, config):
    executable = "arc_editor.exe" if platform.system() == "Windows" else "arc_editor"
    return [
        os.path.join(build_dir, "editor", config, executable),
        os.path.join(build_dir, "editor", executable),
    ]


def find_native_editor_executable(build_dir, config):
    for candidate in native_editor_executable_candidates(build_dir, config):
        if os.path.exists(candidate):
            return candidate
    return None


def run_native_editor(args, repo_root):
    build_dir_name = args.build_dir
    if build_dir_name == DEFAULT_NATIVE_BUILD_DIR and not args.vulkan_render:
        build_dir_name = DEFAULT_NATIVE_NO_VULKAN_BUILD_DIR
    build_dir = os.path.abspath(os.path.join(repo_root, build_dir_name))
    cmake = find_executable(args.cmake)

    if cmake is None:
        print("error: could not find CMake executable '{}'".format(args.cmake), file=sys.stderr)
        return 1

    executable = find_native_editor_executable(build_dir, args.config)
    cache_matches = cmake_cache_matches(build_dir, args.vulkan_render)
    needs_build = args.force_build or executable is None or not cache_matches

    if needs_build:
        configure_command = [
            cmake,
            "-B",
            build_dir,
            "-S",
            repo_root,
            "-DCMAKE_BUILD_TYPE={}".format(args.config),
            "-DARC_BUILD_EDITOR=ON",
            "-DARC_BUILD_RENDER_VULKAN={}".format("ON" if args.vulkan_render else "OFF"),
        ]
        if not cache_matches:
            run(configure_command, repo_root)

        jobs = args.parallel or str(cpu_count())
        run([cmake, "--build", build_dir, "--config", args.config, "--target", "arc_editor", "--parallel", jobs], repo_root)
        executable = find_native_editor_executable(build_dir, args.config)

    if executable is None:
        print("error: native editor executable was not found after build", file=sys.stderr)
        return 1

    if args.build_only:
        print("native editor is ready: {}".format(executable))
        return 0

    print("+ {}".format(executable))
    sys.stdout.flush()
    return subprocess.call([executable], cwd=repo_root, env=os.environ.copy())


def editor2_dependencies_ready(editor2_dir):
    return os.path.isdir(os.path.join(editor2_dir, "node_modules"))


def run_editor2(args, repo_root):
    editor2_dir = os.path.abspath(os.path.join(repo_root, args.editor2_dir))
    npm = find_executable(args.npm)

    if not os.path.isdir(editor2_dir):
        print("error: editor2 directory was not found: {}".format(editor2_dir), file=sys.stderr)
        return 1

    if npm is None:
        print("error: could not find npm executable '{}'".format(args.npm), file=sys.stderr)
        return 1

    if not args.skip_npm_install and (args.force_build or not editor2_dependencies_ready(editor2_dir)):
        run([npm, "install"], editor2_dir)

    if args.build_only:
        run([npm, "run", "typecheck"], editor2_dir)
        print("editor2 is ready: {}".format(editor2_dir))
        return 0

    command = [npm, "run", args.editor2_script]
    print("+ " + " ".join(command))
    sys.stdout.flush()
    return subprocess.call(command, cwd=editor2_dir, env=os.environ.copy())


def main():
    args = parse_args()
    repo_root = os.path.dirname(os.path.abspath(__file__))

    if args.native:
        return run_native_editor(args, repo_root)

    return run_editor2(args, repo_root)


if __name__ == "__main__":
    sys.exit(main())
