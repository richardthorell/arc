#!/usr/bin/env python
"""Build and run the ARC editor."""

from __future__ import print_function

import argparse
import io
import multiprocessing
import os
import platform
import subprocess
import sys


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
    parser = argparse.ArgumentParser(description="Build and run the ARC editor.")
    parser.add_argument("--build-dir", default="build", help="CMake build directory to use.")
    parser.add_argument("--config", default="Release", help="CMake configuration to build and run.")
    parser.add_argument("--cmake", default="cmake", help="CMake executable to invoke.")
    parser.add_argument("--parallel", default=None, help="Parallel build job count. Defaults to the host CPU count.")
    parser.add_argument("--force-build", action="store_true", help="Build even if the editor executable already exists.")
    parser.add_argument("--build-only", action="store_true", help="Build the editor without launching it.")
    return parser.parse_args()


def run(command, cwd):
    print("+ " + " ".join(command))
    sys.stdout.flush()
    subprocess.check_call(command, cwd=cwd)


def cmake_cache_has_editor(build_dir):
    cache = os.path.join(build_dir, "CMakeCache.txt")
    if not os.path.exists(cache):
        return False

    try:
        with io.open(cache, "r", encoding="utf-8", errors="replace") as handle:
            text = handle.read()
    except IOError:
        return False

    return "ARC_BUILD_EDITOR:BOOL=ON" in text


def editor_executable_candidates(build_dir, config):
    executable = "arc_editor.exe" if platform.system() == "Windows" else "arc_editor"
    return [
        os.path.join(build_dir, "editor", config, executable),
        os.path.join(build_dir, "editor", executable),
    ]


def find_editor_executable(build_dir, config):
    for candidate in editor_executable_candidates(build_dir, config):
        if os.path.exists(candidate):
            return candidate
    return None


def main():
    args = parse_args()
    repo_root = os.path.dirname(os.path.abspath(__file__))
    build_dir = os.path.abspath(os.path.join(repo_root, args.build_dir))
    cmake = find_executable(args.cmake)

    if cmake is None:
        print("error: could not find CMake executable '{}'".format(args.cmake), file=sys.stderr)
        return 1

    executable = find_editor_executable(build_dir, args.config)
    needs_build = args.force_build or executable is None

    if needs_build:
        configure_command = [
            cmake,
            "-B",
            build_dir,
            "-S",
            repo_root,
            "-DCMAKE_BUILD_TYPE={}".format(args.config),
            "-DARC_BUILD_EDITOR=ON",
        ]
        if not cmake_cache_has_editor(build_dir):
            run(configure_command, repo_root)

        jobs = args.parallel or str(cpu_count())
        run([cmake, "--build", build_dir, "--config", args.config, "--target", "arc_editor", "--parallel", jobs], repo_root)
        executable = find_editor_executable(build_dir, args.config)

    if executable is None:
        print("error: editor executable was not found after build", file=sys.stderr)
        return 1

    if args.build_only:
        print("editor is ready: {}".format(executable))
        return 0

    print("+ {}".format(executable))
    sys.stdout.flush()
    return subprocess.call([executable], cwd=repo_root, env=os.environ.copy())


if __name__ == "__main__":
    sys.exit(main())
