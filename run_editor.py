#!/usr/bin/env python
"""Build the native host and run the ARC Electron editor."""

from __future__ import print_function

import argparse
import io
import multiprocessing
import os
import platform
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
import zipfile

try:
    from urllib.request import urlopen
except ImportError:
    from urllib2 import urlopen


DEFAULT_BUILD_DIR = "out/build/editor-vulkan"
DEFAULT_NO_VULKAN_BUILD_DIR = "out/build/editor-no-vulkan"
DEFAULT_QUICK_START_PROJECT = os.path.join("out", "editor-quick-start-project")
SLANG_VERSION = "2026.14.1"
SLANG_RELEASE_BASE_URL = "https://github.com/shader-slang/slang/releases/download/v{}".format(SLANG_VERSION)


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


def cmake_cache_requires_configure(build_dir, vulkan_render):
    cache = os.path.join(build_dir, "CMakeCache.txt")
    if not os.path.exists(cache):
        return True

    try:
        with io.open(cache, "r", encoding="utf-8", errors="replace") as handle:
            text = handle.read()
    except IOError:
        return True

    expected_vulkan = "ON" if vulkan_render else "OFF"
    return not (
        "ARC_BUILD_EDITOR:BOOL=ON" in text
        and "ARC_BUILD_RENDER_VULKAN:BOOL={}".format(expected_vulkan) in text
        and "FETCHCONTENT_FULLY_DISCONNECTED:BOOL=OFF" in text
    )


def slang_archive():
    machine = platform.machine().lower()
    if machine not in ("amd64", "x86_64"):
        raise RuntimeError("automatic Slang setup currently supports x86_64 hosts; got '{}'".format(machine))

    system = platform.system()
    if system == "Windows":
        return "slang-{}-windows-x86_64.zip".format(SLANG_VERSION)
    if system == "Linux":
        return "slang-{}-linux-x86_64.tar.gz".format(SLANG_VERSION)
    raise RuntimeError(
        "automatic Slang setup is not available on {}; set ARC_SLANGC_EXECUTABLE to Slang {}".format(
            system, SLANG_VERSION
        )
    )


def slang_version_output(executable):
    try:
        return subprocess.check_output(
            [executable, "-version"], stderr=subprocess.STDOUT, universal_newlines=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return ""


def is_pinned_slang(executable):
    if not executable or not os.path.isfile(executable):
        return False
    output = slang_version_output(executable)
    pattern = r"(^|[^0-9.]){}([^0-9.]|$)".format(re.escape(SLANG_VERSION))
    return re.search(pattern, output) is not None


def find_slangc_under(root):
    executable = "slangc.exe" if platform.system() == "Windows" else "slangc"
    if not os.path.isdir(root):
        return None
    for directory, _, files in os.walk(root):
        if executable in files:
            return os.path.join(directory, executable)
    return None


def slang_cache_root(repo_root):
    host = "{}-{}".format(platform.system().lower(), platform.machine().lower())
    return os.path.join(repo_root, "out", "toolchains", "slang", SLANG_VERSION, host)


def remove_partial_download(destination):
    try:
        if os.path.exists(destination):
            os.remove(destination)
    except OSError:
        pass


def download_with_curl(url, destination):
    curl = find_executable("curl")
    if curl is None:
        return False
    try:
        subprocess.check_call(
            [curl, "--fail", "--location", "--retry", "3", "--output", destination, url]
        )
        return True
    except (OSError, subprocess.CalledProcessError):
        remove_partial_download(destination)
        return False


def powershell_literal(value):
    return "'{}'".format(value.replace("'", "''"))


def download_with_powershell(url, destination):
    if platform.system() != "Windows":
        return False
    powershell = find_executable("powershell")
    if powershell is None:
        return False
    command = (
        "$ProgressPreference='SilentlyContinue'; "
        "[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12; "
        "Invoke-WebRequest -UseBasicParsing -Uri {} -OutFile {}"
    ).format(powershell_literal(url), powershell_literal(destination))
    try:
        subprocess.check_call(
            [powershell, "-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass", "-Command", command]
        )
        return True
    except (OSError, subprocess.CalledProcessError):
        remove_partial_download(destination)
        return False


def download_with_python(url, destination):
    try:
        response = urlopen(url)
        try:
            with open(destination, "wb") as output:
                shutil.copyfileobj(response, output)
        finally:
            response.close()
        return True
    except Exception:
        remove_partial_download(destination)
        return False


def download_file(url, destination):
    print("Downloading {}".format(url))
    sys.stdout.flush()

    # Prefer native HTTPS clients so old Python runtimes do not depend on their
    # bundled OpenSSL being new enough to negotiate with GitHub. This matters
    # for legacy toolchain Pythons such as Emscripten's Python 2.7.5.
    if download_with_curl(url, destination):
        return
    if download_with_powershell(url, destination):
        return
    if download_with_python(url, destination):
        return

    raise RuntimeError(
        "failed to download the pinned Slang toolchain; install curl or use a Python runtime with modern TLS support"
    )


def extract_slang_archive(archive, destination):
    if archive.endswith(".zip"):
        with zipfile.ZipFile(archive, "r") as package:
            package.extractall(destination)
        return
    with tarfile.open(archive, "r:gz") as package:
        package.extractall(destination)


def provision_slang(repo_root):
    archive_name = slang_archive()
    cache_root = slang_cache_root(repo_root)
    cache_parent = os.path.dirname(cache_root)
    if not os.path.isdir(cache_parent):
        os.makedirs(cache_parent)

    temporary_root = tempfile.mkdtemp(prefix=".install-", dir=cache_parent)
    try:
        archive_path = os.path.join(temporary_root, archive_name)
        download_file("{}/{}".format(SLANG_RELEASE_BASE_URL, archive_name), archive_path)
        extract_slang_archive(archive_path, temporary_root)
        os.remove(archive_path)

        slangc = find_slangc_under(temporary_root)
        if not is_pinned_slang(slangc):
            raise RuntimeError("downloaded Slang archive did not contain a working Slang {} compiler".format(SLANG_VERSION))

        if os.path.isdir(cache_root):
            shutil.rmtree(cache_root)
        os.rename(temporary_root, cache_root)
        temporary_root = None
        installed = find_slangc_under(cache_root)
        print("Installed Slang {}: {}".format(SLANG_VERSION, installed))
        return installed
    finally:
        if temporary_root and os.path.isdir(temporary_root):
            shutil.rmtree(temporary_root, ignore_errors=True)


def resolve_slangc(repo_root):
    configured = os.environ.get("ARC_SLANGC_EXECUTABLE")
    if configured:
        configured = os.path.abspath(os.path.expanduser(configured))
        if is_pinned_slang(configured):
            return configured
        raise RuntimeError(
            "ARC_SLANGC_EXECUTABLE does not point to the pinned Slang {} compiler: {}".format(
                SLANG_VERSION, configured
            )
        )

    system_slang = find_executable("slangc")
    if is_pinned_slang(system_slang):
        return system_slang

    cached = find_slangc_under(slang_cache_root(repo_root))
    if is_pinned_slang(cached):
        return cached

    print("Slang {} was not found; installing the pinned editor toolchain...".format(SLANG_VERSION))
    return provision_slang(repo_root)


def add_slang_to_environment(environment, slangc):
    environment["ARC_SLANGC_EXECUTABLE"] = slangc
    slang_dir = os.path.dirname(slangc)
    current_path = environment.get("PATH", "")
    path_entries = current_path.split(os.pathsep) if current_path else []
    if slang_dir not in path_entries:
        environment["PATH"] = slang_dir + (os.pathsep + current_path if current_path else "")


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


def run(command, cwd, env=None):
    print("+ " + " ".join(command))
    sys.stdout.flush()
    subprocess.check_call(command, cwd=cwd, env=env)


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
    cmake = find_executable(args.cmake)
    if cmake is None:
        raise RuntimeError("could not find CMake executable '{}'".format(args.cmake))

    # Existing build trees created by older helpers may prohibit FetchContent
    # during CMake's automatic regeneration. Configure once with population
    # enabled so newly pinned dependencies can bootstrap. Subsequent launches
    # retain the setting and go directly to the incremental build.
    if cmake_cache_requires_configure(build_dir, args.vulkan_render):
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
                "-DFETCHCONTENT_FULLY_DISCONNECTED=OFF",
            ],
            repo_root,
            env,
        )

    # Always ask the build system for the host. CMake/MSBuild/Ninja perform an
    # incremental no-op when it is current, while checking timestamps prevents
    # Electron from speaking a newer protocol to a stale executable.
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
        env,
    )
    run(
        [
            cmake,
            "--build",
            build_dir,
            "--config",
            args.config,
            "--target",
            "arc-project-cli",
            "--parallel",
            args.parallel or str(cpu_count()),
        ],
        repo_root,
        env,
    )
    host = find_host_executable(build_dir, args.config)
    project_tool = find_project_tool_executable(build_dir, args.config)

    if host is None:
        raise RuntimeError("arc_host_process was not found after the native build")
    if project_tool is None:
        raise RuntimeError("arc-project was not found after the native build")
    return host, project_tool


def dependencies_ready(editor_dir):
    return os.path.isdir(os.path.join(editor_dir, "node_modules"))


def main():
    args = parse_args()
    repo_root = os.path.dirname(os.path.abspath(__file__))
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

    host = None
    project_tool = None
    tool_env = os.environ.copy()
    if args.perf or args.perf_slow_ms is not None:
        tool_env["ARC_EDITOR_PERF"] = "1"
    if args.perf_slow_ms is not None:
        tool_env["ARC_EDITOR_PERF_SLOW_MS"] = str(args.perf_slow_ms)
    if not args.ui_lab:
        try:
            slangc = resolve_slangc(repo_root)
            add_slang_to_environment(tool_env, slangc)
            host, project_tool = prepare_native_editor(args, repo_root, tool_env)
        except (RuntimeError, OSError, subprocess.CalledProcessError) as error:
            print("error: {}".format(error), file=sys.stderr)
            return 1

    npm = find_executable(args.npm)
    if npm is None:
        print("error: could not find npm executable '{}'".format(args.npm), file=sys.stderr)
        return 1

    try:
        if not args.skip_npm_install and (args.force_build or not dependencies_ready(editor_dir)):
            run([npm, "install"], editor_dir)

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
            run([npm, "run", "typecheck"], editor_dir, editor_env)
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
