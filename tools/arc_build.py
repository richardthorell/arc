#!/usr/bin/env python
"""Shared ARC build, toolchain, and process helpers."""

from __future__ import print_function

import io
import multiprocessing
import os
import platform
import re
import shutil
import stat
import subprocess
import sys
import tarfile
import tempfile
import time
import zipfile

try:
    from urllib.request import urlopen
except ImportError:
    from urllib2 import urlopen

try:
    text_input = raw_input
except NameError:
    text_input = input


SLANG_VERSION = "2026.14.1"
SLANG_RELEASE_BASE_URL = "https://github.com/shader-slang/slang/releases/download/v{}".format(SLANG_VERSION)
EDITOR_NODE_VERSION = "22.23.3"
EDITOR_NPM_VERSION = "10.9.9"
VISUAL_STUDIO_GENERATORS = {
    18: "Visual Studio 18 2026",
    17: "Visual Studio 17 2022",
}
WINDOWS_CMAKE_PACKAGE = "Kitware.CMake"
WINDOWS_NODE_PACKAGE = "OpenJS.NodeJS.22"
VISUAL_STUDIO_DOWNLOAD_URL = "https://visualstudio.microsoft.com/downloads/"


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


def run(command, cwd, env=None):
    print("+ " + " ".join(command))
    sys.stdout.flush()
    subprocess.check_call(command, cwd=cwd, env=env)


def find_vswhere():
    executable = find_executable("vswhere")
    if executable:
        return executable

    for variable in ("ProgramFiles(x86)", "ProgramFiles"):
        root = os.environ.get(variable)
        if not root:
            continue
        candidate = os.path.join(root, "Microsoft Visual Studio", "Installer", "vswhere.exe")
        if os.path.isfile(candidate):
            return candidate
    return None


def visual_studio_version():
    if platform.system() != "Windows":
        return None

    vswhere = find_vswhere()
    if vswhere is None:
        return None

    try:
        return subprocess.check_output(
            [
                vswhere,
                "-latest",
                "-products",
                "*",
                "-requires",
                "Microsoft.VisualStudio.Component.VC.Tools.x86.x64",
                "-property",
                "installationVersion",
            ],
            stderr=subprocess.STDOUT,
            universal_newlines=True,
        ).strip() or None
    except (OSError, subprocess.CalledProcessError):
        return None


def resolve_visual_studio_generator(cmake):
    if platform.system() != "Windows":
        return None

    version = visual_studio_version()
    if not version:
        raise RuntimeError(
            "Visual Studio with the Desktop development with C++ workload was not found; "
            "install Visual Studio 2026 or 2022 and select the C++ desktop workload"
        )

    try:
        major_version = int(version.split(".", 1)[0])
    except ValueError:
        raise RuntimeError("could not determine the installed Visual Studio version from '{}'".format(version))

    generator = VISUAL_STUDIO_GENERATORS.get(major_version)
    if generator is None:
        raise RuntimeError("unsupported Visual Studio version {}; update ARC build generator mapping".format(version))

    try:
        cmake_help = subprocess.check_output(
            [cmake, "--help"], stderr=subprocess.STDOUT, universal_newlines=True
        )
    except (OSError, subprocess.CalledProcessError) as error:
        raise RuntimeError("failed to query CMake generators: {}".format(error))

    if generator not in cmake_help:
        requirement = (
            "CMake 4.2 or newer"
            if major_version >= 18
            else "a CMake version that supports Visual Studio 2022"
        )
        raise RuntimeError("{} does not support '{}'; install {}".format(cmake, generator, requirement))

    return generator


def command_version(executable, arguments=None):
    arguments = arguments or ["--version"]
    try:
        return subprocess.check_output(
            [executable] + list(arguments),
            stderr=subprocess.STDOUT,
            universal_newlines=True,
        ).strip().splitlines()[0]
    except (OSError, subprocess.CalledProcessError, IndexError):
        return ""


def semantic_version_parts(value):
    match = re.match(r"^v?(\d+)\.(\d+)(?:\.(\d+))?", (value or "").strip())
    if not match:
        return None
    return tuple(int(part) if part is not None else 0 for part in match.groups())


def editor_node_toolchain_supported(node_version, npm_version):
    return (
        semantic_version_parts(node_version) == semantic_version_parts(EDITOR_NODE_VERSION)
        and semantic_version_parts(npm_version) == semantic_version_parts(EDITOR_NPM_VERSION)
    )


def check_editor_prerequisites(cmake="cmake", npm="npm", require_native=True):
    checks = []

    cmake_executable = None
    if require_native:
        cmake_executable = find_executable(cmake)
        checks.append(
            {
                "key": "cmake",
                "name": "CMake",
                "ok": cmake_executable is not None,
                "detail": command_version(cmake_executable) if cmake_executable else "not found on PATH",
                "installable": platform.system() == "Windows",
            }
        )

    node_executable = find_executable("node")
    npm_executable = find_executable(npm)
    node_present = node_executable is not None and npm_executable is not None
    node_ok = False
    node_detail = "not found on PATH"
    if node_present:
        node_version = command_version(node_executable)
        npm_version = command_version(npm_executable)
        node_ok = editor_node_toolchain_supported(node_version, npm_version)
        node_detail = "{} / npm {}".format(node_version, npm_version)
        if not node_ok:
            node_detail += " (requires Node.js {} / npm {})".format(EDITOR_NODE_VERSION, EDITOR_NPM_VERSION)
    checks.append(
        {
            "key": "node",
            "name": "Node.js / npm",
            "ok": node_ok,
            "detail": node_detail,
            "installable": platform.system() == "Windows",
        }
    )

    if require_native and platform.system() == "Windows":
        version = visual_studio_version()
        checks.append(
            {
                "key": "visual_studio",
                "name": "Visual Studio C++",
                "ok": version is not None,
                "detail": version or "Visual Studio 2026/2022 with Desktop development with C++ was not found",
                "installable": False,
            }
        )

        if cmake_executable and version:
            try:
                generator = resolve_visual_studio_generator(cmake_executable)
                checks.append(
                    {
                        "key": "generator",
                        "name": "CMake generator",
                        "ok": True,
                        "detail": generator,
                        "installable": False,
                    }
                )
            except RuntimeError as error:
                checks.append(
                    {
                        "key": "generator",
                        "name": "CMake generator",
                        "ok": False,
                        "detail": str(error),
                        "installable": True,
                    }
                )

    return checks


def prerequisites_ready(checks):
    return all(check["ok"] for check in checks)


def print_prerequisite_report(checks, show_install_hint=True):
    print("ARC editor prerequisites")
    print("")
    for check in checks:
        status = "OK" if check["ok"] else "MISSING"
        print("  {:<18} {:<7} {}".format(check["name"], status, check["detail"]))

    missing = [check for check in checks if not check["ok"]]
    if not missing:
        print("")
        print("Ready to build the ARC editor.")
        return

    print("")
    if any(check["key"] == "visual_studio" for check in missing):
        print("Visual Studio must be installed manually:")
        print("  {}".format(VISUAL_STUDIO_DOWNLOAD_URL))
        print("  Select 'Desktop development with C++' and include the MSVC x64/x86 tools and Windows SDK.")

    if show_install_hint and any(check["installable"] for check in missing):
        print("Run 'python run_editor.py --install-prerequisites' to install supported missing tools.")


def prompt_yes_no(message, input_fn=None):
    input_fn = input_fn or text_input
    try:
        response = input_fn("{} [y/N] ".format(message)).strip().lower()
    except (EOFError, KeyboardInterrupt):
        print("")
        return False
    return response in ("y", "yes")


def prerequisite_install_action(check):
    if check["key"] == "cmake":
        return "install", WINDOWS_CMAKE_PACKAGE, "CMake"
    if check["key"] == "generator":
        return "upgrade", WINDOWS_CMAKE_PACKAGE, "CMake"
    if check["key"] == "node":
        return "install", WINDOWS_NODE_PACKAGE, "Node.js {} / npm {}".format(EDITOR_NODE_VERSION, EDITOR_NPM_VERSION)
    return None


def install_editor_prerequisites(cmake="cmake", npm="npm", require_native=True, input_fn=None):
    checks = check_editor_prerequisites(cmake=cmake, npm=npm, require_native=require_native)
    missing = [check for check in checks if not check["ok"]]
    installable = [check for check in missing if check["installable"]]

    if not installable:
        return checks, False

    if platform.system() != "Windows":
        raise RuntimeError("automatic prerequisite installation is currently supported on Windows only")

    winget = find_executable("winget")
    if winget is None:
        raise RuntimeError(
            "Windows Package Manager (winget) was not found; install CMake and Node.js/npm manually"
        )

    installed = False
    common = [
        "--exact",
        "--accept-package-agreements",
        "--accept-source-agreements",
    ]

    # Ask separately for every component. ARC never silently installs or
    # upgrades development tools on the host.
    handled_packages = set()
    for check in installable:
        action = prerequisite_install_action(check)
        if action is None:
            continue
        verb, package, label = action
        package_key = (verb, package)
        if package_key in handled_packages:
            continue
        handled_packages.add(package_key)

        prompt = "{} {} using Windows Package Manager?".format(
            "Upgrade" if verb == "upgrade" else "Install",
            label,
        )
        if not prompt_yes_no(prompt, input_fn=input_fn):
            print("Skipped {}.".format(label))
            continue

        command = [winget, verb, "--id", package]
        if package == WINDOWS_NODE_PACKAGE:
            command.extend(["--version", EDITOR_NODE_VERSION])
        run(command + common, os.getcwd())
        installed = True

    return checks, installed


def cmake_cache_value(build_dir, key):
    cache = os.path.join(build_dir, "CMakeCache.txt")
    if not os.path.isfile(cache):
        return None
    prefix = "{}:INTERNAL=".format(key)
    try:
        with io.open(cache, "r", encoding="utf-8", errors="replace") as handle:
            for line in handle:
                if line.startswith(prefix):
                    return line.split("=", 1)[1].strip()
    except IOError:
        return None
    return None


def cmake_cache_generator(build_dir):
    return cmake_cache_value(build_dir, "CMAKE_GENERATOR")


def cmake_cache_generator_platform(build_dir):
    return cmake_cache_value(build_dir, "CMAKE_GENERATOR_PLATFORM")


def remove_readonly_path(function, path, _exc_info):
    # Git can leave pack/index files read-only on Windows, and scanners can
    # briefly keep them open after a failed configure. Clear the flag and retry
    # a few times so a clean build does not require manual cleanup.
    os.chmod(path, stat.S_IREAD | stat.S_IWRITE)
    for attempt in range(5):
        try:
            function(path)
            return
        except OSError:
            if attempt == 4:
                raise
            time.sleep(0.1 * (attempt + 1))


def reset_cmake_build_directory(build_dir, cmake=None):
    if not os.path.exists(build_dir):
        return
    if not os.path.isdir(build_dir):
        raise RuntimeError("CMake build path is not a directory: {}".format(build_dir))
    print("Removing CMake build directory: {}".format(build_dir))

    # CMake owns the generated build tree and its FetchContent checkouts. On
    # Windows, let CMake remove that tree first so Python does not have to walk
    # dependency filenames that may be awkward for the Win32 path parser.
    if platform.system() == "Windows" and cmake:
        try:
            subprocess.check_call([cmake, "-E", "rm", "-rf", build_dir])
            if not os.path.exists(build_dir):
                return
        except (OSError, subprocess.CalledProcessError):
            pass

    shutil.rmtree(build_dir, onerror=remove_readonly_path)


def build_cmake_target(cmake, build_dir, target, config, cwd, env=None, parallel=None):
    run(
        [
            cmake,
            "--build",
            build_dir,
            "--config",
            config,
            "--target",
            target,
            "--parallel",
            parallel or str(cpu_count()),
        ],
        cwd,
        env,
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
            raise RuntimeError(
                "downloaded Slang archive did not contain a working Slang {} compiler".format(SLANG_VERSION)
            )

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
