from __future__ import annotations

import io
import os
import pathlib
import sys
import tempfile
import unittest
from unittest import mock


REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools import arc_build


class ArcBuildTests(unittest.TestCase):
    def test_prompt_yes_no_requires_explicit_yes(self) -> None:
        self.assertTrue(arc_build.prompt_yes_no("Install CMake?", input_fn=lambda _: "yes"))
        self.assertFalse(arc_build.prompt_yes_no("Install CMake?", input_fn=lambda _: ""))
        self.assertFalse(arc_build.prompt_yes_no("Install CMake?", input_fn=lambda _: "no"))

    def test_prerequisite_installer_prompts_for_each_component(self) -> None:
        checks = [
            {
                "key": "cmake",
                "name": "CMake",
                "ok": False,
                "detail": "not found",
                "installable": True,
            },
            {
                "key": "node",
                "name": "Node.js / npm",
                "ok": False,
                "detail": "not found",
                "installable": True,
            },
            {
                "key": "visual_studio",
                "name": "Visual Studio C++",
                "ok": False,
                "detail": "not found",
                "installable": False,
            },
        ]
        answers = iter(["yes", "no"])

        with mock.patch.object(arc_build.platform, "system", return_value="Windows"), mock.patch.object(
            arc_build, "check_editor_prerequisites", return_value=checks
        ), mock.patch.object(
            arc_build, "find_executable", return_value="winget.exe"
        ), mock.patch.object(arc_build, "run") as run:
            returned_checks, installed = arc_build.install_editor_prerequisites(
                input_fn=lambda _: next(answers)
            )

        self.assertEqual(returned_checks, checks)
        self.assertTrue(installed)
        run.assert_called_once()
        command = run.call_args.args[0]
        self.assertEqual(command[:4], ["winget.exe", "install", "--id", arc_build.WINDOWS_CMAKE_PACKAGE])

    def test_prerequisite_installer_does_nothing_when_all_installs_declined(self) -> None:
        checks = [
            {
                "key": "cmake",
                "name": "CMake",
                "ok": False,
                "detail": "not found",
                "installable": True,
            },
            {
                "key": "node",
                "name": "Node.js / npm",
                "ok": False,
                "detail": "not found",
                "installable": True,
            },
        ]

        with mock.patch.object(arc_build.platform, "system", return_value="Windows"), mock.patch.object(
            arc_build, "check_editor_prerequisites", return_value=checks
        ), mock.patch.object(
            arc_build, "find_executable", return_value="winget.exe"
        ), mock.patch.object(arc_build, "run") as run:
            _, installed = arc_build.install_editor_prerequisites(input_fn=lambda _: "no")

        self.assertFalse(installed)
        run.assert_not_called()

    def test_remove_readonly_path_clears_flag_and_retries(self) -> None:
        retry = mock.Mock()

        with mock.patch.object(arc_build.os, "chmod") as chmod:
            arc_build.remove_readonly_path(retry, "locked.idx", None)

        chmod.assert_called_once_with(
            "locked.idx",
            arc_build.stat.S_IREAD | arc_build.stat.S_IWRITE,
        )
        retry.assert_called_once_with("locked.idx")

    def test_cmake_cache_generator_platform_reads_cached_platform(self) -> None:
        with tempfile.TemporaryDirectory() as build_dir:
            cache = pathlib.Path(build_dir) / "CMakeCache.txt"
            cache.write_text(
                "CMAKE_GENERATOR:INTERNAL=Visual Studio 17 2022\n"
                "CMAKE_GENERATOR_PLATFORM:INTERNAL=Win32\n",
                encoding="utf-8",
            )

            self.assertEqual(
                arc_build.cmake_cache_generator_platform(build_dir),
                "Win32",
            )

    def test_windows_reset_prefers_cmake_owned_removal(self) -> None:
        with mock.patch.object(arc_build.platform, "system", return_value="Windows"), mock.patch.object(
            arc_build.os.path, "exists", side_effect=[True, False]
        ), mock.patch.object(
            arc_build.os.path, "isdir", return_value=True
        ), mock.patch.object(
            arc_build.subprocess, "check_call"
        ) as check_call, mock.patch.object(
            arc_build.shutil, "rmtree"
        ) as rmtree:
            arc_build.reset_cmake_build_directory("C:\\arc\\out\\build", cmake="cmake.exe")

        check_call.assert_called_once_with(
            ["cmake.exe", "-E", "rm", "-rf", "C:\\arc\\out\\build"]
        )
        rmtree.assert_not_called()

    def test_reset_cmake_build_directory_removes_existing_tree(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_root:
            build_dir = pathlib.Path(temporary_root) / "build"
            build_dir.mkdir()
            (build_dir / "CMakeCache.txt").write_text("stale", encoding="utf-8")

            arc_build.reset_cmake_build_directory(str(build_dir))

            self.assertFalse(build_dir.exists())

    def test_selects_windows_pinned_archive(self) -> None:
        with mock.patch.object(arc_build.platform, "system", return_value="Windows"), mock.patch.object(
            arc_build.platform, "machine", return_value="AMD64"
        ):
            self.assertEqual(
                arc_build.slang_archive(),
                "slang-2026.14.1-windows-x86_64.zip",
            )

    def test_selects_linux_pinned_archive(self) -> None:
        with mock.patch.object(arc_build.platform, "system", return_value="Linux"), mock.patch.object(
            arc_build.platform, "machine", return_value="x86_64"
        ):
            self.assertEqual(
                arc_build.slang_archive(),
                "slang-2026.14.1-linux-x86_64.tar.gz",
            )

    def test_resolves_visual_studio_2026_generator(self) -> None:
        with mock.patch.object(arc_build.platform, "system", return_value="Windows"), mock.patch.object(
            arc_build, "find_vswhere", return_value="vswhere.exe"
        ), mock.patch.object(
            arc_build.subprocess,
            "check_output",
            side_effect=["18.0.0", "Generators\n  Visual Studio 18 2026"],
        ):
            self.assertEqual(
                arc_build.resolve_visual_studio_generator("cmake.exe"),
                "Visual Studio 18 2026",
            )

    def test_visual_studio_2026_requires_supported_cmake(self) -> None:
        with mock.patch.object(arc_build.platform, "system", return_value="Windows"), mock.patch.object(
            arc_build, "find_vswhere", return_value="vswhere.exe"
        ), mock.patch.object(
            arc_build.subprocess,
            "check_output",
            side_effect=["18.0.0", "Generators\n  Ninja"],
        ):
            with self.assertRaisesRegex(RuntimeError, "CMake 4.2 or newer"):
                arc_build.resolve_visual_studio_generator("cmake.exe")

    def test_rejects_invalid_explicit_slang_override(self) -> None:
        with mock.patch.dict(os.environ, {"ARC_SLANGC_EXECUTABLE": "/missing/slangc"}, clear=True), mock.patch.object(
            arc_build, "provision_slang"
        ) as provision:
            with self.assertRaisesRegex(RuntimeError, "ARC_SLANGC_EXECUTABLE"):
                arc_build.resolve_slangc(str(REPO_ROOT))
            provision.assert_not_called()

    def test_installs_pinned_slang_when_no_valid_compiler_exists(self) -> None:
        installed = os.path.join(str(REPO_ROOT), "out", "toolchains", "slangc")
        with mock.patch.dict(os.environ, {}, clear=True), mock.patch.object(
            arc_build, "find_executable", return_value=None
        ), mock.patch.object(arc_build, "find_slangc_under", return_value=None), mock.patch.object(
            arc_build, "provision_slang", return_value=installed
        ) as provision:
            self.assertEqual(arc_build.resolve_slangc(str(REPO_ROOT)), installed)
            provision.assert_called_once_with(str(REPO_ROOT))

    def test_reuses_cached_pinned_slang(self) -> None:
        cached = os.path.join(str(REPO_ROOT), "out", "toolchains", "slangc")
        with mock.patch.dict(os.environ, {}, clear=True), mock.patch.object(
            arc_build, "find_executable", return_value=None
        ), mock.patch.object(arc_build, "find_slangc_under", return_value=cached), mock.patch.object(
            arc_build, "is_pinned_slang", side_effect=lambda executable: executable == cached
        ), mock.patch.object(arc_build, "provision_slang") as provision:
            self.assertEqual(arc_build.resolve_slangc(str(REPO_ROOT)), cached)
            provision.assert_not_called()

    def test_download_file_prefers_curl_over_python_tls(self) -> None:
        descriptor, destination = tempfile.mkstemp()
        os.close(descriptor)
        self.addCleanup(lambda: os.path.exists(destination) and os.remove(destination))

        with mock.patch.object(arc_build, "find_executable", return_value="curl"), mock.patch.object(
            arc_build.subprocess, "check_call"
        ) as check_call, mock.patch.object(arc_build, "urlopen") as opener:
            arc_build.download_file("https://example.invalid/slang.zip", destination)

        check_call.assert_called_once_with(
            [
                "curl",
                "--fail",
                "--location",
                "--retry",
                "3",
                "--output",
                destination,
                "https://example.invalid/slang.zip",
            ]
        )
        opener.assert_not_called()

    def test_download_file_falls_back_to_compatibility_url_opener(self) -> None:
        response = io.BytesIO(b"slang archive")
        descriptor, destination = tempfile.mkstemp()
        os.close(descriptor)
        self.addCleanup(lambda: os.path.exists(destination) and os.remove(destination))

        with mock.patch.object(arc_build, "find_executable", return_value=None), mock.patch.object(
            arc_build.platform, "system", return_value="Linux"
        ), mock.patch.object(arc_build, "urlopen", return_value=response) as opener:
            arc_build.download_file("https://example.invalid/slang.zip", destination)

        opener.assert_called_once_with("https://example.invalid/slang.zip")
        with open(destination, "rb") as downloaded:
            self.assertEqual(downloaded.read(), b"slang archive")
        self.assertTrue(response.closed)

    def test_windows_download_falls_back_to_powershell_with_tls12(self) -> None:
        descriptor, destination = tempfile.mkstemp()
        os.close(descriptor)
        self.addCleanup(lambda: os.path.exists(destination) and os.remove(destination))

        def executable(name):
            return "powershell" if name == "powershell" else None

        with mock.patch.object(arc_build.platform, "system", return_value="Windows"), mock.patch.object(
            arc_build, "find_executable", side_effect=executable
        ), mock.patch.object(arc_build.subprocess, "check_call") as check_call, mock.patch.object(
            arc_build, "urlopen"
        ) as opener:
            arc_build.download_file("https://example.invalid/slang.zip", destination)

        command = check_call.call_args.args[0]
        self.assertEqual(command[0], "powershell")
        self.assertIn("Tls12", command[-1])
        self.assertIn("Invoke-WebRequest", command[-1])
        opener.assert_not_called()

    def test_exposes_slang_to_child_processes(self) -> None:
        environment = {"PATH": os.pathsep.join(["existing", "tools"])}
        slangc = os.path.join("toolchains", "slang", "bin", "slangc.exe")

        arc_build.add_slang_to_environment(environment, slangc)

        self.assertEqual(environment["ARC_SLANGC_EXECUTABLE"], slangc)
        self.assertEqual(environment["PATH"].split(os.pathsep)[0], os.path.dirname(slangc))


if __name__ == "__main__":
    unittest.main()
