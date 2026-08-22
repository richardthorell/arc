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

import run_editor


class EditorSlangBootstrapTests(unittest.TestCase):
    def test_selects_windows_pinned_archive(self) -> None:
        with mock.patch.object(run_editor.platform, "system", return_value="Windows"), mock.patch.object(
            run_editor.platform, "machine", return_value="AMD64"
        ):
            self.assertEqual(
                run_editor.slang_archive(),
                "slang-2026.14.1-windows-x86_64.zip",
            )

    def test_selects_linux_pinned_archive(self) -> None:
        with mock.patch.object(run_editor.platform, "system", return_value="Linux"), mock.patch.object(
            run_editor.platform, "machine", return_value="x86_64"
        ):
            self.assertEqual(
                run_editor.slang_archive(),
                "slang-2026.14.1-linux-x86_64.tar.gz",
            )

    def test_rejects_invalid_explicit_slang_override(self) -> None:
        with mock.patch.dict(os.environ, {"ARC_SLANGC_EXECUTABLE": "/missing/slangc"}, clear=True), mock.patch.object(
            run_editor, "provision_slang"
        ) as provision:
            with self.assertRaisesRegex(RuntimeError, "ARC_SLANGC_EXECUTABLE"):
                run_editor.resolve_slangc(str(REPO_ROOT))
            provision.assert_not_called()

    def test_installs_pinned_slang_when_no_valid_compiler_exists(self) -> None:
        installed = os.path.join(str(REPO_ROOT), "out", "toolchains", "slangc")
        with mock.patch.dict(os.environ, {}, clear=True), mock.patch.object(
            run_editor, "find_executable", return_value=None
        ), mock.patch.object(run_editor, "find_slangc_under", return_value=None), mock.patch.object(
            run_editor, "provision_slang", return_value=installed
        ) as provision:
            self.assertEqual(run_editor.resolve_slangc(str(REPO_ROOT)), installed)
            provision.assert_called_once_with(str(REPO_ROOT))

    def test_reuses_cached_pinned_slang(self) -> None:
        cached = os.path.join(str(REPO_ROOT), "out", "toolchains", "slangc")
        with mock.patch.dict(os.environ, {}, clear=True), mock.patch.object(
            run_editor, "find_executable", return_value=None
        ), mock.patch.object(run_editor, "find_slangc_under", return_value=cached), mock.patch.object(
            run_editor, "is_pinned_slang", side_effect=lambda executable: executable == cached
        ), mock.patch.object(run_editor, "provision_slang") as provision:
            self.assertEqual(run_editor.resolve_slangc(str(REPO_ROOT)), cached)
            provision.assert_not_called()

    def test_download_file_prefers_curl_over_python_tls(self) -> None:
        descriptor, destination = tempfile.mkstemp()
        os.close(descriptor)
        self.addCleanup(lambda: os.path.exists(destination) and os.remove(destination))

        with mock.patch.object(run_editor, "find_executable", return_value="curl"), mock.patch.object(
            run_editor.subprocess, "check_call"
        ) as check_call, mock.patch.object(run_editor, "urlopen") as opener:
            run_editor.download_file("https://example.invalid/slang.zip", destination)

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

        with mock.patch.object(run_editor, "find_executable", return_value=None), mock.patch.object(
            run_editor.platform, "system", return_value="Linux"
        ), mock.patch.object(run_editor, "urlopen", return_value=response) as opener:
            run_editor.download_file("https://example.invalid/slang.zip", destination)

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

        with mock.patch.object(run_editor.platform, "system", return_value="Windows"), mock.patch.object(
            run_editor, "find_executable", side_effect=executable
        ), mock.patch.object(run_editor.subprocess, "check_call") as check_call, mock.patch.object(
            run_editor, "urlopen"
        ) as opener:
            run_editor.download_file("https://example.invalid/slang.zip", destination)

        command = check_call.call_args.args[0]
        self.assertEqual(command[0], "powershell")
        self.assertIn("Tls12", command[-1])
        self.assertIn("Invoke-WebRequest", command[-1])
        opener.assert_not_called()

    def test_exposes_slang_to_editor_and_native_child_processes(self) -> None:
        environment = {"PATH": os.pathsep.join(["existing", "tools"])}
        slangc = os.path.join("toolchains", "slang", "bin", "slangc.exe")

        run_editor.add_slang_to_environment(environment, slangc)

        self.assertEqual(environment["ARC_SLANGC_EXECUTABLE"], slangc)
        self.assertEqual(environment["PATH"].split(os.pathsep)[0], os.path.dirname(slangc))


if __name__ == "__main__":
    unittest.main()
