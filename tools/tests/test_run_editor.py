from __future__ import annotations

import pathlib
import sys
import unittest
from types import SimpleNamespace
from unittest import mock


REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import run_editor


class EditorArgumentTests(unittest.TestCase):
    def test_no_install_skips_prerequisite_preflight_flag(self) -> None:
        with mock.patch.object(sys, "argv", ["run_editor.py", "--no-install"]):
            args = run_editor.parse_args()

        self.assertTrue(args.no_install)

    def test_no_install_cannot_combine_with_install_prerequisites(self) -> None:
        with mock.patch.object(
            sys,
            "argv",
            ["run_editor.py", "--no-install", "--install-prerequisites"],
        ), self.assertRaises(SystemExit):
            run_editor.parse_args()


class EditorNativeBuildTests(unittest.TestCase):
    def make_args(self, force_build: bool = False) -> SimpleNamespace:
        return SimpleNamespace(
            build_dir="out/build/editor-vulkan",
            vulkan_render=True,
            cmake="cmake",
            force_build=force_build,
            config="Release",
            parallel=None,
        )

    def test_force_build_resets_before_configuring(self) -> None:
        calls = []

        with mock.patch.object(run_editor.arc_build, "find_executable", return_value="cmake"), mock.patch.object(
            run_editor.arc_build,
            "reset_cmake_build_directory",
            side_effect=lambda *_args, **_kwargs: calls.append("reset"),
        ) as reset, mock.patch.object(
            run_editor.arc_build,
            "resolve_visual_studio_generator",
            side_effect=lambda _: calls.append("resolve"),
        ), mock.patch.object(
            run_editor.arc_build, "cmake_cache_generator", return_value=None
        ), mock.patch.object(
            run_editor.arc_build,
            "run",
            side_effect=lambda *_args, **_kwargs: calls.append("configure"),
        ), mock.patch.object(
            run_editor.arc_build,
            "build_cmake_target",
            side_effect=lambda _cmake, _build, target, *_args, **_kwargs: calls.append(target),
        ), mock.patch.object(
            run_editor, "find_host_executable", return_value="arc_host_process"
        ), mock.patch.object(
            run_editor, "find_project_tool_executable", return_value="arc-project"
        ):
            run_editor.prepare_native_editor(self.make_args(force_build=True), str(REPO_ROOT))

        reset.assert_called_once_with(
            str(REPO_ROOT / "out" / "build" / "editor-vulkan"),
            cmake="cmake",
        )
        self.assertEqual(
            calls,
            ["reset", "resolve", "configure", "arc_host_process", "arc-project-cli"],
        )

    def test_generator_platform_mismatch_requires_force_build(self) -> None:
        with mock.patch.object(run_editor.arc_build, "find_executable", return_value="cmake"), mock.patch.object(
            run_editor.arc_build, "resolve_visual_studio_generator", return_value="Visual Studio 17 2022"
        ), mock.patch.object(
            run_editor.arc_build, "cmake_cache_generator", return_value="Visual Studio 17 2022"
        ), mock.patch.object(
            run_editor.arc_build, "cmake_cache_generator_platform", return_value=""
        ):
            with self.assertRaisesRegex(RuntimeError, "generator platform '<default>'"):
                run_editor.prepare_native_editor(self.make_args(), str(REPO_ROOT))

    def test_existing_tree_is_always_reconfigured_before_build(self) -> None:
        calls = []

        with mock.patch.object(run_editor.arc_build, "find_executable", return_value="cmake"), mock.patch.object(
            run_editor.arc_build, "resolve_visual_studio_generator", return_value="Visual Studio 18 2026"
        ), mock.patch.object(
            run_editor.arc_build, "cmake_cache_generator", return_value="Visual Studio 18 2026"
        ), mock.patch.object(
            run_editor.arc_build,
            "run",
            side_effect=lambda command, *_args, **_kwargs: calls.append(("configure", command)),
        ) as configure, mock.patch.object(
            run_editor.arc_build,
            "build_cmake_target",
            side_effect=lambda _cmake, _build, target, *_args, **_kwargs: calls.append(("build", target)),
        ), mock.patch.object(
            run_editor, "find_host_executable", return_value="arc_host_process"
        ), mock.patch.object(
            run_editor, "find_project_tool_executable", return_value="arc-project"
        ):
            run_editor.prepare_native_editor(self.make_args(), str(REPO_ROOT))

        configure.assert_called_once()
        configure_command = configure.call_args.args[0]
        self.assertIn("-G", configure_command)
        self.assertIn("Visual Studio 18 2026", configure_command)
        self.assertEqual(
            [entry[0] for entry in calls],
            ["configure", "build", "build"],
        )


if __name__ == "__main__":
    unittest.main()
