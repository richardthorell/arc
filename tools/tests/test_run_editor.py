from __future__ import annotations

import json
import os
import pathlib
import sys
import tempfile
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


class EditorDependencyTests(unittest.TestCase):
    def create_editor_tree(self, root: pathlib.Path, missing: str | None = None) -> pathlib.Path:
        editor = root / "editor"
        node_modules = editor / "node_modules"
        node_modules.mkdir(parents=True)
        manifest = {
            "dependencies": {
                "react": "19.2.7",
                "react-icons": "5.5.0",
                "@scope/example": "1.0.0",
            },
            "devDependencies": {
                "vite": "8.1.3",
            },
        }
        (editor / "package.json").write_text(json.dumps(manifest), encoding="utf-8")
        (editor / "package-lock.json").write_text('{"lockfileVersion": 3}', encoding="utf-8")
        (node_modules / ".package-lock.json").write_text('{"lockfileVersion": 3}', encoding="utf-8")

        for dependency in ("react", "react-icons", "@scope/example", "vite"):
            if dependency == missing:
                continue
            package_dir = node_modules.joinpath(*dependency.split("/"))
            package_dir.mkdir(parents=True, exist_ok=True)
            (package_dir / "package.json").write_text("{}", encoding="utf-8")

        return editor

    def test_dependencies_ready_requires_all_declared_direct_packages(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            editor = self.create_editor_tree(pathlib.Path(temporary), missing="react-icons")

            self.assertFalse(run_editor.dependencies_ready(str(editor)))

    def test_dependencies_ready_accepts_complete_current_tree(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            editor = self.create_editor_tree(pathlib.Path(temporary))
            source_time = max(
                os.path.getmtime(editor / "package.json"),
                os.path.getmtime(editor / "package-lock.json"),
            )
            os.utime(editor / "node_modules" / ".package-lock.json", (source_time + 1, source_time + 1))

            self.assertTrue(run_editor.dependencies_ready(str(editor)))

    def test_dependencies_ready_rejects_stale_hidden_lockfile(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            editor = self.create_editor_tree(pathlib.Path(temporary))
            os.utime(editor / "node_modules" / ".package-lock.json", (1, 1))

            self.assertFalse(run_editor.dependencies_ready(str(editor)))

    def test_install_editor_dependencies_uses_npm_ci(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            editor = pathlib.Path(temporary)
            (editor / "package-lock.json").write_text('{"lockfileVersion": 3}', encoding="utf-8")

            with mock.patch.object(run_editor.arc_build, "run") as run, mock.patch.object(
                run_editor, "dependencies_ready", return_value=True
            ):
                run_editor.install_editor_dependencies("npm", str(editor))

            run.assert_called_once_with(["npm", "ci"], str(editor))


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
