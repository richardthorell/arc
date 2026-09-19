from __future__ import annotations

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


class EditorBuildCacheTests(unittest.TestCase):
    def test_force_build_resets_native_build_tree_before_generator_check(self) -> None:
        args = SimpleNamespace(
            build_dir="out/build/editor-vulkan",
            vulkan_render=True,
            cmake="cmake",
            force_build=True,
            config="Release",
            parallel=None,
        )

        with mock.patch.object(run_editor.arc_build, "find_executable", return_value="cmake"), mock.patch.object(
            run_editor.arc_build, "reset_cmake_build_directory"
        ) as reset, mock.patch.object(
            run_editor.arc_build, "resolve_visual_studio_generator", return_value=None
        ) as resolve_generator, mock.patch.object(
            run_editor.arc_build, "cmake_cache_generator", return_value=None
        ), mock.patch.object(
            run_editor, "cmake_cache_requires_configure", return_value=False
        ), mock.patch.object(
            run_editor.arc_build, "build_cmake_target"
        ), mock.patch.object(
            run_editor, "find_host_executable", return_value="arc_host_process"
        ), mock.patch.object(
            run_editor, "find_project_tool_executable", return_value="arc-project"
        ):
            run_editor.prepare_native_editor(args, str(REPO_ROOT))

        reset.assert_called_once_with(str(REPO_ROOT / "out" / "build" / "editor-vulkan"))
        resolve_generator.assert_called_once_with("cmake")

    def test_missing_cache_requires_configure(self) -> None:
        with tempfile.TemporaryDirectory() as build_dir:
            self.assertTrue(run_editor.cmake_cache_requires_configure(build_dir, True))

    def test_matching_editor_cache_skips_configure(self) -> None:
        with tempfile.TemporaryDirectory() as build_dir:
            cache = pathlib.Path(build_dir) / "CMakeCache.txt"
            cache.write_text(
                "\n".join(
                    [
                        "ARC_BUILD_EDITOR:BOOL=ON",
                        "ARC_BUILD_RENDER_VULKAN:BOOL=ON",
                        "FETCHCONTENT_FULLY_DISCONNECTED:BOOL=OFF",
                    ]
                ),
                encoding="utf-8",
            )
            self.assertFalse(run_editor.cmake_cache_requires_configure(build_dir, True))

    def test_vulkan_change_requires_reconfigure(self) -> None:
        with tempfile.TemporaryDirectory() as build_dir:
            cache = pathlib.Path(build_dir) / "CMakeCache.txt"
            cache.write_text(
                "\n".join(
                    [
                        "ARC_BUILD_EDITOR:BOOL=ON",
                        "ARC_BUILD_RENDER_VULKAN:BOOL=OFF",
                        "FETCHCONTENT_FULLY_DISCONNECTED:BOOL=OFF",
                    ]
                ),
                encoding="utf-8",
            )
            self.assertTrue(run_editor.cmake_cache_requires_configure(build_dir, True))


if __name__ == "__main__":
    unittest.main()
