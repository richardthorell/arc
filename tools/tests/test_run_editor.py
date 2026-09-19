from __future__ import annotations

import pathlib
import sys
import tempfile
import unittest


REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import run_editor


class EditorBuildCacheTests(unittest.TestCase):
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
