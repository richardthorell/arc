import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from classify_ci_changes import classify


class ClassifyCiChangesTests(unittest.TestCase):
    def test_docs_only_skips_expensive_checks(self):
        result = classify(["README.md", "docs/architecture.md"])
        self.assertTrue(result.docs_only)
        self.assertFalse(result.clang)
        self.assertFalse(result.msvc)
        self.assertFalse(result.quality_editor)

    def test_editor_web_change_is_editor_only(self):
        result = classify(["editor/src/renderer/App.tsx"])
        self.assertTrue(result.quality_editor)
        self.assertFalse(result.clang)
        self.assertFalse(result.gcc)
        self.assertFalse(result.msvc)
        self.assertFalse(result.coverage)

    def test_native_change_runs_compilers_and_coverage(self):
        result = classify(["engine/core/src/JobSystem.cpp"])
        self.assertTrue(result.clang)
        self.assertTrue(result.gcc)
        self.assertTrue(result.msvc)
        self.assertTrue(result.coverage)
        self.assertTrue(result.quality_native)
        self.assertFalse(result.shaders)

    def test_render_change_enables_shader_validation(self):
        result = classify(["engine/render-vulkan/src/Renderer.cpp"])
        self.assertTrue(result.clang)
        self.assertTrue(result.gcc)
        self.assertTrue(result.msvc)
        self.assertTrue(result.coverage)
        self.assertTrue(result.shaders)

    def test_template_change_runs_cook_smoke_without_gcc_or_coverage(self):
        result = classify(["templates/blank-3d/project.arcproject"])
        self.assertTrue(result.clang)
        self.assertTrue(result.msvc)
        self.assertTrue(result.cooker)
        self.assertFalse(result.gcc)
        self.assertFalse(result.coverage)

    def test_workflow_change_uses_full_safety_matrix(self):
        result = classify([".github/workflows/pr-gate.yml"])
        self.assertTrue(result.clang)
        self.assertTrue(result.gcc)
        self.assertTrue(result.msvc)
        self.assertTrue(result.coverage)
        self.assertTrue(result.quality_native)
        self.assertTrue(result.quality_editor)
        self.assertTrue(result.shaders)
        self.assertTrue(result.cooker)

    def test_unknown_area_falls_back_to_full_matrix(self):
        result = classify(["new-subsystem/config.custom"])
        self.assertTrue(result.clang)
        self.assertTrue(result.gcc)
        self.assertTrue(result.msvc)
        self.assertTrue(result.coverage)
        self.assertTrue(result.shaders)
        self.assertTrue(result.cooker)


if __name__ == "__main__":
    unittest.main()
