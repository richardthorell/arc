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
        self.assertFalse(result.android)

    def test_editor_web_change_is_editor_only(self):
        result = classify(["editor/src/renderer/App.tsx"])
        self.assertTrue(result.quality_editor)
        self.assertFalse(result.clang)
        self.assertFalse(result.gcc)
        self.assertFalse(result.msvc)
        self.assertFalse(result.coverage)
        self.assertFalse(result.android)

    def test_editor_package_impact_selects_windows_smoke(self):
        for path in (
            "editor/forge.config.ts",
            "editor/package.json",
            "editor/package-lock.json",
            "editor/src/main/index.ts",
            "editor/src/preload/index.ts",
            "editor/native/src/host.cpp",
            "editor/resources/ai-skills/play-workflows/SKILL.md",
            ".github/workflows/build-msvc.yml",
            "tools/classify_ci_changes.py",
        ):
            with self.subTest(path=path):
                selection = classify([path])
                self.assertTrue(selection.editor_package)
                self.assertTrue(selection.msvc)

    def test_renderer_only_does_not_package_editor(self):
        selection = classify(["editor/src/renderer/src/app/Workbench.tsx"])
        self.assertTrue(selection.quality_editor)
        self.assertFalse(selection.editor_package)
        self.assertFalse(selection.msvc)

    def test_android_apk_verifier_change_runs_android(self):
        selection = classify(["tools/verify_android_apk.py"])
        self.assertTrue(selection.android)
        self.assertTrue(selection.quality_native)
        self.assertFalse(selection.editor_package)

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
        self.assertTrue(result.android)

    def test_template_change_runs_cook_smoke_without_gcc_or_coverage(self):
        result = classify(["templates/blank-3d/project.arcproject"])
        self.assertTrue(result.clang)
        self.assertTrue(result.msvc)
        self.assertTrue(result.cooker)
        self.assertFalse(result.gcc)
        self.assertFalse(result.coverage)

    def test_distribution_pipeline_changes_trigger_shipping_acceptance(self):
        for path in (
            "tools/project_cli/package_command.cpp",
            "engine/project/src/common/distribution_assembler.cpp",
            "engine/project/inc/arc/project/distribution_zip.h",
            "engine/project/src/common/windows_distribution_finalizer.cpp",
            "engine/platform/windows/src/windows_entry.cpp",
            "tools/d3_shipping_acceptance.py",
        ):
            with self.subTest(path=path):
                selection = classify([path])
                self.assertTrue(selection.msvc)
                self.assertTrue(selection.cooker)

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
        self.assertTrue(result.android)

    def test_android_host_change_requires_ndk_build(self):
        result = classify(["engine/platform/android/app/src/main/cpp/android_host.cpp"])
        self.assertTrue(result.android)
        self.assertTrue(result.clang)
        self.assertTrue(result.msvc)

    def test_android_scene_and_runtime_changes_require_ndk_build(self):
        for path in ("engine/scene/src/common/runtime_scene.cpp",
                     "engine/assets/src/common/runtime_content.cpp",
                     "engine/project/src/common/runtime_catalog.cpp"):
            with self.subTest(path=path):
                self.assertTrue(classify([path]).android)

    def test_unrelated_native_change_does_not_require_android(self):
        result = classify(["engine/network/src/common/transport.cpp"])
        self.assertFalse(result.android)

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
