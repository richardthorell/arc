#!/usr/bin/env python3
"""Classify changed repository paths into the CI checks a pull request needs."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from pathlib import PurePosixPath
import sys


CPP_SUFFIXES = {".c", ".cc", ".cpp", ".cxx", ".h", ".hh", ".hpp", ".hxx", ".inl", ".ixx", ".cppm"}
SHADER_SUFFIXES = {".vert", ".frag", ".comp", ".geom", ".tesc", ".tese", ".glsl", ".hlsl"}
DOC_NAMES = {
    "LICENSE",
    "LICENSE.md",
    "CODE_OF_CONDUCT.md",
    "CONTRIBUTING.md",
    "SECURITY.md",
}


@dataclass
class CiSelection:
    clang: bool = False
    gcc: bool = False
    msvc: bool = False
    android: bool = False
    coverage: bool = False
    quality_native: bool = False
    quality_editor: bool = False
    shaders: bool = False
    cooker: bool = False
    editor_package: bool = False
    docs_only: bool = False

    def enable_full_matrix(self) -> None:
        self.clang = True
        self.gcc = True
        self.msvc = True
        self.android = True
        self.coverage = True
        self.quality_native = True
        self.quality_editor = True
        self.shaders = True
        self.cooker = True


def _normalise(path: str) -> str:
    path = path.strip().replace("\\", "/")
    while path.startswith("./"):
        path = path[2:]
    return path


def _is_docs(path: str) -> bool:
    # Markdown packaged into the editor is an executable distribution input,
    # not documentation-only PR content.
    if _needs_editor_package(path):
        return False
    p = PurePosixPath(path)
    return (
        path.startswith("docs/")
        or p.name in DOC_NAMES
        or p.suffix.lower() in {".md", ".mdx", ".rst"}
    )


def _is_build_or_ci(path: str) -> bool:
    p = PurePosixPath(path)
    return (
        path.startswith(".github/")
        or path.startswith("cmake/")
        or path.startswith("third_party/")
        or path.startswith("external/")
        or p.name == "CMakeLists.txt"
        or p.name in {
            "CMakePresets.json",
            "CMakeUserPresets.json",
            "vcpkg.json",
            "vcpkg-configuration.json",
            ".clang-format",
            ".clang-tidy",
        }
    )


def _is_editor_web(path: str) -> bool:
    return path.startswith("editor/") and not path.startswith("editor/native/") and PurePosixPath(path).name != "CMakeLists.txt"


def _needs_editor_package(path: str) -> bool:
    """Select Windows Electron packaging for changes affecting the packaged app.

    Ordinary renderer/UI changes already get editor lint/typecheck/tests and
    should not rebuild the native host + installer on every PR.
    """
    return (
        path.startswith(("editor/src/main/", "editor/src/preload/", "editor/native/", "editor/resources/"))
        or path in {
            "editor/forge.config.ts",
            "editor/package.json",
            "editor/package-lock.json",
            "editor/vite.main.config.ts",
            "editor/vite.preload.config.ts",
            "editor/CMakeLists.txt",
            ".github/workflows/build-msvc.yml",
            ".github/workflows/pr-gate.yml",
            ".github/actions/setup-windows-toolchain/action.yml",
            "tools/classify_ci_changes.py",
        }
    )


def _is_native(path: str) -> bool:
    p = PurePosixPath(path)
    return (
        p.suffix.lower() in CPP_SUFFIXES
        or path.startswith("engine/")
        or path.startswith("editor/native/")
        or path.startswith("samples/")
        or path.startswith("benchmarks/")
    )


def _is_shader_related(path: str) -> bool:
    p = PurePosixPath(path)
    return (
        p.suffix.lower() in SHADER_SUFFIXES
        or "/shaders/" in f"/{path}"
        or path.startswith("engine/render/")
        or path.startswith("engine/render-vulkan/")
    )


def _is_cooker_related(path: str) -> bool:
    return (
        path.startswith("templates/")
        or path.startswith("engine/assets/")
        or path.startswith("tools/asset_cooker/")
        or path.startswith("tools/project_cli/")
        or path in {
            "tools/d3_shipping_acceptance.py",
            "tools/d3_shipping_negative.py",
            "tools/tests/test_d3_shipping_negative.py",
            "tools/verify_relocated_package.py",
            "engine/platform/windows/src/windows_entry.cpp",
        }
        or path.startswith("engine/project/src/common/distribution_")
        or path.startswith("engine/project/inc/arc/project/distribution_")
        or path.startswith("engine/project/src/common/windows_distribution")
        or path.startswith("engine/project/inc/arc/project/windows_distribution")
    )


def classify(paths: list[str]) -> CiSelection:
    paths = [p for raw in paths if (p := _normalise(raw))]
    result = CiSelection()

    if paths and all(_is_docs(path) for path in paths):
        result.docs_only = True
        return result

    recognised = False
    for path in paths:
        if _needs_editor_package(path):
            result.editor_package = True
            result.msvc = True
            recognised = True
        if path in {"tools/verify_android_apk.py", "tools/tests/test_verify_android_apk.py"}:
            result.android = True
            result.quality_native = True
            recognised = True
        if _is_build_or_ci(path):
            result.enable_full_matrix()
            recognised = True
            continue

        editor_web = _is_editor_web(path)
        native = _is_native(path)
        shader = _is_shader_related(path)
        cooker = _is_cooker_related(path)

        if editor_web:
            result.quality_editor = True
            recognised = True

        # Android reuses these modules in its standalone NDK build.
        # Editor-only changes do not need an Android APK build.
        if path.startswith((
            "engine/platform/android/",
            "engine/render/",
            "engine/render-vulkan/",
            "engine/scene/",
            "engine/assets/",
            "engine/project/",
            "engine/ecs/",
        )):
            result.android = True

        if native:
            result.clang = True
            result.gcc = True
            result.msvc = True
            result.coverage = True
            result.quality_native = True
            recognised = True

        if shader:
            result.clang = True
            result.gcc = True
            result.msvc = True
            result.shaders = True
            result.quality_native = True
            recognised = True

        if cooker:
            # Template/cooker changes need representative Linux/Windows builds and
            # the Windows cook smoke, but do not need the full GCC/coverage matrix
            # unless the changed file is native source (handled above).
            result.clang = True
            result.msvc = True
            result.cooker = True
            result.quality_native = True
            recognised = True

        if path.startswith("tools/") and PurePosixPath(path).suffix.lower() == ".py":
            result.quality_native = True
            recognised = True

    # Safety first: a new top-level area should get the complete PR matrix until
    # the classifier explicitly understands it.
    if paths and not recognised:
        result.enable_full_matrix()

    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--github-output", help="Write key=value outputs to this GitHub Actions output file")
    parser.add_argument("paths", nargs="*", help="Changed paths; if omitted, paths are read from stdin")
    args = parser.parse_args()

    paths = args.paths if args.paths else [line.rstrip("\n") for line in sys.stdin]
    result = classify(paths)
    values = {key: str(value).lower() for key, value in asdict(result).items()}

    for key, value in values.items():
        print(f"{key}={value}")

    if args.github_output:
        with open(args.github_output, "a", encoding="utf-8") as output:
            for key, value in values.items():
                output.write(f"{key}={value}\n")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
