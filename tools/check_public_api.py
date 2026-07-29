#!/usr/bin/env python3
"""Enforce ARC's source-level public API conventions."""

from __future__ import annotations

import json
import pathlib
import re
import sys
from typing import List


ROOT = pathlib.Path(__file__).resolve().parents[1]
ENGINE = ROOT / "engine"


def public_headers() -> List[pathlib.Path]:
    headers: List[pathlib.Path] = []
    for path in ENGINE.glob("*/inc/arc/**/*.h"):
        relative = path.as_posix()
        if (
            "/simd/arch/" in relative
            or "/detail/" in relative
            or relative.endswith("/simd/core/ops/detail.h")
        ):
            continue
        headers.append(path)
    return sorted(headers)


def main() -> int:
    errors: List[str] = []
    forbidden_files = (
        ENGINE / "assets/inc/arc/assets.h",
        ENGINE / "io/inc/arc/io.h",
        ENGINE / "persistence/inc/arc/persistence.h",
        ENGINE / "scene/inc/arc/scene/entity.h",
        ENGINE / "scene/inc/arc/scene/entity_guid.h",
        ENGINE / "scene/inc/arc/scene/registry.h",
    )
    for path in forbidden_files:
        if path.exists():
            errors.append(f"obsolete public wrapper still exists: {path.relative_to(ROOT)}")

    forbidden_include = re.compile(
        r"#include\s*<arc/(?:assets|io|persistence)\.h>|"
        r"#include\s*<arc/scene/(?:entity|entity_guid|registry)\.h>"
    )
    scene_alias = re.compile(
        r"\busing\s+(?:entity|entity_guid|registry)\s*=\s*ecs::"
    )
    public_desc = re.compile(r"\b[A-Za-z_][A-Za-z0-9_]*_desc\b")
    discardable_result = re.compile(
        r"\bstruct\s+(?!\[\[nodiscard\]\])"
        r"[A-Za-z_][A-Za-z0-9_]*_result\b"
    )
    bool_error_output = re.compile(
        r"\bbool\s+[A-Za-z_][A-Za-z0-9_]*\s*\([^;{}]*"
        r"std::string\s*(?:[&*])\s*(?:error|message)\b",
        re.DOTALL,
    )
    third_party = re.compile(r"\b(?:Vk[A-Z][A-Za-z0-9_]*|sqlite3|nlohmann::)\b")

    for header in public_headers():
        text = header.read_text(encoding="utf-8")
        relative = header.relative_to(ROOT)
        if forbidden_include.search(text):
            errors.append(f"{relative}: uses a removed include path")
        if scene_alias.search(text):
            errors.append(f"{relative}: exposes a removed scene compatibility alias")
        if public_desc.search(text):
            errors.append(f"{relative}: exposes a public _desc identifier")
        if discardable_result.search(text):
            errors.append(f"{relative}: exposes a result type without [[nodiscard]]")
        if bool_error_output.search(text):
            errors.append(f"{relative}: exposes bool plus string error output")
        if "render-vulkan" not in relative.parts and third_party.search(text):
            errors.append(f"{relative}: leaks a private third-party type")

    normalized_modules = ("diagnostics", "jobs", "memory", "framework", "simd")
    for module in normalized_modules:
        for header in (ENGINE / module / "inc" / "arc" / module).rglob("*.h"):
            text = header.read_text(encoding="utf-8")
            if re.search(r"namespace\s+arc\s*(?:\{|$)", text, re.MULTILINE):
                errors.append(
                    f"{header.relative_to(ROOT)}: declares symbols in root namespace arc"
                )

    schema_path = ENGINE / "scene/schema/components.arccomponents.json"
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    for component in schema["components"]:
        label = f"{schema['namespace']}.{component.get('type', '<unknown>')}"
        if not str(component.get("description", "")).strip():
            errors.append(f"{schema_path.relative_to(ROOT)}: {label} lacks a description")
        for field in component.get("fields", []):
            field_label = f"{label}.{field.get('name', '<unknown>')}"
            if not str(field.get("description", "")).strip():
                errors.append(f"{schema_path.relative_to(ROOT)}: {field_label} lacks a description")
            if not str(field.get("unit", "")).strip():
                errors.append(f"{schema_path.relative_to(ROOT)}: {field_label} lacks a unit")
            if not isinstance(field.get("constraints"), dict):
                errors.append(f"{schema_path.relative_to(ROOT)}: {field_label} lacks constraints")

    if errors:
        print("ARC public API policy violations:", file=sys.stderr)
        for error in errors:
            print(f"  - {error}", file=sys.stderr)
        return 1
    print(f"ARC public API policy passed for {len(public_headers())} headers")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
