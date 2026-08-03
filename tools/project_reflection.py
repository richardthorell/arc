#!/usr/bin/env python3
"""Generate project-module reflection metadata from ARC C++ annotations."""

from __future__ import annotations

import argparse
import json
import pathlib
import re
import sys
from dataclasses import dataclass

KINDS = {"bool": "boolean", "int": "signed_integer", "uint": "unsigned_integer", "float": "floating_point",
         "string": "string", "enum": "enumeration", "vector2": "vector2", "vector3": "vector3",
         "vector4": "vector4", "quaternion": "quaternion", "entity": "entity_reference",
         "asset": "asset_reference", "structure": "structure", "sequence": "sequence"}
FLAGS = {"editable": "editable", "readonly": "read_only", "transient": "transient", "save_game": "save_game",
         "prefab": "prefab_override", "replicated": "replicated", "serialized": "serialized"}


def split_arguments(value: str) -> list[str]:
    result, current = [], []
    quoted = escaped = False
    for character in value:
        if escaped:
            current.append(character); escaped = False
        elif character == "\\":
            current.append(character); escaped = True
        elif character == '"':
            current.append(character); quoted = not quoted
        elif character == "," and not quoted:
            result.append("".join(current).strip()); current.clear()
        else:
            current.append(character)
    result.append("".join(current).strip())
    return result


def string_argument(value: str, label: str) -> str:
    try:
        decoded = json.loads(value)
    except json.JSONDecodeError as exception:
        raise ValueError(f"{label} must be a C++ string literal") from exception
    if not isinstance(decoded, str):
        raise ValueError(f"{label} must be a string")
    return decoded


def require_hex(value: str, length: int, label: str) -> None:
    if not re.fullmatch(rf"[0-9a-fA-F]{{{length}}}", value):
        raise ValueError(f"{label} must contain exactly {length} hexadecimal digits")


@dataclass
class Field:
    stable_id: str; cpp_name: str; display_name: str; category: str; tooltip: str; kind: str
    default_json: str; minimum: str; maximum: str; flags: list[str]; asset_type: str; entity_component: str


@dataclass
class Component:
    stable_id: str; cpp_name: str; display_name: str; category: str; tooltip: str; schema_version: int
    fields: list[Field]


def next_declaration(lines: list[str], start: int, expression: re.Pattern[str], label: str) -> tuple[int, str]:
    for index in range(start, min(start + 8, len(lines))):
        match = expression.search(lines[index])
        if match:
            return index, match.group(1)
    raise ValueError(f"{label} must be followed by its C++ declaration")


def macro_arguments(lines: list[str], start: int, name: str) -> tuple[str, int] | None:
    marker = re.search(rf"\b{name}\s*\(", lines[start])
    if not marker: return None
    text = "\n".join(lines[start:min(start + 24, len(lines))])
    position = marker.end(); depth = 1; quoted = escaped = False
    while position < len(text):
        character = text[position]
        if escaped: escaped = False
        elif character == "\\": escaped = True
        elif character == '"': quoted = not quoted
        elif not quoted and character == "(": depth += 1
        elif not quoted and character == ")":
            depth -= 1
            if depth == 0:
                consumed = text[:position + 1].count("\n")
                return text[marker.end():position], start + consumed
        position += 1
    raise ValueError(f"{name} annotation is not terminated")


def parse_headers(paths: list[pathlib.Path]) -> list[Component]:
    components, type_ids = [], set()
    for path in paths:
        lines = path.read_text(encoding="utf-8").splitlines(); index = 0
        while index < len(lines):
            component_macro = macro_arguments(lines, index, "ARC_COMPONENT")
            if not component_macro:
                index += 1; continue
            component_arguments, component_end = component_macro
            args = split_arguments(component_arguments)
            if len(args) != 5:
                raise ValueError(f"{path}:{index + 1}: ARC_COMPONENT requires 5 arguments")
            stable_id = string_argument(args[0], "component stable ID").lower(); require_hex(stable_id, 32, "component stable ID")
            if stable_id in type_ids: raise ValueError(f"duplicate component stable ID {stable_id}")
            type_ids.add(stable_id)
            try: schema_version = int(args[1])
            except ValueError as exception: raise ValueError("component schema version must be an integer") from exception
            if schema_version < 1: raise ValueError("component schema version must be positive")
            declaration_index, cpp_name = next_declaration(lines, component_end + 1,
                re.compile(r"\b(?:struct|class)\s+([A-Za-z_]\w*)"), "ARC_COMPONENT")
            component = Component(stable_id, cpp_name, string_argument(args[2], "component display name"),
                string_argument(args[3], "component category"), string_argument(args[4], "component tooltip"),
                schema_version, [])
            if re.search(r"}\s*;", lines[declaration_index]):
                components.append(component); index = declaration_index + 1; continue
            field_ids, cursor = set(), declaration_index + 1
            while cursor < len(lines) and not re.search(r"^\s*};", lines[cursor]):
                field_macro = macro_arguments(lines, cursor, "ARC_PROPERTY")
                if not field_macro: cursor += 1; continue
                field_arguments, field_end = field_macro
                field_args = split_arguments(field_arguments)
                if len(field_args) != 11: raise ValueError(f"{path}:{cursor + 1}: ARC_PROPERTY requires 11 arguments")
                field_id = string_argument(field_args[0], "field stable ID").lower(); require_hex(field_id, 16, "field stable ID")
                if field_id in field_ids: raise ValueError(f"duplicate field stable ID {field_id} in {cpp_name}")
                field_ids.add(field_id)
                kind = string_argument(field_args[4], "field kind")
                if kind not in KINDS: raise ValueError(f"unsupported reflected field kind {kind!r}")
                flags = [part.strip() for part in string_argument(field_args[8], "field flags").split("|") if part.strip()]
                unknown = set(flags) - FLAGS.keys()
                if unknown: raise ValueError(f"unsupported reflected field flags {sorted(unknown)}")
                _, cpp_field = next_declaration(lines, field_end + 1,
                    re.compile(r"\b([A-Za-z_]\w*)\s*(?:\{|=|;)"), "ARC_PROPERTY")
                default_json = string_argument(field_args[5], "field default")
                try: json.loads(default_json)
                except json.JSONDecodeError as exception: raise ValueError(f"default for {cpp_name}.{cpp_field} is invalid JSON") from exception
                component.fields.append(Field(field_id, cpp_field, string_argument(field_args[1], "field display name"),
                    string_argument(field_args[2], "field category"), string_argument(field_args[3], "field tooltip"),
                    kind, default_json, string_argument(field_args[6], "field minimum"),
                    string_argument(field_args[7], "field maximum"), flags,
                    string_argument(field_args[9], "asset restriction"), string_argument(field_args[10], "entity restriction")))
                cursor = field_end + 1
            components.append(component); index = cursor + 1
    return components


def q(value: str) -> str: return json.dumps(value)


def generate_cpp(components: list[Component], headers: list[pathlib.Path], namespace: str,
                 component_namespace: str) -> str:
    lines = ["#pragma once", "", "// Generated by ARC project reflection. Do not edit.",
             "#include <arc/ecs/reflection.h>", "#include <arc/project/project_module.h>",
             "#include <array>", "#include <cstddef>", "#include <type_traits>"]
    lines += [f'#include "{header.as_posix()}"' for header in headers]
    lines += ["", f"namespace {namespace}", "{", ""]
    for ci, component in enumerate(components):
        lines.append(
            f"inline constexpr std::array<arc::project::game_field_descriptor_v1, {len(component.fields)}> "
            f"component_{ci}_fields{{{{")
        for field in component.fields:
            flags = " | ".join(f"arc::project::game_field_flags_v1::{FLAGS[item]}" for item in field.flags) or "arc::project::game_field_flags_v1::none"
            lines.append("    {" + f"0x{field.stable_id}ull, {q(field.cpp_name)}, {q(field.display_name)}, {q(field.category)}, "
                f"{q(field.tooltip)}, arc::project::game_field_kind_v1::{KINDS[field.kind]}, {flags}, {q(field.default_json)}, "
                f"{field.minimum or '0.0'}, {field.maximum or '0.0'}, {'true' if field.minimum else 'false'}, "
                f"{'true' if field.maximum else 'false'}, {q(field.asset_type) if field.asset_type else 'nullptr'}, "
                f"{q(field.entity_component) if field.entity_component else 'nullptr'}" + "},")
        lines.append("}};")
    lines.append(
        f"inline constexpr std::array<arc::project::game_component_descriptor_v1, {len(components)}> components{{{{")
    for ci, component in enumerate(components):
        lines.append("    {" + f"{q(component.stable_id)}, {q(component.cpp_name)}, {q(component.display_name)}, "
            f"{q(component.category)}, {q(component.tooltip)}, {component.schema_version}, "
            f"component_{ci}_fields.data(), component_{ci}_fields.size()" + "},")
    lines += ["}};", "", f"}} // namespace {namespace}", "", "namespace arc::ecs", "{", ""]
    kind_map = {"bool": "boolean", "int": "signed_integer", "uint": "unsigned_integer",
                "float": "floating_point", "string": "string", "enum": "enumeration",
                "vector2": "vector2", "vector3": "vector3", "vector4": "vector4",
                "quaternion": "quaternion", "entity": "entity_reference", "asset": "asset_reference",
                "structure": "structure", "sequence": "sequence"}
    flag_map = {"editable": "editable", "readonly": "read_only", "transient": "transient",
                "save_game": "save_game", "prefab": "prefab_override", "replicated": "replicated",
                "serialized": "serialized"}
    for ci, component in enumerate(components):
        qualified = f"{component_namespace}::{component.cpp_name}" if component_namespace else component.cpp_name
        high, low = component.stable_id[:16], component.stable_id[16:]
        lines += [f"static_assert(std::is_standard_layout_v<{qualified}>,",
                  f"              {q('Reflected ARC components must use standard-layout storage')});",
                  f"template <> struct component_traits<{qualified}>", "{",
                  "    static constexpr bool reflected = true;",
                  f"    static constexpr std::string_view canonical_name = {q(component.cpp_name)};",
                  f"    static constexpr component_type_id id{{0x{high}ull, 0x{low}ull}};",
                  f"    static constexpr std::array<component_field_descriptor, {len(component.fields)}> fields{{{{"]
        for field in component.fields:
            flags = " | ".join(f"reflected_field_flags::{flag_map[item]}" for item in field.flags) or \
                "reflected_field_flags::none"
            minimum = f"std::optional<double>{{{field.minimum}}}" if field.minimum else "std::nullopt"
            maximum = f"std::optional<double>{{{field.maximum}}}" if field.maximum else "std::nullopt"
            lines.append("        {" + f"0x{field.stable_id}ull, {q(field.cpp_name)}, {q(field.display_name)}, "
                         f"reflected_field_kind::{kind_map[field.kind]}, {flags}, offsetof({qualified}, {field.cpp_name}), "
                         f"sizeof(decltype({qualified}::{field.cpp_name})), {q(field.tooltip)}, \"\", {minimum}, {maximum}, "
                         f"{q(field.category)}, {q(field.asset_type)}, {q(field.entity_component)}" + "},")
        lines += ["    }};", "    static constexpr component_descriptor descriptor{",
                  f"        id, canonical_name, {q(component.display_name)}, {component.schema_version}, "
                  f"sizeof({qualified}), alignof({qualified}), fields, false, false, {q(component.tooltip)}}};",
                  "};", ""]
    lines += ["} // namespace arc::ecs", ""]
    return "\n".join(lines)


def generate_json(components: list[Component]) -> str:
    return json.dumps({"schemaVersion": 1, "components": [{"id": c.stable_id, "canonicalName": c.cpp_name,
        "displayName": c.display_name, "category": c.category, "tooltip": c.tooltip, "schemaVersion": c.schema_version,
        "fields": [{"id": f.stable_id, "name": f.cpp_name, "displayName": f.display_name, "category": f.category,
            "tooltip": f.tooltip, "kind": f.kind, "default": json.loads(f.default_json),
            "minimum": float(f.minimum) if f.minimum else None, "maximum": float(f.maximum) if f.maximum else None,
            "flags": f.flags, "assetType": f.asset_type or None, "entityComponent": f.entity_component or None}
            for f in c.fields]} for c in components]}, indent=2) + "\n"


def write_if_changed(path: pathlib.Path, content: str) -> None:
    if path.exists() and path.read_text(encoding="utf-8") == content: return
    path.parent.mkdir(parents=True, exist_ok=True); path.write_text(content, encoding="utf-8", newline="\n")


def main() -> int:
    parser = argparse.ArgumentParser(); parser.add_argument("--header", action="append", required=True, type=pathlib.Path)
    parser.add_argument("--cpp-output", required=True, type=pathlib.Path); parser.add_argument("--json-output", required=True, type=pathlib.Path)
    parser.add_argument("--namespace", default="arc_project_generated")
    parser.add_argument("--component-namespace", default="")
    arguments = parser.parse_args()
    try:
        components = parse_headers(arguments.header)
        if not components: raise ValueError("no ARC_COMPONENT annotations were found")
        write_if_changed(arguments.cpp_output, generate_cpp(components, arguments.header, arguments.namespace,
                                                             arguments.component_namespace))
        write_if_changed(arguments.json_output, generate_json(components))
    except (OSError, ValueError) as exception:
        print(f"ARC reflection error: {exception}", file=sys.stderr); return 1
    return 0


if __name__ == "__main__": raise SystemExit(main())
