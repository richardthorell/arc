#!/usr/bin/env python3
"""Compile, validate, and deterministically package ARC built-in shaders."""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import re
import struct
import subprocess
import sys


STAGE_EXTENSIONS = {".vert", ".frag", ".comp"}
INCLUDE_PATTERN = re.compile(r'^\s*#include\s+["<]([^">]+)[">]', re.MULTILINE)


def discover_sources(root: pathlib.Path) -> list[pathlib.Path]:
    return sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in STAGE_EXTENSIONS
    )


def include_closure(path: pathlib.Path, source_root: pathlib.Path, active: set[pathlib.Path]) -> list[pathlib.Path]:
    resolved = path.resolve()
    if resolved in active:
        raise RuntimeError(f"Shader include cycle at {path}")
    active.add(resolved)
    closure: list[pathlib.Path] = []
    text = path.read_text(encoding="utf-8")
    for include in INCLUDE_PATTERN.findall(text):
        candidates = ((path.parent / include).resolve(), (source_root / include).resolve())
        candidate = next((value for value in candidates if value.is_file()), candidates[-1])
        if not candidate.is_file() or source_root.resolve() not in candidate.parents:
            raise RuntimeError(f"{path}: missing or invalid include {include!r}")
        closure.extend(include_closure(candidate, source_root, active))
        closure.append(candidate)
    active.remove(resolved)
    return closure


def shader_name(path: pathlib.Path, source_root: pathlib.Path) -> str:
    relative = path.relative_to(source_root).as_posix()
    name = re.sub(r"[^0-9A-Za-z_]", "_", relative) + "_spv"
    if name[0].isdigit():
        name = "_" + name
    return name


def build_header(compiled: list[tuple[str, bytes]]) -> str:
    lines = [
        "#pragma once",
        "",
        "// Generated from assets/shaders with glslc. Do not edit by hand.",
        "#include <cstdint>",
        "",
        "namespace arc::render::vulkan::builtin",
        "{",
    ]
    for name, data in compiled:
        if len(data) % 4:
            raise RuntimeError(f"{name}: SPIR-V byte count is not word aligned")
        words = struct.unpack(f"<{len(data) // 4}I", data)
        lines.extend(["", f"inline constexpr std::uint32_t {name}[] = {{"])
        for offset in range(0, len(words), 8):
            row = ", ".join(f"0x{word:08x}u" for word in words[offset : offset + 8])
            suffix = "," if offset + 8 < len(words) else ""
            lines.append(f"    {row}{suffix}")
        lines.append("};")
    lines.extend(["", "} // namespace arc::render::vulkan::builtin", ""])
    return "\n".join(lines)


def run_tool(command: list[str]) -> None:
    completed = subprocess.run(command, text=True, capture_output=True)
    if completed.returncode:
        if completed.stdout:
            print(completed.stdout, file=sys.stderr)
        if completed.stderr:
            print(completed.stderr, file=sys.stderr)
        raise RuntimeError(f"Command failed: {' '.join(command)}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=pathlib.Path, required=True)
    parser.add_argument("--output-dir", type=pathlib.Path, required=True)
    parser.add_argument("--header", type=pathlib.Path, required=True)
    parser.add_argument("--glslc", default="glslc")
    parser.add_argument("--spirv-val", default="spirv-val")
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()

    source_root = args.source_root.resolve()
    sources = discover_sources(source_root)
    if not sources:
        raise RuntimeError(f"No shader stages found under {source_root}")

    names: set[str] = set()
    compiled: list[tuple[str, bytes]] = []
    manifest: list[dict[str, object]] = []
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for source in sources:
        name = shader_name(source, source_root)
        if name in names:
            raise RuntimeError(f"Duplicate generated shader name: {name}")
        names.add(name)
        closure = sorted(set(include_closure(source, source_root, set())))
        published = args.output_dir / f"{name}.spv"
        temporary = published.with_suffix(".spv.tmp")
        run_tool(
            [
                args.glslc,
                "--target-env=vulkan1.2",
                f"-I{source_root}",
                f"-I{source_root / 'include'}",
                str(source),
                "-o",
                str(temporary),
            ]
        )
        run_tool([args.spirv_val, "--target-env", "vulkan1.2", str(temporary)])
        data = temporary.read_bytes()
        temporary.replace(published)
        compiled.append((name, data))
        digest = hashlib.sha256()
        digest.update(source.read_bytes())
        for dependency in closure:
            digest.update(dependency.relative_to(source_root).as_posix().encode("utf-8"))
            digest.update(dependency.read_bytes())
        manifest.append(
            {
                "source": source.relative_to(source_root).as_posix(),
                "output": published.name,
                "sourceClosureHash": digest.hexdigest(),
                "spirvHash": hashlib.sha256(data).hexdigest(),
                "size": len(data),
            }
        )

    expected = build_header(compiled)
    if args.write:
        args.header.write_text(expected, encoding="utf-8", newline="\n")
    if args.check:
        actual = args.header.read_text(encoding="utf-8")
        if actual.replace("\r\n", "\n") != expected:
            print(
                f"{args.header} is stale; regenerate with tools/compile_builtin_shaders.py --write",
                file=sys.stderr,
            )
            return 1
    manifest_path = args.output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps({"format": "arc-builtin-shaders", "version": 1, "shaders": manifest}, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(f"Compiled and validated {len(compiled)} ARC built-in shaders")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, RuntimeError) as error:
        print(error, file=sys.stderr)
        raise SystemExit(1)
