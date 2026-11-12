#!/usr/bin/env python3
"""Relocate and boot an ARC D0 Shipping distribution without using its build tree.

Use only real arc-project package output. The launched process executes from a
fresh unrelated working directory; it discovers content relative to itself.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import sys
import tempfile
import zipfile


class SmokeFailure(Exception):
    pass


def _relative(name: str) -> PurePosixPath:
    if not isinstance(name, str) or not name or "\\" in name or ":" in name:
        raise SmokeFailure(f"unsafe package path: {name!r}")
    path = PurePosixPath(name)
    if path.is_absolute() or any(part in ("", ".", "..") for part in name.split("/")):
        raise SmokeFailure(f"unsafe package path: {name!r}")
    return path


def verify_distribution(root: Path) -> dict:
    document_path = root / "arc-distribution.json"
    if not document_path.is_file() or document_path.is_symlink():
        raise SmokeFailure("missing arc-distribution.json")
    try:
        manifest = json.loads(document_path.read_text(encoding="utf-8"))
        if manifest["format"] != "arc.distribution" or manifest["formatVersion"] != 1:
            raise SmokeFailure("incompatible D0 distribution manifest")
        if manifest["target"]["configuration"] != "Shipping":
            raise SmokeFailure("expected Shipping distribution")
        if manifest["entrypoint"]["kind"] != "executable":
            raise SmokeFailure("expected a native executable entrypoint")
        if manifest["role"] not in ("client", "server"):
            raise SmokeFailure("unsupported runtime role")
        inventory = manifest["files"]
        entry = _relative(manifest["entrypoint"]["value"]).as_posix()
        config = _relative(manifest["runtimeConfig"]).as_posix()
        cooks = [_relative(value).as_posix() for value in manifest["content"]["cookManifests"]]
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise SmokeFailure(f"invalid distribution manifest: {exc}") from exc
    expected: set[str] = {"arc-distribution.json"}
    kinds: dict[str, str] = {}
    folded: set[str] = set()
    for item in inventory:
        name = _relative(item["path"]).as_posix()
        if name.casefold() in folded or name.casefold() == "arc-distribution.json":
            raise SmokeFailure(f"duplicate or reserved inventory path: {name}")
        folded.add(name.casefold())
        expected.add(name)
        kinds[name] = item["kind"]
        location = root.joinpath(*PurePosixPath(name).parts)
        if location.is_symlink() or not location.is_file():
            raise SmokeFailure(f"missing or unsafe inventory file: {name}")
        if location.stat().st_size != item["size"]:
            raise SmokeFailure(f"file size mismatch: {name}")
        digest = hashlib.sha256()
        with location.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        if digest.hexdigest() != item["sha256"]:
            raise SmokeFailure(f"file hash mismatch: {name}")

    if kinds.get(entry) != "executable" or kinds.get(config) != "configuration":
        raise SmokeFailure("entrypoint or runtime configuration missing from inventory")
    if not cooks or any(kinds.get(path) != "content" for path in cooks):
        raise SmokeFailure("cooked-content manifest missing from inventory")

    forbidden = {"source", "editor", ".arc", "intermediate", "node_modules"}
    for path in root.rglob("*"):
        if path.is_symlink():
            raise SmokeFailure(f"symlink in distribution: {path.relative_to(root)}")
        if path.is_dir():
            continue
        relative = path.relative_to(root).as_posix()
        if relative not in expected:
            raise SmokeFailure(f"undeclared file in distribution: {relative}")
        if forbidden.intersection(part.casefold() for part in path.relative_to(root).parts):
            raise SmokeFailure(f"authoring/build file in distribution: {relative}")
        if path.suffix.casefold() in {".pdb", ".obj", ".lib", ".ilk"}:
            raise SmokeFailure(f"development artifact in Shipping distribution: {relative}")
    return manifest


def extract_archive(archive: Path, destination: Path) -> None:
    with zipfile.ZipFile(archive) as source:
        seen: set[str] = set()
        for info in source.infolist():
            name = _relative(info.filename).as_posix()
            if name.casefold() in seen or info.is_dir():
                raise SmokeFailure(f"duplicate or unexpected ZIP entry: {name}")
            seen.add(name.casefold())
            # ZIP unix mode symlinks are never supported for shipping products.
            if ((info.external_attr >> 16) & 0o170000) == 0o120000:
                raise SmokeFailure(f"ZIP contains a symbolic link: {name}")
        source.extractall(destination)


def smoke(distribution: Path | None, archive: Path | None, timeout: float,
          arguments: list[str], artifacts: Path | None = None,
          expected_output: str | None = None) -> None:
    with tempfile.TemporaryDirectory(prefix="arc-d3-relocated-") as temporary:
        workspace = Path(temporary)
        stage = workspace / "relocated" / "Shipping"
        stage.parent.mkdir(parents=True)
        if archive is not None:
            extract_archive(archive, stage)
        else:
            assert distribution is not None
            shutil.copytree(distribution, stage)
        manifest = verify_distribution(stage)
        entry = stage.joinpath(*PurePosixPath(manifest["entrypoint"]["value"]).parts)
        unrelated_working_directory = workspace / "unrelated-working-directory"
        unrelated_working_directory.mkdir()
        writable_data = workspace / "writable-data"
        writable_data.mkdir()
        env = os.environ.copy()
        env.pop("ELECTRON_RUN_AS_NODE", None)
        env["ARC_SERVER_DATA_ROOT"] = str(writable_data)
        command = [str(entry), *arguments]
        if manifest["role"] == "server" and not arguments:
            command += ["--ticks", "10", "--no-sleep"]
        try:
            result = subprocess.run(command, cwd=unrelated_working_directory,
                                    env=env, capture_output=True, text=True,
                                    errors="replace", timeout=timeout, check=False)
            code, stdout, stderr = result.returncode, result.stdout, result.stderr
        except subprocess.TimeoutExpired as exc:
            code = 124
            stdout = (exc.stdout or b"").decode(errors="replace") if isinstance(exc.stdout, bytes) else (exc.stdout or "")
            stderr = (exc.stderr or b"").decode(errors="replace") if isinstance(exc.stderr, bytes) else (exc.stderr or "")
            stderr += f"\nTimed out after {timeout}s"
        except OSError as exc:
            code, stdout, stderr = 127, "", str(exc)

        if artifacts is not None:
            artifacts.mkdir(parents=True, exist_ok=True)
            (artifacts / "stdout.txt").write_text(stdout, encoding="utf-8")
            (artifacts / "stderr.txt").write_text(stderr, encoding="utf-8")
            (artifacts / "launch.json").write_text(
                json.dumps({"exitCode": code, "role": manifest["role"],
                            "entrypoint": manifest["entrypoint"]["value"],
                            "timeoutSeconds": timeout}, indent=2) + "\n", encoding="utf-8")
            shutil.copyfile(stage / "arc-distribution.json", artifacts / "arc-distribution.json")

        if code != 0:
            raise SmokeFailure(f"relocated {manifest['role']} exited {code}: {stderr[-3000:]}")
        if expected_output is not None and expected_output not in stdout:
            raise SmokeFailure(f"missing expected startup output: {expected_output!r}; output: {stdout[-2000:]}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--distribution", type=Path)
    source.add_argument("--archive", type=Path)
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--artifacts", type=Path)
    parser.add_argument("--expect-output")
    parser.add_argument("arguments", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.timeout <= 0:
        parser.error("--timeout must be positive")
    remaining = args.arguments
    if remaining and remaining[0] == "--":
        remaining = remaining[1:]
    try:
        smoke(args.distribution, args.archive, args.timeout, remaining,
              args.artifacts, args.expect_output)
    except (SmokeFailure, OSError, zipfile.BadZipFile, ValueError) as exc:
        print(f"ARC relocated package smoke failed: {exc}", file=sys.stderr)
        return 1
    print("ARC relocated package smoke passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
