#!/usr/bin/env python3
"""C9.1 transport-neutral cooked-content release and change-feed contract.

This module validates publication/discovery metadata only. It never serves
project source files, downloads content or activates runtime overlays.
"""
from __future__ import annotations

import json
import re
from collections.abc import Mapping
from typing import Any

FORMAT = "arc.remote-content"
FORMAT_VERSION = 1
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_GUID = re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\Z")
_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*\Z")
_TARGET_KEYS = ("platform", "architecture", "renderer", "cookProfile", "textureFamily")
_MAX_OBJECTS = 100_000
_MAX_SIZE = 1 << 44  # Metadata guard only; transport imposes stricter request bounds.


class ContractError(ValueError):
    """Incompatible or unsafe remote release metadata."""


def _mapping(value: Any, field: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise ContractError(f"{field} must be an object")
    return value


def _positive_size(value: Any, field: str) -> int:
    if type(value) is not int or value <= 0 or value > _MAX_SIZE:
        raise ContractError(f"{field} must be a positive bounded integer")
    return value


def _sha(value: Any, field: str) -> str:
    if not isinstance(value, str) or not _SHA256.fullmatch(value):
        raise ContractError(f"{field} must be a lowercase SHA-256 digest")
    return value


def _safe_name(value: Any, field: str) -> str:
    if not isinstance(value, str) or not _NAME.fullmatch(value) or value in {".", ".."}:
        raise ContractError(f"{field} is an unsafe name")
    return value


def _object_ref(value: Any, field: str) -> dict[str, Any]:
    item = _mapping(value, field)
    digest = _sha(item.get("sha256"), field + ".sha256")
    expected = f"objects/sha256/{digest}"
    if item.get("path") != expected:
        raise ContractError(f"{field}.path must be the immutable object address {expected}")
    size = _positive_size(item.get("size"), field + ".size")
    return {"path": expected, "sha256": digest, "size": size}


def canonical_release_bytes(document: Mapping[str, Any]) -> bytes:
    """Stable publisher bytes for identical metadata; no timestamps or URLs."""
    return (json.dumps(document, sort_keys=True, ensure_ascii=False,
                       separators=(",", ":"), allow_nan=False) + "\n").encode("utf-8")


def validate_release(document: Any, *, project_guid: str, engine_version: str,
                     base_build_id: str, target: Mapping[str, str],
                     minimum_generation: int = 0) -> dict[str, Any]:
    """Validate an untrusted descriptor against the compiled/selected runtime.

    The existing .arccookmanifest remains the dependency graph and asset GUID
    inventory. This descriptor merely maps it and immutable chunk objects to
    versioned transport references.
    """
    release = _mapping(document, "release")
    if release.get("format") != FORMAT or type(release.get("formatVersion")) is not int or release["formatVersion"] != FORMAT_VERSION:
        raise ContractError("unsupported remote release format/version")
    guid = release.get("projectGuid")
    if not isinstance(guid, str) or not _GUID.fullmatch(guid):
        raise ContractError("invalid projectGuid")
    for key, expected in (("projectGuid", project_guid),
                          ("engineVersion", engine_version),
                          ("baseBuildId", base_build_id)):
        if not isinstance(expected, str) or not expected or release.get(key) != expected:
            raise ContractError(f"incompatible {key}")
    published_target = _mapping(release.get("target"), "target")
    for key in _TARGET_KEYS:
        value = published_target.get(key)
        if not isinstance(value, str) or not value or value != target.get(key):
            raise ContractError(f"incompatible target.{key}")

    generation = release.get("generation")
    if type(generation) is not int or generation < 1 or generation <= minimum_generation:
        raise ContractError("stale or invalid release generation")
    manifest = _object_ref(release.get("cookManifest"), "cookManifest")
    manifest_name = _safe_name(release.get("cookManifestName"), "cookManifestName")
    if not manifest_name.endswith(".arccookmanifest"):
        raise ContractError("cookManifestName must identify the existing cooker manifest")

    chunks = release.get("chunks")
    if not isinstance(chunks, list) or len(chunks) > _MAX_OBJECTS:
        raise ContractError("chunks must be a bounded array")
    names: set[str] = set()
    hashes: set[str] = {manifest["sha256"]}
    validated_chunks = []
    for index, chunk in enumerate(chunks):
        field = f"chunks[{index}]"
        item = _mapping(chunk, field)
        name = _safe_name(item.get("name"), field + ".name")
        if not name.endswith(".arcpak") or name.casefold() in names:
            raise ContractError(f"{field}.name is not a unique cooked package name")
        names.add(name.casefold())
        ref = _object_ref(item, field)
        if ref["sha256"] in hashes:
            raise ContractError(f"{field}.sha256 duplicates another release object")
        hashes.add(ref["sha256"])
        validated_chunks.append({"name": name, **ref})

    # A publisher must expose an entire immutable generation, never a partial
    # manifest/chunk set. Chunk-reference completeness is verified when the
    # existing cooked manifest is parsed by C9.2.
    if not isinstance(release.get("complete"), bool) or release["complete"] is not True:
        raise ContractError("release generation is not completely published")
    return {
        "format": FORMAT, "formatVersion": FORMAT_VERSION,
        "projectGuid": guid, "engineVersion": engine_version,
        "baseBuildId": base_build_id,
        "target": {key: published_target[key] for key in _TARGET_KEYS},
        "generation": generation, "complete": True,
        "cookManifestName": manifest_name, "cookManifest": manifest,
        "chunks": sorted(validated_chunks, key=lambda x: x["name"]),
    }


def feed_recovery(*, last_acknowledged: int, oldest_available: int,
                  newest_available: int) -> str:
    """Decide whether a development change-feed can replay or must reset.

    Sequence is strictly monotonic; the catalog referenced by a generation is
    published atomically before its feed notification. A CDN has no feed.
    """
    for value in (last_acknowledged, oldest_available, newest_available):
        if type(value) is not int or value < 0:
            raise ContractError("change-feed sequence must be a nonnegative integer")
    if newest_available < oldest_available:
        raise ContractError("change-feed bounds are inverted")
    if last_acknowledged > newest_available:
        return "reject-regression"
    if last_acknowledged == newest_available:
        return "up-to-date"
    if oldest_available and last_acknowledged + 1 < oldest_available:
        return "full-snapshot"
    return "replay"
