"""C9.1 remote release contract and reconnect tests."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from remote_content_contract import (ContractError, canonical_release_bytes,
                                     feed_recovery, validate_release)


class RemoteContentContractTests(unittest.TestCase):
    def setUp(self) -> None:
        self.target = {"platform": "windows", "architecture": "x86_64",
                       "renderer": "vulkan", "cookProfile": "windows-x64-vulkan",
                       "textureFamily": "bc"}
        manifest_hash = hashlib.sha256(b"native cook manifest").hexdigest()
        chunk_hash = hashlib.sha256(b"native cooker chunk").hexdigest()
        self.release = {
            "format": "arc.remote-content", "formatVersion": 1,
            "projectGuid": "11111111-2222-4333-8444-555555555555",
            "engineVersion": "0.1.0", "baseBuildId": "build-abc",
            "target": dict(self.target), "generation": 5, "complete": True,
            "cookManifestName": "windows-x64-vulkan.arccookmanifest",
            "cookManifest": {
                "path": f"objects/sha256/{manifest_hash}",
                "sha256": manifest_hash, "size": 123,
            },
            "chunks": [{
                "name": "startup-hash.arcpak",
                "path": f"objects/sha256/{chunk_hash}",
                "sha256": chunk_hash, "size": 456,
            }],
        }

    def validate(self, release=None, **kwargs):
        return validate_release(self.release if release is None else release,
                                project_guid=self.release["projectGuid"],
                                engine_version="0.1.0", base_build_id="build-abc",
                                target=self.target, **kwargs)

    def test_shared_native_fixture(self):
        fixture = (Path(__file__).resolve().parents[2] / "engine" / "assets" /
                   "tests" / "fixtures" / "remote-content-v1.json")
        self.assertEqual(self.validate(json.loads(fixture.read_text(encoding="utf-8"))),
                         self.validate())

    def test_valid_windows_and_android_targets(self):
        self.assertEqual(self.validate()["generation"], 5)
        android = copy.deepcopy(self.release)
        android["target"] = {"platform": "android", "architecture": "arm64",
                             "renderer": "vulkan", "cookProfile": "android-arm64-vulkan",
                             "textureFamily": "astc"}
        self.assertEqual(validate_release(
            android, project_guid=android["projectGuid"], engine_version="0.1.0",
            base_build_id="build-abc", target=android["target"])["target"]["textureFamily"], "astc")

    def test_wrong_identity_and_target_rejected(self):
        for key in ("projectGuid", "engineVersion", "baseBuildId"):
            value = copy.deepcopy(self.release)
            value[key] = "another-project" if key == "projectGuid" else "other"
            with self.subTest(field=key), self.assertRaises(ContractError):
                self.validate(value)
        for key in self.target:
            value = copy.deepcopy(self.release)
            value["target"][key] = "incorrect"
            with self.subTest(field=key), self.assertRaisesRegex(ContractError, "target"):
                self.validate(value)

    def test_rejects_unsupported_version_and_stale_generation(self):
        value = copy.deepcopy(self.release)
        value["formatVersion"] = 2
        with self.assertRaisesRegex(ContractError, "format/version"):
            self.validate(value)
        with self.assertRaisesRegex(ContractError, "generation"):
            self.validate(minimum_generation=5)
        self.assertEqual(self.validate(minimum_generation=4)["generation"], 5)

    def test_rejects_partial_release_and_invalid_hash(self):
        value = copy.deepcopy(self.release)
        value["complete"] = False
        with self.assertRaisesRegex(ContractError, "not completely published"):
            self.validate(value)
        value["complete"] = True
        value["chunks"][0]["sha256"] = "NOT-A-HASH"
        with self.assertRaisesRegex(ContractError, "SHA-256"):
            self.validate(value)

    def test_rejects_traversal_and_unsafe_object_address(self):
        for name in ("../escape.arcpak", "/tmp/evil.arcpak", "foo\\bar.arcpak",
                     "foo/bar.arcpak", ".arcpak"):
            value = copy.deepcopy(self.release)
            value["chunks"][0]["name"] = name
            with self.subTest(name=name), self.assertRaises(ContractError):
                self.validate(value)
        for path in ("../secrets", "https://evil.invalid/blob", "objects/sha256/../bad",
                     "objects/sha256/" + "f" * 64):
            value = copy.deepcopy(self.release)
            value["chunks"][0]["path"] = path
            with self.subTest(path=path), self.assertRaisesRegex(ContractError, "path"):
                self.validate(value)

    def test_rejects_duplicate_chunk_names_and_bool_sizes(self):
        value = copy.deepcopy(self.release)
        value["chunks"].append(copy.deepcopy(value["chunks"][0]))
        with self.assertRaises(ContractError):
            self.validate(value)
        value = copy.deepcopy(self.release)
        value["cookManifest"]["size"] = True
        with self.assertRaisesRegex(ContractError, "size"):
            self.validate(value)

    def test_canonical_bytes_are_stable_across_key_order(self):
        validated = self.validate()
        reverse = dict(reversed(list(validated.items())))
        self.assertEqual(canonical_release_bytes(validated), canonical_release_bytes(reverse))
        self.assertEqual(json.loads(canonical_release_bytes(validated)), validated)

    def test_feed_reconnect_replay_reset_and_reject(self):
        self.assertEqual(feed_recovery(last_acknowledged=8, oldest_available=7,
                                       newest_available=10), "replay")
        self.assertEqual(feed_recovery(last_acknowledged=5, oldest_available=8,
                                       newest_available=10), "full-snapshot")
        self.assertEqual(feed_recovery(last_acknowledged=10, oldest_available=8,
                                       newest_available=10), "up-to-date")
        self.assertEqual(feed_recovery(last_acknowledged=11, oldest_available=8,
                                       newest_available=10), "reject-regression")
        with self.assertRaises(ContractError):
            feed_recovery(last_acknowledged=-1, oldest_available=0, newest_available=2)


if __name__ == "__main__":
    unittest.main()
