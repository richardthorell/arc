"""Fast source-fixture checks for A6's real external Android acceptance."""
from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from tools import test_android_a6_external_game as a6


ROOT = Path(__file__).resolve().parents[2]


class A6FixtureTests(unittest.TestCase):
    def test_required_scene_dependencies_are_present_and_consistent(self):
        source = ROOT / "tests/fixtures/android-device-smoke/Content/Scenes/Startup.arcscene"
        scene = json.loads(source.read_text(encoding="utf-8"))
        metadata = json.loads((source.parent / "Startup.arcscene.arcmeta").read_text(encoding="utf-8"))
        self.assertEqual(metadata["guid"], a6.SCENE_GUID)
        sources = {destination: ROOT / original for original, destination in a6.CONTENT}
        self.assertEqual(len(sources), len(a6.CONTENT))
        self.assertEqual(len(a6.PROJECTS), 2)
        self.assertNotEqual(
            a6.PROJECTS["A6Alpha"]["applicationId"],
            a6.PROJECTS["A6Beta"]["applicationId"],
        )
        for source in sources.values():
            self.assertTrue(source.is_file(), f"A6 fixture missing: {source}")
        for dependency in scene["dependencies"]:
            candidate = dependency["pathHint"]
            self.assertTrue(
                candidate in sources or
                (ROOT / "templates/blank-3d" / candidate).is_file(),
                f"A6 startup-scene required dependency was not included: {candidate}",
            )
            meta_path = sources.get(candidate + ".arcmeta")
            if meta_path is None:
                meta_path = ROOT / "templates/blank-3d" / (candidate + ".arcmeta")
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            self.assertEqual(meta["guid"], dependency["guid"])
            self.assertEqual(meta["type"], dependency["expectedType"])

    def test_snapshot_ignores_generated_paths_but_tracks_authored_files(self):
        with tempfile.TemporaryDirectory(prefix="arc-a6-snapshot-") as temp:
            root = Path(temp)
            for folder in ("Saved", "Build", "Content", "Source"):
                (root / folder).mkdir()
            (root / "Content/Scene.arcscene").write_text('{"scene":1}')
            (root / "Source/Runtime.cpp").write_text("void run() {}")
            (root / "Game.arcproject").write_text('{"guid":"abc"}')
            initial = a6.snapshot(root)
            (root / "Saved/log.txt").write_text("new log")
            (root / "Build/package.apk").write_bytes(b"apk")
            self.assertEqual(initial, a6.snapshot(root))
            (root / "Content/Scene.arcscene").write_text('{"scene":2}')
            self.assertNotEqual(initial, a6.snapshot(root))

    def test_command_rejects_unexpected_subprocess_outcomes(self):
        with tempfile.TemporaryDirectory(prefix="arc-a6-command-") as temp:
            root = Path(temp)
            _, result = a6.command(["python3", "-c", "print('{\"succeeded\":true}')"],
                                   cwd=root)
            self.assertTrue(result[0]["succeeded"])
            with self.assertRaisesRegex(ValueError, "Unexpected command status"):
                a6.command(["python3", "-c", "raise SystemExit(3)"], cwd=root)


if __name__ == "__main__":
    unittest.main()
