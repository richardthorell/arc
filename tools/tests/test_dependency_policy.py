from __future__ import annotations

import json
import pathlib
import shutil
import subprocess
import sys
import unittest
import uuid


SCRIPT = pathlib.Path(__file__).resolve().parents[1] / "check_dependencies.py"


class DependencyPolicyTests(unittest.TestCase):
    def make_fixture(self) -> pathlib.Path:
        temporary_parent = SCRIPT.parents[1] / "out" / "policy-tests"
        temporary_parent.mkdir(parents=True, exist_ok=True)
        root = temporary_parent / str(uuid.uuid4())
        root.mkdir()
        self.addCleanup(shutil.rmtree, root, True)
        (root / "editor").mkdir()
        (root / ".github/workflows").mkdir(parents=True)
        package = {"dependencies": {"react": "19.2.7"}, "devDependencies": {}}
        lock = {
            "packages": {
                "": {
                    "dependencies": {"react": "19.2.7"},
                    "devDependencies": {},
                }
            }
        }
        (root / "editor/package.json").write_text(json.dumps(package), encoding="utf-8")
        (root / "editor/package-lock.json").write_text(json.dumps(lock), encoding="utf-8")
        (root / "CMakeLists.txt").write_text("set(OK ON)\n", encoding="utf-8")
        (root / ".github/workflows/build.yml").write_text(
            "jobs:\n  build:\n    runs-on: ubuntu-24.04\n    steps:\n"
            "      - uses: actions/checkout@11d5960a326750d5838078e36cf38b85af677262\n",
            encoding="utf-8",
        )
        return root

    def run_policy(self, root: pathlib.Path) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [sys.executable, str(SCRIPT), "--root", str(root)],
            text=True,
            capture_output=True,
            check=False,
        )

    def test_accepts_exact_dependencies(self) -> None:
        self.assertEqual(self.run_policy(self.make_fixture()).returncode, 0)

    def test_rejects_npm_ranges(self) -> None:
        root = self.make_fixture()
        package = json.loads((root / "editor/package.json").read_text(encoding="utf-8"))
        package["dependencies"]["react"] = "^19.2.7"
        (root / "editor/package.json").write_text(json.dumps(package), encoding="utf-8")
        self.assertNotEqual(self.run_policy(root).returncode, 0)

    def test_rejects_lockfile_drift(self) -> None:
        root = self.make_fixture()
        lock = json.loads((root / "editor/package-lock.json").read_text(encoding="utf-8"))
        lock["packages"][""]["dependencies"]["react"] = "19.2.6"
        (root / "editor/package-lock.json").write_text(json.dumps(lock), encoding="utf-8")
        self.assertNotEqual(self.run_policy(root).returncode, 0)

    def test_rejects_floating_git_tags(self) -> None:
        root = self.make_fixture()
        (root / "CMakeLists.txt").write_text("FetchContent_Declare(x GIT_TAG master)\n", encoding="utf-8")
        self.assertNotEqual(self.run_policy(root).returncode, 0)

    def test_rejects_unpinned_actions(self) -> None:
        root = self.make_fixture()
        (root / ".github/workflows/build.yml").write_text(
            "jobs:\n  build:\n    runs-on: ubuntu-24.04\n    steps:\n      - uses: actions/checkout@v4\n",
            encoding="utf-8",
        )
        self.assertNotEqual(self.run_policy(root).returncode, 0)

    def test_rejects_floating_runner_families(self) -> None:
        root = self.make_fixture()
        (root / ".github/workflows/build.yml").write_text(
            "jobs:\n  build:\n    runs-on: ubuntu-latest\n    steps:\n"
            "      - uses: actions/checkout@11d5960a326750d5838078e36cf38b85af677262\n",
            encoding="utf-8",
        )
        self.assertNotEqual(self.run_policy(root).returncode, 0)


if __name__ == "__main__":
    unittest.main()
