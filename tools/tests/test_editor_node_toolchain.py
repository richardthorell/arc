from __future__ import annotations

import pathlib
import sys
import unittest
from unittest import mock


REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools import arc_build


class EditorNodeToolchainTests(unittest.TestCase):
    def test_accepts_ci_node_and_npm_versions(self) -> None:
        self.assertTrue(arc_build.editor_node_toolchain_supported("v22.23.3", "10.9.9"))
        self.assertTrue(arc_build.editor_node_toolchain_supported("v22.23.7", "10.10.0"))

    def test_rejects_other_node_and_npm_lines(self) -> None:
        self.assertFalse(arc_build.editor_node_toolchain_supported("v24.13.0", "11.2.1"))
        self.assertFalse(arc_build.editor_node_toolchain_supported("v22.23.3", "11.0.0"))
        self.assertFalse(arc_build.editor_node_toolchain_supported("v22.12.0", "10.9.0"))
        self.assertFalse(arc_build.editor_node_toolchain_supported("v22.24.0", "10.9.9"))

    def test_prerequisite_check_reports_version_mismatch(self) -> None:
        def find_executable(name: str):
            return {"node": "node", "npm": "npm"}.get(name)

        def command_version(executable: str, _arguments=None):
            return {"node": "v24.13.0", "npm": "11.2.1"}[executable]

        with mock.patch.object(arc_build, "find_executable", side_effect=find_executable), mock.patch.object(
            arc_build, "command_version", side_effect=command_version
        ), mock.patch.object(arc_build.platform, "system", return_value="Windows"):
            checks = arc_build.check_editor_prerequisites(require_native=False)

        self.assertEqual(len(checks), 1)
        self.assertFalse(checks[0]["ok"])
        self.assertTrue(checks[0]["installable"])
        self.assertIn("v24.13.0 / npm 11.2.1", checks[0]["detail"])
        self.assertIn("requires Node.js 22.23.x / npm 10.x", checks[0]["detail"])

    def test_windows_installer_requests_the_pinned_node_version(self) -> None:
        checks = [
            {
                "key": "node",
                "name": "Node.js / npm",
                "ok": False,
                "detail": "unsupported",
                "installable": True,
            }
        ]

        with mock.patch.object(arc_build.platform, "system", return_value="Windows"), mock.patch.object(
            arc_build, "check_editor_prerequisites", return_value=checks
        ), mock.patch.object(arc_build, "find_executable", return_value="winget.exe"), mock.patch.object(
            arc_build, "run"
        ) as run:
            _, installed = arc_build.install_editor_prerequisites(input_fn=lambda _: "yes")

        self.assertTrue(installed)
        command = run.call_args.args[0]
        self.assertEqual(command[:4], ["winget.exe", "install", "--id", arc_build.WINDOWS_NODE_PACKAGE])
        self.assertIn("--version", command)
        self.assertEqual(command[command.index("--version") + 1], arc_build.EDITOR_NODE_VERSION)


if __name__ == "__main__":
    unittest.main()
