"""A5 workflow runner acceptance: shared plan, execution, rollback and cancellation."""
from __future__ import annotations

import argparse
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from tools import android_build_workflow as workflow


class A5WorkflowTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="a5-unit-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.project = self.root / "Game"
        self.project.mkdir()
        self.descriptor = self.project / "Game.arcproject"
        self.descriptor.write_text(json.dumps({
            "format": "arc-project", "guid": "a1111111-1111-4111-8111-111111111111",
            "engineVersion": "0.1.0",
            "cookProfiles": [{
                "id": "android-arm64-vulkan-debug",
                "textures": {"outputs": ["astc"]}, "configuration": "Debug",
            }],
            "startupScenes": [{"guid": "b80a0000-0000-4000-8000-000000000005"}],
        }))
        self.sdk = self.root / "SDK"
        gradle = self.sdk / "share/arc/android/gradle"
        (gradle / "app").mkdir(parents=True)
        (gradle / "settings.gradle.kts").write_text("rootProject.name = \"A5\"\n")
        (gradle / "app/build.gradle.kts").write_text("plugins {}\n")
        host = self.sdk / "share/arc/android/host"
        host.mkdir(parents=True)
        (host / "host.json").write_text("{}\n")
        cooker = self.sdk / "bin/arc-cook"
        cooker.parent.mkdir(parents=True)
        cooker.write_text("mock cooker\n")
        self.plan = {
            "succeeded": True,
            "stages": workflow.STAGES,
            "projectGuid": "a1111111-1111-4111-8111-111111111111",
            "engineVersion": "0.1.0",
            "platform": "android", "architecture": "arm64",
            "renderer": "vulkan", "configuration": "Debug",
            "finalizer": "android-gradle", "targetId": workflow.TARGET,
            "profile": workflow.DEFAULT_PROFILE,
            "cookOutput": str(self.project / "Build/Cooked/android-arm64-vulkan-debug"),
            "packageOutput": str(self.project / "Build/Packages/android-arm64-vulkan-debug/Runtime"),
        }
        self.settings = {
            "format": "arc.android-project-settings",
            "projectGuid": self.plan["projectGuid"],
            "applicationId": "com.example.game",
        }

    def args(self, run_id="run-one") -> argparse.Namespace:
        return argparse.Namespace(
            project=str(self.descriptor), sdk_root=str(self.sdk),
            project_tool="arc-project", cooker=None, profile=workflow.DEFAULT_PROFILE,
            target=workflow.TARGET, output=None, runtime_config=None, gradle="gradle",
            run_id=run_id, json=True, action="execute",
        )

    def runner(self, *, failure: str | None = None, cancel: bool = False):
        instance = workflow.Executor(self.args(), self.descriptor)
        commands: list[list[str]] = []

        def capture(argv):
            if argv[1:3] == ["workflow", "plan"]:
                return self.plan
            if argv[1] == "validate":
                return {"succeeded": True}
            if argv[1:3] == ["settings", "export-android"]:
                return self.settings
            raise AssertionError(argv)

        def command(argv, cwd=None):
            commands.append(argv)
            if failure and failure in argv:
                raise workflow.WorkflowError("Simulated failure: " + failure)
            if "package" in argv and argv[0] == instance.cooker:
                cooked = instance.scratch / "Cooked"
                cooked.mkdir()
                (cooked / "android-debug.arccookmanifest").write_text('{"cook":true}\n')
            if "android-content" in argv:
                assert "--cook-manifest" in argv
                assert "--runtime-config" in argv
                instance.asset_stage.mkdir()
                (instance.asset_stage / "arc-distribution.json").write_text("{}\n")
            if "android-apk" in argv:
                assert "--workspace-root" in argv
                assert argv[argv.index("--workspace-root") + 1] == str(instance.scratch)
                instance.output.mkdir(parents=True)
                (instance.output / "com.example.game-debug.apk").write_bytes(b"apk")
                (instance.output / "arc-apk-integrity.json").write_text('{"verified":true}\n')
            if cancel and ":app:externalNativeBuildDebug" in argv:
                instance.cancel_path.touch()

        instance.capture = capture
        instance.command = command
        return instance, commands

    def test_complete_five_stage_build_uses_shared_plan_and_one_gradle_workspace(self):
        runner, commands = self.runner()
        with redirect_stdout(io.StringIO()) as output:
            result = runner.execute()
        self.assertEqual(result, 0)
        state = workflow.read_json(runner.state_path)
        self.assertEqual(state["status"], "succeeded")
        self.assertEqual([stage["status"] for stage in state["stages"]], ["succeeded"] * 5)
        self.assertEqual(len([c for c in commands if ":app:externalNativeBuildDebug" in c]), 1)
        self.assertEqual(len([c for c in commands if "android-apk" in c]), 1)
        self.assertEqual(len([c for c in commands if "android-content" in c]), 1)
        self.assertEqual(len([c for c in commands if "verify" in c]), 1)
        self.assertEqual(state["projectGuid"], self.plan["projectGuid"])
        self.assertEqual(state["targetId"], workflow.TARGET)
        self.assertTrue(Path(state["apk"]).is_file())
        self.assertFalse(any(p.name.startswith(".a5-") for p in (self.project / "Build").iterdir()))
        self.assertIn('"event": "workflow.complete"', output.getvalue())
        self.assertIn('"event": "workflow.stage"', output.getvalue())
        with self.assertRaisesRegex(workflow.WorkflowError, "already exists"):
            runner.execute()

    def test_compile_failure_short_circuits_cook_and_package(self):
        runner, commands = self.runner(failure=":app:externalNativeBuildDebug")
        with redirect_stdout(io.StringIO()):
            self.assertEqual(runner.execute(), 1)
        state = workflow.read_json(runner.state_path)
        self.assertEqual(state["status"], "failed")
        self.assertEqual(state["stages"][2]["status"], "failed")
        self.assertEqual([s["status"] for s in state["stages"][3:]], ["pending", "pending"])
        self.assertFalse(any("android-apk" in c or "android-content" in c for c in commands))
        self.assertFalse((self.project / "Build/Packages").exists())

    def test_cancel_terminates_before_cooking_or_publishing(self):
        runner, commands = self.runner(cancel=True)
        with redirect_stdout(io.StringIO()):
            self.assertEqual(runner.execute(), 130)
        state = workflow.read_json(runner.state_path)
        self.assertEqual(state["status"], "cancelled")
        self.assertEqual(state["stages"][2]["status"], "cancelled")
        self.assertFalse(any("android-apk" in c for c in commands))

    def test_status_logs_and_request_cancel(self):
        runner, _ = self.runner()
        runner.store.mkdir(parents=True)
        runner.log_path.write_text("first\nsecond\n")
        workflow.write_atomic(runner.state_path, runner.status)
        args = self.args()
        args.action = "status"
        with redirect_stdout(io.StringIO()) as output:
            self.assertEqual(workflow.control(args), 0)
        self.assertEqual(json.loads(output.getvalue())["status"], "running")
        args.action = "logs"
        with redirect_stdout(io.StringIO()) as output:
            self.assertEqual(workflow.control(args), 0)
        self.assertIn("second", json.loads(output.getvalue())["log"])
        args.action = "cancel"
        with redirect_stdout(io.StringIO()) as output:
            self.assertEqual(workflow.control(args), 0)
        self.assertTrue(json.loads(output.getvalue())["cancellationRequested"])
        self.assertTrue(runner.cancel_path.exists())
        runner.status["status"] = "succeeded"
        workflow.write_atomic(runner.state_path, runner.status)
        with self.assertRaisesRegex(workflow.WorkflowError, "Only running builds"):
            workflow.control(args)

    def test_rejects_shipping_profile_and_wrong_target_before_tools(self):
        runner, commands = self.runner()
        runner.plan = dict(self.plan)
        self.plan["configuration"] = "Shipping"
        with redirect_stdout(io.StringIO()):
            self.assertEqual(runner.execute(), 1)
        self.assertEqual(workflow.read_json(runner.state_path)["status"], "failed")
        self.assertEqual(commands, [])

    def test_run_id_rejects_directory_traversal(self):
        for value in ("../escape", "/", "", " spaces", "a" * 65):
            with self.assertRaises(workflow.WorkflowError):
                workflow.safe_run_id(value)

    def test_process_failure_writes_log_and_throws(self):
        runner, _ = self.runner()
        runner.store.mkdir(parents=True)
        runner.log_path.touch()
        with mock.patch.object(workflow.subprocess, "Popen") as popen:
            process = popen.return_value
            process.poll.return_value = 2
            process.returncode = 2
            with self.assertRaisesRegex(workflow.WorkflowError, "exited 2"):
                workflow.Executor.command(runner, ["fake-command"])
        self.assertIn("fake-command", runner.log_path.read_text())


if __name__ == "__main__":
    unittest.main()
