#!/usr/bin/env python3
"""ARC Android A5 synchronous workflow executor.

The plan comes exclusively from arc-project workflow plan (#1132/#1163).
Each execution is owned by its invoking CLI process; status, logs and
cancellation operate on its durable Saved/Builds/android/<run-id> record.
No daemon, editor-specific orchestration, or implicit background work.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import uuid

STAGES = ["validate", "configure", "compile", "cook", "package"]
RUN_ID = re.compile(r"[A-Za-z0-9-]{1,64}\Z")
TARGET = "android-arm64-vulkan"


class WorkflowError(RuntimeError):
    pass


class WorkflowCancelled(WorkflowError):
    pass


def timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def ensure(condition: bool, message: str) -> None:
    if not condition:
        raise WorkflowError(message)


def descriptor_path(arg: str) -> Path:
    path = Path(arg).expanduser().absolute()
    if path.is_dir():
        matches = list(path.glob("*.arcproject"))
        ensure(len(matches) == 1, "Expected one .arcproject in the selected directory")
        path = matches[0]
    ensure(path.is_file() and not path.is_symlink(), "Project descriptor is missing or symbolic")
    return path.resolve(strict=True)


def read_json(path: Path) -> dict:
    try:
        result = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, ValueError) as exc:
        raise WorkflowError(f"Cannot read {path}: {exc}") from exc
    ensure(isinstance(result, dict), f"Invalid document: {path}")
    return result


def safe_run_id(value: str) -> str:
    ensure(bool(RUN_ID.fullmatch(value)), "Run ID must be 1–64 alphanumeric or hyphen characters")
    return value


def state_root(project: Path, run_id: str) -> Path:
    # Do not use arbitrary source inputs as directory names.
    return project.parent / "Saved" / "Builds" / "android" / safe_run_id(run_id)


def write_atomic(path: Path, payload: dict) -> None:
    temporary = path.with_name(path.name + ".pending")
    with temporary.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def emit(as_json: bool, event: str, **fields: object) -> None:
    if as_json:
        print(json.dumps({"event": event, **fields}, sort_keys=True), flush=True)
    elif event == "workflow.stage":
        print(f"[{fields['status']}] {fields['stage']}", flush=True)
    elif event == "workflow.complete":
        print(f"Android APK built: {fields['apk']}", flush=True)
        print(f"Build log: {fields['log']}", flush=True)
    elif event == "workflow.failure":
        print(f"Android build {fields['status']}: {fields['error']}", file=sys.stderr, flush=True)


def tail(path: Path, limit: int = 8192) -> str:
    if not path.is_file():
        return ""
    with path.open("rb") as stream:
        stream.seek(0, os.SEEK_END)
        stream.seek(max(0, stream.tell() - limit))
        return stream.read().decode("utf-8", errors="replace")


class Executor:
    def __init__(self, args: argparse.Namespace, project: Path):
        self.args = args
        self.project = project
        self.root = project.parent
        self.run_id = safe_run_id(args.run_id or uuid.uuid4().hex)
        self.store = state_root(project, self.run_id)
        self.log_path = self.store / "build.log"
        self.state_path = self.store / "status.json"
        self.cancel_path = self.store / "cancel.request"
        self.scratch: Path | None = None
        self.gradle_root: Path | None = None
        self.settings_path: Path | None = None
        self.cooked_manifest: Path | None = None
        self.asset_stage: Path | None = None
        self.output: Path | None = None
        self.sdk = Path(args.sdk_root).expanduser().absolute()
        self.project_tool = args.project_tool
        self.cooker = args.cooker or str(self.sdk / "bin" / ("arc-cook.exe" if os.name == "nt" else "arc-cook"))
        self.status = {
            "format": "arc.workflow-run",
            "formatVersion": 1,
            "runId": self.run_id,
            "project": str(project),
            "status": "running",
            "stage": "validate",
            "startedAt": timestamp(),
            "updatedAt": timestamp(),
            "log": str(self.log_path),
            "stages": [{"name": name, "status": "pending"} for name in STAGES],
        }
        self.cancelled_signal = False

    def cancelled(self) -> bool:
        return self.cancelled_signal or self.cancel_path.exists()

    def on_signal(self, signum, _frame) -> None:
        self.cancelled_signal = True

    def update(self, stage: str, status: str, error: str | None = None) -> None:
        self.status["stage"] = stage
        self.status["updatedAt"] = timestamp()
        for item in self.status["stages"]:
            if item["name"] == stage:
                item["status"] = status
                item["endedAt" if status != "running" else "startedAt"] = timestamp()
        if error:
            self.status["error"] = error
        write_atomic(self.state_path, self.status)
        emit(self.args.json, "workflow.stage", runId=self.run_id, stage=stage, status=status)

    def step(self, name: str, callback) -> None:
        if self.cancelled():
            raise WorkflowCancelled("Build cancelled")
        self.update(name, "running")
        try:
            callback()
            if self.cancelled():
                raise WorkflowCancelled("Build cancelled")
        except BaseException:
            self.update(name, "cancelled" if self.cancelled() else "failed")
            raise
        self.update(name, "succeeded")

    def command(self, argv: list[str], cwd: Path | None = None) -> None:
        if self.cancelled():
            raise WorkflowCancelled("Build cancelled")
        with self.log_path.open("a", encoding="utf-8") as log:
            log.write("+ " + json.dumps(argv) + "\n")
            log.flush()
            flags = subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0
            proc = subprocess.Popen(
                argv, cwd=str(cwd or self.root), stdout=log, stderr=subprocess.STDOUT,
                creationflags=flags, start_new_session=os.name != "nt",
            )
            while proc.poll() is None:
                if self.cancelled():
                    if os.name == "nt":
                        proc.terminate()
                    else:
                        os.killpg(proc.pid, signal.SIGTERM)
                    try:
                        proc.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        if os.name == "nt":
                            proc.kill()
                        else:
                            os.killpg(proc.pid, signal.SIGKILL)
                        proc.wait()
                    raise WorkflowCancelled("Build cancelled")
                time.sleep(0.2)
            if proc.returncode != 0:
                raise WorkflowError(f"{Path(argv[0]).name} exited {proc.returncode}. See {self.log_path}")

    def capture(self, argv: list[str]) -> dict:
        if self.cancelled():
            raise WorkflowCancelled("Build cancelled")
        result = subprocess.run(argv, cwd=self.root, capture_output=True, text=True, check=False)
        with self.log_path.open("a", encoding="utf-8") as log:
            log.write("+ " + json.dumps(argv) + "\n")
            log.write(result.stdout + result.stderr + "\n")
        ensure(result.returncode == 0,
               f"ARC project validation failed: {result.stderr.strip() or result.stdout.strip()}")
        try:
            payload = json.loads(result.stdout)
        except ValueError as exc:
            raise WorkflowError("ARC planner/settings returned non-JSON output") from exc
        ensure(isinstance(payload, dict), "ARC tool returned non-object output")
        return payload

    def validate(self) -> None:
        ensure(self.sdk.is_absolute() and self.sdk.is_dir(),
               "Installed ARC SDK root must be an existing directory")
        ensure((self.sdk / "share/arc/android/host/host.json").is_file(),
               "A1/A2 Android host SDK export is missing")
        ensure((self.sdk / "share/arc/android/gradle/app/build.gradle.kts").is_file(),
               "Android SDK Gradle launcher is missing")
        ensure(Path(self.cooker).is_file() or shutil.which(self.cooker),
               f"Host arc-cook executable unavailable: {self.cooker}")
        plan = self.capture([self.project_tool, "workflow", "plan",
                             "--project", str(self.project), "--profile", self.args.profile,
                             "--target", self.args.target, "--action", "build", "--json"])
        ensure(plan.get("succeeded") is True and plan.get("stages") == STAGES,
               "ARC shared workflow planner returned incompatible stages")
        ensure(plan.get("platform") == "android" and plan.get("architecture") == "arm64" and
               plan.get("renderer") == "vulkan" and plan.get("configuration") == "Debug" and
               plan.get("finalizer") == "android-gradle" and plan.get("targetId") == TARGET,
               "A5 currently supports Android ARM64 Vulkan Debug APKs only")
        descriptor = read_json(self.project)
        ensure(descriptor.get("guid") == plan["projectGuid"] and
               descriptor.get("engineVersion") == plan["engineVersion"],
               "ARC shared plan differs from project descriptor")
        profiles = [p for p in descriptor.get("cookProfiles", []) if p.get("id") == self.args.profile]
        ensure(len(profiles) == 1 and profiles[0].get("textures", {}).get("outputs") == ["astc"],
               "Android A5 supports exactly one ASTC cook texture output")
        ensure(descriptor.get("startupScenes"), "Android A5 requires project startup scenes")
        self.capture([self.project_tool, "validate", "--project", str(self.project), "--json"])
        exported = self.capture([self.project_tool, "settings", "export-android",
                                 "--project", str(self.project), "--json"])
        ensure(exported.get("projectGuid") == plan["projectGuid"] and
               exported.get("format") == "arc.android-project-settings",
               "ARC Android settings export does not match project identity")
        self.status["projectGuid"] = plan["projectGuid"]
        self.status["engineVersion"] = plan["engineVersion"]
        self.status["profile"] = plan["profile"]
        self.status["targetId"] = plan["targetId"]
        self.output = Path(self.args.output).expanduser().absolute() if self.args.output else (
            Path(plan["packageOutput"]) / self.run_id
        )
        ensure(not self.output.exists(), "APK output directory already exists")
        self.status["output"] = str(self.output)
        self.plan = plan
        self.exported = exported

    def gradle(self, task: str, *, content: Path | None = None) -> None:
        ensure(self.gradle_root is not None and self.settings_path is not None, "Gradle workspace not configured")
        cmd = [self.args.gradle, "--no-daemon", "--project-dir", str(self.gradle_root),
               f"-ParcSdkRoot={self.sdk}", f"-ParcProjectRoot={self.root}",
               f"-ParcProjectSettings={self.settings_path}"]
        if content is not None:
            cmd.append(f"-ParcDistributionAssets={content}")
        self.command(cmd + [task, "--stacktrace"], cwd=self.gradle_root)

    def configure(self) -> None:
        ensure(self.scratch is not None, "Missing scratch workspace")
        source = self.sdk / "share/arc/android/gradle"
        self.gradle_root = self.scratch / "gradle"
        shutil.copytree(source, self.gradle_root, ignore=shutil.ignore_patterns("build", ".cxx", ".gradle"))
        self.settings_path = self.scratch / "android-project-settings.json"
        self.settings_path.write_text(json.dumps(self.exported, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        self.gradle(":app:help")

    def compile(self) -> None:
        # This is an actual A2 Android NDK compile, not a desktop CMake build.
        # The same private Gradle workspace is reused by A4 in package().
        self.gradle(":app:externalNativeBuildDebug")

    def cook(self) -> None:
        ensure(self.scratch is not None, "Missing cook workspace")
        target = self.scratch / "Cooked"
        self.command([self.cooker, "package", "--project", str(self.root),
                      "--profile", self.args.profile, "--output", str(target), "--json"])
        self.command([self.cooker, "verify", "--project", str(self.root),
                      "--profile", self.args.profile, "--output", str(target), "--json"])
        manifests = list(target.rglob("*.arccookmanifest"))
        ensure(len(manifests) == 1, "Android A5 requires exactly one verified ASTC cook manifest")
        self.cooked_manifest = manifests[0]

    def runtime_configuration(self) -> Path:
        ensure(self.scratch is not None, "Missing workspace")
        if self.args.runtime_config:
            source = Path(self.args.runtime_config).expanduser()
            if not source.is_absolute():
                source = self.root / source
            ensure(source.is_file(), f"Explicit runtime configuration is missing: {source}")
            return source.resolve()
        default = self.root / "Config" / "runtime.json"
        if default.is_file():
            return default
        generated = self.scratch / "runtime.json"
        generated.write_text(json.dumps({
            "format": "arc.runtime-config", "formatVersion": 1,
            "projectGuid": self.plan["projectGuid"], "tickRate": 60,
        }, sort_keys=True) + "\n", encoding="utf-8")
        return generated

    def package(self) -> None:
        ensure(self.scratch is not None and self.cooked_manifest is not None and self.output is not None,
               "Android cook stage is incomplete")
        self.asset_stage = self.scratch / "Assets"
        self.command([self.project_tool, "package", "android-content",
                      "--project", str(self.project), "--profile", self.args.profile,
                      "--target", self.args.target, "--cook-manifest", str(self.cooked_manifest),
                      "--runtime-config", str(self.runtime_configuration()),
                      "--output", str(self.asset_stage), "--json"])
        self.command([self.project_tool, "package", "android-apk",
                      "--project", str(self.project), "--sdk-root", str(self.sdk),
                      "--assets", str(self.asset_stage), "--output", str(self.output),
                      "--settings", str(self.settings_path), "--gradle", self.args.gradle,
                      "--workspace-root", str(self.scratch), "--json"])
        apk_files = list(self.output.glob("*.apk"))
        report = self.output / "arc-apk-integrity.json"
        ensure(len(apk_files) == 1 and report.is_file(),
               "A4 did not publish a complete, verified Android APK")
        self.status["apk"] = str(apk_files[0])
        self.status["integrity"] = str(report)

    def execute(self) -> int:
        ensure(not self.store.exists(), "Requested Android build run ID already exists")
        self.store.mkdir(parents=True)
        self.log_path.touch(exist_ok=False)
        write_atomic(self.state_path, self.status)
        emit(self.args.json, "workflow.started", runId=self.run_id, log=str(self.log_path))
        previous_int = signal.getsignal(signal.SIGINT)
        previous_term = signal.getsignal(signal.SIGTERM)
        signal.signal(signal.SIGINT, self.on_signal)
        signal.signal(signal.SIGTERM, self.on_signal)
        try:
            self.step("validate", self.validate)
            base = Path(self.plan["cookOutput"]).parent.parent
            base.mkdir(parents=True, exist_ok=True)
            with tempfile.TemporaryDirectory(prefix=".a5-" + self.run_id + "-", dir=base) as scratch:
                self.scratch = Path(scratch)
                self.step("configure", self.configure)
                self.step("compile", self.compile)
                self.step("cook", self.cook)
                self.step("package", self.package)
            self.status["status"] = "succeeded"
            self.status["completedAt"] = timestamp()
            write_atomic(self.state_path, self.status)
            emit(self.args.json, "workflow.complete", succeeded=True, runId=self.run_id,
                 apk=self.status["apk"], log=str(self.log_path), status="succeeded")
            return 0
        except (WorkflowCancelled, KeyboardInterrupt) as exc:
            self.status["status"] = "cancelled"
            self.status["error"] = str(exc) or "Build cancelled"
            self.status["completedAt"] = timestamp()
            write_atomic(self.state_path, self.status)
            emit(self.args.json, "workflow.failure", succeeded=False, runId=self.run_id,
                 status="cancelled", error=self.status["error"], log=str(self.log_path))
            return 130
        except (WorkflowError, OSError, subprocess.SubprocessError, ValueError) as exc:
            self.status["status"] = "failed"
            self.status["error"] = str(exc)
            self.status["completedAt"] = timestamp()
            write_atomic(self.state_path, self.status)
            emit(self.args.json, "workflow.failure", succeeded=False, runId=self.run_id,
                 status="failed", error=str(exc), log=str(self.log_path))
            return 1
        finally:
            signal.signal(signal.SIGINT, previous_int)
            signal.signal(signal.SIGTERM, previous_term)


def control(args: argparse.Namespace) -> int:
    project = descriptor_path(args.project)
    ensure(bool(args.run_id), "--run-id is required for status/logs/cancel")
    store = state_root(project, args.run_id)
    status_file = store / "status.json"
    ensure(status_file.is_file(), f"Android build run not found: {args.run_id}")
    record = read_json(status_file)
    ensure(record.get("format") == "arc.workflow-run" and record.get("runId") == args.run_id and
           record.get("project") == str(project), "Invalid Android workflow status record")
    if args.action == "logs":
        payload = tail(store / "build.log", 128 * 1024)
        if args.json:
            print(json.dumps({"runId": args.run_id, "log": payload}))
        else:
            print(payload, end="")
    elif args.action == "cancel":
        ensure(record.get("status") == "running", "Only running builds can be cancelled")
        flag = store / "cancel.request"
        if not flag.exists():
            flag.touch(exist_ok=False)
        response = {"runId": args.run_id, "cancellationRequested": True}
        print(json.dumps(response) if args.json else f"Cancellation requested for {args.run_id}")
    else:
        print(json.dumps(record, sort_keys=True) if args.json else
              f"{record['runId']}: {record['status']} ({record['stage']})\nLog: {record['log']}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("execute", "status", "logs", "cancel"))
    parser.add_argument("--project", required=True)
    parser.add_argument("--sdk-root")
    parser.add_argument("--project-tool", default="arc-project")
    parser.add_argument("--cooker")
    parser.add_argument("--profile", default=TARGET)
    parser.add_argument("--target", default=TARGET)
    parser.add_argument("--output")
    parser.add_argument("--runtime-config")
    parser.add_argument("--gradle", default="gradle")
    parser.add_argument("--run-id")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    try:
        if args.action == "execute":
            ensure(bool(args.sdk_root), "Android build requires --sdk-root")
            return Executor(args, descriptor_path(args.project)).execute()
        return control(args)
    except (WorkflowError, OSError, ValueError) as exc:
        if args.json:
            print(json.dumps({"succeeded": False, "error": str(exc)}))
        else:
            print(f"ARC Android workflow: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
