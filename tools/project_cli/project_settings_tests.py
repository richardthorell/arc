"""Project settings native validation/export; no SDK, device or GPU required."""
import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile

tool, templates, version = sys.argv[1:]


def run(*args, success=True):
    result = subprocess.run([tool, *args, "--json"], capture_output=True, text=True, check=False)
    assert (result.returncode == 0) == success, result.stdout + result.stderr
    return json.loads(result.stdout)


with tempfile.TemporaryDirectory(prefix="arc project settings ") as temporary:
    root = Path(temporary)
    run("create", "--name", "Game", "--destination", str(root), "--template", "blank-3d",
        "--templates", templates, "--engine", version)
    descriptor = root / "Game.arcproject"
    original = json.loads(descriptor.read_text(encoding="utf-8"))
    capabilities = run("settings", "describe")
    assert "android-arm64-vulkan" in capabilities["targets"]
    assert capabilities["androidMinimumSdk"] == 26
    original.pop("platformSettings", None)
    descriptor.write_text(json.dumps(original), encoding="utf-8")
    run("validate", "--project", str(descriptor))
    run("settings", "export-android", "--project", str(descriptor), success=False)
    valid = copy.deepcopy(original)
    valid["package"]["applicationName"] = "Game & Friends"
    valid["platformSettings"] = {"android": {
        "applicationId": "com.example.game", "versionName": "2.3", "versionCode": 12,
        "minimumSdk": 26, "targetSdk": 35,
    }}
    descriptor.write_text(json.dumps(valid), encoding="utf-8")
    before = descriptor.read_bytes()
    exported = run("settings", "export-android", "--project", str(descriptor))
    assert exported["applicationId"] == "com.example.game"
    assert exported["displayName"] == "Game & Friends"
    assert exported["projectGuid"] == original["guid"]
    assert exported["versionCode"] == 12 and exported["versionName"] == "2.3"
    assert exported["minimumSdk"] == 26 and exported["targetSdk"] == 35
    assert descriptor.read_bytes() == before
    for key, value in [("applicationId", "invalid"), ("applicationId", "com.1invalid.game"),
                       ("versionCode", 0), ("versionCode", 4294967297), ("versionCode", 1.5),
                       ("minimumSdk", 25), ("minimumSdk", 36), ("targetSdk", 25),
                       ("targetSdk", 36), ("versionName", "")]:
        invalid = copy.deepcopy(valid)
        invalid["platformSettings"]["android"][key] = value
        descriptor.write_text(json.dumps(invalid), encoding="utf-8")
        run("validate", "--project", str(descriptor), success=False)
    invalid = copy.deepcopy(valid)
    invalid["package"]["output"] = "../outside"
    descriptor.write_text(json.dumps(invalid), encoding="utf-8")
    run("validate", "--project", str(descriptor), success=False)
    invalid = copy.deepcopy(valid)
    invalid["cookProfiles"][0]["platform"] = "nonexistent"
    descriptor.write_text(json.dumps(invalid), encoding="utf-8")
    run("validate", "--project", str(descriptor), success=False)
    valid["targetPlatforms"][1]["enabled"] = False
    descriptor.write_text(json.dumps(valid), encoding="utf-8")
    run("validate", "--project", str(descriptor))
