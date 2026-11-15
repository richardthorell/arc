"""Verify the Gradle project-export handoff without building a second APK."""
import json
from pathlib import Path
import subprocess
import tempfile

android = Path(__file__).resolve().parents[1] / "engine/platform/android"
settings = {
    "format": "arc.android-project-settings", "version": 1,
    "projectGuid": "12345678-1234-4234-8234-123456789abc",
    "applicationId": "com.example.arcgame", "displayName": "ARC Test & Friends",
    "versionName": "2.3", "versionCode": 12, "minimumSdk": 26, "targetSdk": 35,
}
with tempfile.TemporaryDirectory(prefix="arc-android-settings-") as directory:
    export = Path(directory) / "settings.json"

    def verify(success):
        export.write_text(json.dumps(settings), encoding="utf-8")
        result = subprocess.run(
            ["gradle", "--no-daemon", ":app:verifyArcProjectSettings",
             f"-ParcProjectSettings={export}"], cwd=android, check=False, timeout=180,
        )
        assert (result.returncode == 0) == success

    verify(True)
    settings["applicationId"] = ""
    verify(False)
    settings["applicationId"] = "com.example.arcgame"
    settings["targetSdk"] = 25
    verify(False)
