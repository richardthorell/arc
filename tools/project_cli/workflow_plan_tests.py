"""Exercise the installed CLI contract without building or modifying a project."""

import json
from pathlib import Path
import subprocess
import sys
import tempfile


tool, templates, engine = sys.argv[1:]
with tempfile.TemporaryDirectory(prefix="arc-workflow-") as temporary:
    root = Path(temporary) / "Game"
    subprocess.run(
        [tool, "create", "--name", "Game", "--destination", str(root),
         "--template", "blank-3d", "--templates", templates, "--engine", engine],
        check=True, capture_output=True, text=True,
    )
    descriptor = root / "Game.arcproject"
    before = descriptor.read_bytes()

    def plan(action, *extra):
        result = subprocess.run(
            [tool, "workflow", "plan", "--json", "--project", str(descriptor),
             "--profile", "windows-x64-vulkan", "--target", "windows-x64-vulkan",
             "--action", action, *extra],
            capture_output=True, text=True, check=False,
        )
        payload = json.loads(result.stdout)
        assert payload["succeeded"] == (result.returncode == 0), result
        return payload

    assert plan("cook")["stages"] == ["validate", "cook"]
    assert plan("buildCode")["stages"] == ["validate", "configure", "compile"]
    assert plan("build")["stages"] == ["validate", "configure", "compile", "cook", "package"]
    built = plan("runBuilt", "--destination", "local")
    assert built["stages"] == ["validate", "launch"]
    assert built["version"] == 1
    assert built["profile"] == "windows-x64-vulkan"
    assert not plan("build", "--destination", "local")["succeeded"]
    assert not plan("cook", "--target", "android-arm64-vulkan")["succeeded"]
    assert not plan("play", "--destination", "local", "--session-id", "../escape")["succeeded"]
    assert descriptor.read_bytes() == before
    assert not (root / "Build" / "Packages").exists()
    assert not (root / "Build" / "Cooked").exists()

print("Workflow CLI stage plans and read-only behavior passed")
