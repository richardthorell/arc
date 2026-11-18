"""Build new external projects through the source editor's SDK handoff.

Uses only SDK components, so missing libraries/headers in component installs
cannot be hidden by a previous full installation. Does not launch any runtime.
"""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

tool, templates, version, sdk, configuration, cmake, *kinds = sys.argv[1:]
environment = os.environ.copy()
environment["ARC_DEVELOPMENT_SDK"] = sdk


def run(*args):
    result = subprocess.run([tool, *args, "--json"], env=environment,
                            capture_output=True, text=True, timeout=1800)
    assert result.returncode == 0, result.stdout + "\n" + result.stderr
    payload = json.loads(result.stdout)
    assert payload["succeeded"], payload


with tempfile.TemporaryDirectory(prefix="ARC custom SDK project ") as directory:
    for index, template in enumerate(kinds):
        name = "CustomGame" + str(index)
        root = Path(directory) / name
        run("create", "--name", name, "--destination", str(root), "--template", template,
            "--templates", templates, "--engine", version)
        run("build", "--project", str(root / (name + ".arcproject")),
            "--config", configuration, "--cmake", cmake)
        cache = (root / "Build" / "default" / "CMakeCache.txt").read_text(encoding="utf-8")
        expected = (Path(sdk) / configuration / "lib" / "cmake" / "ARC").as_posix()
        selected = [line.partition("=")[2] for line in cache.replace("\\", "/").splitlines()
                    if line.startswith("ARC_DIR:")]
        assert selected == [expected], selected
        print("Built", template, "through the development SDK")
