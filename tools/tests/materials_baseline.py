#!/usr/bin/env python3
"""Structural, dependency-free validation for ARC Materials Baseline (#943)."""
import argparse
import json
from pathlib import Path

SCENE = Path("assets/fixtures/materials_baseline.arcscene")
GROUPS = ["01 / Core Material Families","02 / Standard Lit Response","03 / Composition, Textures & Emission"]
SPECIMENS = ["Standard Lit","Unlit","Glass / Forward","Subsurface","Dielectric / Satin","Metal / Polished","Metal / Rough","Clear Coat","Floor / Default Checker","Checker / Blue","Unlit / Textured","Emissive"]
PRIMITIVES = {"Plane", "Cube", "Sphere", "Cylinder", "Cone", "Capsule"}


def validate(root: Path) -> None:
    def require(ok, reason):
        if not ok:
            raise AssertionError(reason)

    def read(path: Path):
        require((root / path).is_file(), f"Missing asset: {path}")
        return json.loads((root / path).read_text(encoding="utf-8"))

    scene = read(SCENE)
    project = read(Path("MaterialsBaseline.arcproject"))
    meta = read(Path(str(SCENE) + ".arcmeta"))
    require(scene["format"] == "arc.scene" and scene["formatVersion"] == 3, "Unexpected scene serialization format")
    require(scene["scene"]["id"] == "c9430000-0000-4000-8000-000000000001", "Baseline scene GUID changed")
    require(meta["guid"] == "c9430000-0000-4000-8000-000000000002", "Baseline .arcmeta GUID changed")
    require(project["defaultScene"]["guid"] == meta["guid"], "Default scene GUID mismatch")
    require(project["defaultScene"]["pathHint"] == SCENE.as_posix(), "Default scene path mismatch")
    require(project["assetRoots"] == ["assets"], "Baseline project asset roots drifted")
    entities = scene["entities"]
    by_id = {entity["id"]: entity for entity in entities}
    require(len(entities) == len(by_id), "Duplicate entity GUID")
    require(all(i == e["order"] for i, e in enumerate(entities)), "Entity order changed")
    require(all(e["parent"] is None or e["parent"] in by_id for e in entities), "Dangling entity parent")
    by_name = {e["components"]["Name"]["value"]: e for e in entities}
    for group in GROUPS:
        require(group in by_name, f"Missing material group: {group}")
    camera = by_name["Baseline Camera"]["components"]
    require(camera["Transform"]["position"] == [0, 6.4, 17.5], "Baseline camera moved")
    require(camera["Camera"]["exposure"]["mode"] == 0, "Camera exposure must be manual")
    environment = by_name["Fixed World Environment"]["components"]["WorldEnvironment"]
    require(environment["skySource"] == "solidColor" and environment["timeMode"] == "fixed", "Environment is not fixed")
    for key in ("cloudsEnabled", "fogEnabled", "starsEnabled", "moonEnabled", "playing"):
        require(environment[key] is False, f"Nondeterministic environment property: {key}")
    require("DirectionalLight" in by_name["Key Directional Light"]["components"], "Fixed directional light missing")
    bound = {}
    for name in SPECIMENS + ["Ground / Floor"]:
        require(name in by_name, f"Missing material specimen: {name}")
        entity = by_name[name]
        binding = entity.get("assetBinding", {})
        require(binding.get("kind") == "primitive" and binding.get("subresource") in PRIMITIVES,
                f"{name}: invalid primitive binding")
        ref = binding["material"]
        asset_path = Path(ref["pathHint"])
        require(not asset_path.is_absolute() and ".." not in asset_path.parts,
                f"{name}: material path escapes project")
        asset = read(asset_path)
        metadata = read(Path(str(asset_path) + ".arcmeta"))
        require(metadata["guid"] == ref["guid"], f"{name}: stale material GUID")
        require(metadata["type"] == ref["expectedType"], f"{name}: incorrect asset type")
        require(asset.get("version", 0) >= 1, f"{name}: invalid material document")
        bound[entity["id"]] = (ref["guid"], ref["pathHint"], name)
    require(len(bound) == len(SPECIMENS) + 1, "Material specimen count mismatch")
    for dependency in scene["dependencies"]:
        require(dependency["ownerEntity"] in bound, "Orphaned material dependency")
        guid, path, name = bound[dependency["ownerEntity"]]
        require((guid, path) == (dependency["guid"], dependency["pathHint"]),
                f"{name}: scene material dependency mismatch")
    require(len(scene["dependencies"]) == len(bound), "Missing or duplicate material dependencies")
    print(f"Validated {len(bound)} Materials Baseline assignments and deterministic scene inputs")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    args = parser.parse_args()
    validate(args.root.resolve())
