---
format: arc-skill
formatVersion: 1
id: material-authoring
name: Material Authoring
version: 1.0.0
description: Inspect, create, bind, and visually verify ARC materials through typed asset and scene-edit workflows.
requires:
  - scene.read
  - scene.mutate
  - asset.read
  - asset.mutate
  - viewport.read
tools:
  - assets.list
  - scene.getEntity
  - scene.componentSchemas
  - edit.request
  - edit.begin
  - edit.apply
  - edit.commit
  - edit.cancel
  - viewport.debug
contexts:
  - project
  - selection
  - assets
  - viewport
  - diagnostics
---

# Material Authoring

Inspect the target entity, component schema, and existing assets before authoring or binding a material. Prefer typed material definitions and validated project asset references over raw file assumptions.

All persistent changes must stay inside the editor harness approval and transaction flow described by the ARC Editor Gateway concepts. Verify the affected entity and rendered result before commit; cancel when the result is stale, incomplete, or visually incorrect. Skill instructions never substitute for edit approval.
