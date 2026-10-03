---
format: arc-skill
formatVersion: 1
id: terrain-workflows
name: Terrain Workflows
version: 1.0.0
description: Inspect terrain state, diagnose terrain rendering, and perform supported terrain edits through the ARC harness.
requires:
  - scene.read
  - scene.mutate
  - viewport.read
  - viewport.control
  - diagnostics.read
tools:
  - scene.getEntity
  - scene.componentSchemas
  - scene.spatialQuery
  - viewport.state
  - viewport.debug
  - diagnostics.get
  - edit.request
  - edit.begin
  - edit.apply
  - edit.commit
  - edit.cancel
contexts:
  - project
  - scene
  - selection
  - viewport
  - diagnostics
---

# Terrain Workflows

Inspect the terrain entity, reflected terrain component schema, camera state, and relevant renderer diagnostics before changing terrain settings. Use viewport debug captures to separate geometry, material, shadow, and environment problems.

Only use terrain mutations currently projected by the editor harness. Preserve explicit approval, expected scene revisions, and transaction verification for every persistent edit. Do not interpret this skill as permission to run scripts, edit arbitrary project files, or bypass unsupported terrain operations.
