---
format: arc-skill
formatVersion: 1
id: scene-inspection
name: Scene Inspection
version: 1.0.0
description: Inspect ARC scene structure, components, spatial relationships, selection, and recent scene changes before drawing conclusions.
requires:
  - scene.read
  - asset.read
tools:
  - scene.overview
  - scene.findEntities
  - scene.getEntity
  - scene.componentSchemas
  - scene.spatialQuery
  - scene.changes
  - assets.list
contexts:
  - project
  - scene
  - selection
  - assets
  - recentChanges
---

# Scene Inspection

Use persistent entity GUIDs for identity. Start from the current scene overview and selection, then narrow with entity search, component schemas, and spatial queries rather than guessing from names alone.

Treat scene and world revisions as authoritative. Re-read stale state before making conclusions that depend on entity existence, hierarchy, components, or assets. This skill only describes a read workflow; it does not grant scene access or mutation authority.
