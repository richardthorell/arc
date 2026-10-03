---
format: arc-skill
formatVersion: 1
id: flow-authoring
name: Flow Authoring
version: 1.0.0
description: Inspect, create, assign, and verify ARC Flow gameplay graphs using typed assets and approved scene edits.
requires:
  - scene.read
  - scene.mutate
  - asset.read
  - asset.mutate
tools:
  - assets.list
  - scene.getEntity
  - scene.componentSchemas
  - edit.request
  - edit.begin
  - edit.apply
  - edit.commit
  - edit.cancel
  - scene.changes
contexts:
  - project
  - scene
  - selection
  - assets
  - recentChanges
---

# Flow Authoring

Resolve the target entity and current Flow binding first. When creating a Flow asset, use the typed harness asset workflow and a complete versioned graph definition; do not invent arbitrary filesystem writes.

Apply bindings or component changes only inside an approved edit transaction. Read the changed entity back and confirm the scene revision before commit. Cancel on stale state or failed verification. The skill declares requirements but cannot grant mutation authority.
