---
format: arc-skill
formatVersion: 1
id: renderer-diagnostics
name: Renderer Diagnostics
version: 1.0.0
description: Diagnose ARC viewport and renderer problems with coherent captures, debug visualizations, pixel inspection, and comparisons.
requires:
  - scene.read
  - viewport.read
  - viewport.control
  - diagnostics.read
tools:
  - viewport.state
  - viewport.move
  - viewport.setRenderOptions
  - viewport.observe
  - viewport.debug
  - viewport.inspectPixel
  - viewport.compare
  - diagnostics.get
  - events.wait
contexts:
  - scene
  - selection
  - viewport
  - diagnostics
  - recentChanges
---

# Renderer Diagnostics

Follow the ARC Editor Gateway diagnostic workflow: read viewport state, then prefer one coherent `viewport.debug` request over unrelated captures and toggles. For dark, clipped, missing, or incorrect frames, inspect color, depth, object ID, and normals first; add material or scene-color channels when needed.

Treat requested/effective render-option mismatches and capture anomalies as evidence. Use pixel inspection for exact values and capture comparison for regressions. Viewport controls are temporary diagnostic capabilities and do not authorize persistent scene edits.
