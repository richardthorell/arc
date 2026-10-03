---
format: arc-skill
formatVersion: 1
id: play-workflows
name: Play Workflows
version: 1.0.0
description: Observe and control ARC Play sessions when the active editor harness advertises runtime Play capabilities.
requires:
  - scene.read
  - diagnostics.read
  - play.control
tools:
  - play.status
  - play.control
  - events.wait
  - scene.changes
  - diagnostics.get
contexts:
  - project
  - scene
  - selection
  - diagnostics
  - recentChanges
---

# Play Workflows

Use this skill only when the active harness advertises the required Play capability and tools. Read Play state before issuing control actions, preserve valid state transitions, and use event-driven waits rather than aggressive polling while runtime state changes.

After play, pause, step, resume, or stop operations, inspect resulting state and diagnostics before making conclusions about gameplay behavior. Declaring Play tools here does not make unavailable runtime controls callable and does not bypass harness validation.
