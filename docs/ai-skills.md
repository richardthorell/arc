# AI skills

ARC AI skills are versioned Markdown instruction packages. They describe a workflow, the editor context it may need, and the capabilities and tools that must already be available. A skill is **not** a permission grant.

The built-in catalog is shipped from `editor/resources/ai-skills`. An active project may add repository-local skills under `.agents/skills/<skill-id>/SKILL.md`. Project skills are loaded only for that active project and cannot shadow a built-in skill with the same ID.

## Format

The `SKILL.md` convention intentionally follows the existing `.agents/skills/arc-editor-gateway` shape: YAML-style front matter followed by Markdown instructions. ARC adds explicit versioned runtime metadata so the editor can validate skills deterministically.

```markdown
---
format: arc-skill
formatVersion: 1
id: lighting-review
name: Lighting Review
version: 1.0.0
description: Inspect lighting and shadow behavior.
requires:
  - scene.read
  - viewport.read
tools:
  - scene.overview
  - viewport.debug
contexts:
  - scene
  - viewport
  - diagnostics
---

# Lighting Review

Inspect the scene and use coherent viewport debug captures before changing anything.
```

`id` must match the containing directory. `version` is the skill's semantic version; `formatVersion` versions ARC's metadata contract. `requires` contains capability requirements, `tools` names the operations the workflow may request, and `contexts` names project-context sections that may be useful to the workflow.

Version 1 accepts these capability requirements: `scene.read`, `scene.mutate`, `asset.read`, `asset.mutate`, `viewport.read`, `viewport.control`, `diagnostics.read`, and `play.control`. Context declarations use the same section IDs as the AI project-context service.

## Security and authority

Skill discovery is provider-independent and does not execute skill content. The catalog only parses and validates declarations and Markdown instructions. Request-time selection belongs to the instruction/skill resolver, while actual tool availability remains the intersection of the current runtime tool registry, harness capabilities, and AI security policy.

In particular, a skill cannot add a harness operation, enable a restricted filesystem/process operation, create edit authority, or bypass the editor's approval and transaction rules. Persistent scene and asset mutations still flow through `EditorAgentHarness` exactly as described in `docs/editor-agent-harness.md` and `.agents/skills/arc-editor-gateway/SKILL.md`.

Project skill discovery is rooted at the active project's real filesystem path. A project `.agents/skills` symlink that resolves outside that project is rejected, and changing active projects rebuilds the project-local portion of the catalog so instructions cannot leak across project boundaries.
