# ARC GitHub Roadmap and Task Workflow

This document defines how ARC plans work in GitHub and how human and AI contributors should create, organize, and complete roadmap tasks.

The durable source of truth is the repository's GitHub Issues. The GitHub Project named **ARC Roadmap** is the planning view over those issues.

Agents should read this document before creating roadmap issues, milestones, or planning metadata.

## Planning hierarchy

ARC uses this hierarchy:

```text
ARC Roadmap GitHub Project
└── Project Space / Epic issue
    ├── Milestone-sized issue when useful
    │   └── Task issue
    │       └── Pull request
    └── Task issue
        └── Pull request
```

Use the smallest level that adds useful structure. Do not create empty hierarchy merely for consistency.

### Project spaces

Project spaces are long-lived engineering domains. They are represented by **Epic issues**, not GitHub Milestones.

The current project spaces are:

1. Editor Platform & Shared UX
2. Asset Library & Content Browser
3. Materials, Shaders & Textures
4. Shared Graph Framework
5. Flow Gameplay Scripting
6. Play-in-Editor & Runtime Authoring Loop
7. Input & Player Mapping
8. Terrain & World Building
9. Rendering & GPU Architecture
10. Platforms & Rendering Backends
11. Project SDK, C++ Workflow & Packaging
12. Core Game Runtime Systems
13. Agent & Automation Platform

Create a new project space only when the work is expected to remain a distinct long-lived domain. Prefer adding tasks to an existing project space over creating narrow new spaces.

## GitHub Project

Use one GitHub Project:

```text
ARC Roadmap
```

Recommended project fields:

| Field | Values |
| --- | --- |
| Status | Backlog, Ready, In Progress, Review, Done |
| Project Space | Editor, Assets, Materials, Graph, Flow, Play, Input, Terrain, Rendering, Platforms, SDK & Packaging, Runtime, Agents |
| Priority | P0, P1, P2, P3 |
| Horizon | Now, Next, Later, Someday |
| Size | XS, S, M, L, XL |

Recommended views:

- **Now** — Ready and In Progress, grouped by Project Space.
- **Roadmap** — grouped by Horizon.
- **Project Space views** — filtered to one Project Space.
- **Big Rocks** — Epic issues.
- **Bugs** — bug issues.
- **Tech Debt** — technical-debt issues.

If the available GitHub integration cannot edit Project fields, still create the Issue with the metadata described below. Do not invent a second tracking system. A human or later automation can attach the Issue to the Project and copy the metadata into Project fields.

## Issues are the source of truth

Before creating an issue:

1. Search existing open and closed Issues for the same work.
2. Inspect the relevant roadmap/design documents.
3. Inspect the current implementation so the issue does not describe work that has already landed.
4. Check recent and open pull requests when the task may already be in flight.
5. Prefer updating an existing issue when the scope is substantially the same.

Do not create stale roadmap tasks solely because an old discussion or document mentions them. Current code wins when it proves a milestone has already shipped; update stale documentation separately when appropriate.

## Epic issues

Create one Epic issue for each Project Space.

Use this format:

```md
# Goal

One concise statement of what this project space is intended to achieve.

## Current state

What exists in the repository today. Reference relevant modules, editor surfaces, tests, and roadmap documents.

## Roadmap

- [ ] Major outcome or milestone
- [ ] Major outcome or milestone
- [ ] Major outcome or milestone

## Child issues

- #123 Task title
- #124 Task title

## Related documentation

- `docs/...`

## Code anchors

- `engine/...`
- `editor/...`
```

Epic issues should describe outcomes rather than implementation details. Keep speculative long-term work in the Epic until it becomes actionable.

## Task issues

Create individual Issues for concrete work that can reasonably be implemented and reviewed.

Use this format:

```md
## Summary

What should change and why.

## Roadmap metadata

- Project Space: <space>
- Horizon: <Now | Next | Later | Someday>
- Priority: <P0 | P1 | P2 | P3>
- Size: <XS | S | M | L | XL>
- Type: <feature | bug | tech-debt | design>
- Parent Epic: #<issue>

## Scope

- Concrete work item
- Concrete work item

## Acceptance criteria

- Observable outcome
- Testable behavior
- Required tests / diagnostics / documentation

## Dependencies

- #<issue>, if applicable

## References

- Relevant roadmap/design docs
- Relevant code paths
```

The metadata block is mandatory when an agent creates an issue. It allows the issue to be triaged correctly even when Project-field automation is unavailable.

## Milestones

GitHub Milestones are for **finite deliverables**, not long-lived domains.

Good milestone examples:

- Terrain M4 — Mesh-Native Terrain
- Flow F7 — Debugging & Hot Reload
- Android M1 — Device Bring-up
- Editor UX M1 — Shared Control Migration

Do not create milestones named simply `Terrain`, `Rendering`, `Android`, or `Editor`.

A milestone should have a clear definition of done and a bounded set of issues.

## Labels

Keep labels small and orthogonal. Project fields should carry most planning metadata.

Preferred label families:

```text
area:editor
area:assets
area:materials
area:graph
area:flow
area:play
area:input
area:terrain
area:rendering
area:platform
area:sdk
area:runtime
area:agents

type:feature
type:bug
type:tech-debt
type:design

P0
P1
P2
P3

blocked
breaking-change
good-first-issue
```

Do not encode Status or Horizon as labels when the ARC Roadmap Project can represent them.

## Horizons

Use horizons to control backlog size:

- **Now** — work actively being executed or expected immediately.
- **Next** — likely to start after current work.
- **Later** — planned architecture/work with meaningful definition but not imminent.
- **Someday** — directional ideas that should not produce a large child-issue backlog yet.

Only create detailed child issues for Now/Next work and well-defined Later work. Keep speculative decomposition inside the Epic.

## Priority

Priority describes urgency/value, not implementation order:

- **P0** — blocking, broken mainline, data loss, or release-critical.
- **P1** — important current product/architecture work.
- **P2** — valuable planned work.
- **P3** — opportunistic polish or long-term improvement.

Dependencies and Horizon determine sequencing.

## Size

Use rough engineering size only:

- **XS** — isolated change, normally a few hours.
- **S** — small focused PR.
- **M** — several related changes, normally one substantial PR or a short PR series.
- **L** — milestone-sized work that should usually be decomposed.
- **XL** — Epic-scale; do not implement as one PR.

## Pull requests

Implementation PRs should reference the task issue.

When a PR fully completes an issue, use:

```text
Closes #123
```

When it contributes but does not complete the issue, use:

```text
Part of #123
```

Do not use an Epic issue as the direct closure target unless the PR genuinely completes the entire Epic.

PR descriptions should state which acceptance criteria are completed and call out any remaining work.

## Agent workflow for discovering new work

When an agent discovers follow-up work while implementing something:

1. Decide whether the follow-up belongs in the current PR.
2. If not, search Issues for an existing task.
3. If no task exists, identify the correct Project Space/Epic.
4. Inspect the code and roadmap docs enough to write concrete scope and acceptance criteria.
5. Create one task issue with the standard metadata block.
6. Link it from the parent Epic.
7. Add it to ARC Roadmap and set Project fields when the integration supports those actions.
8. Continue the original task; do not expand its PR opportunistically.

Avoid creating tiny issues for trivial cleanup that should simply be part of the current change.

## Roadmap maintenance

When repository state overtakes a plan:

- update the relevant Epic/task rather than leaving known-stale language;
- close completed issues as completed;
- close superseded work as not planned or duplicate with an explanatory comment;
- update roadmap documents when architecture or milestone boundaries materially change;
- do not reopen completed milestones just to hold unrelated future work.

Periodically compare the issue backlog with:

- current code;
- `README.md`;
- `docs/terrain-roadmap.md`;
- `docs/editor/graph-editor-foundation.md`;
- `docs/editor/asset-library.md`;
- `docs/flow-play-integration.md`;
- `docs/editor/flow-graph.md`;
- `docs/project-modules.md`;
- `docs/api-modules.md`;
- `docs/virtual-shadow-maps.md`;
- recent merged PRs.

## Platform and backend planning

Platform work belongs under **Platforms & Rendering Backends**, separate from backend-neutral renderer features.

The intended sequence is:

```text
Windows + Vulkan
    ↓
Android + Vulkan
    ↓
D3D12 + Windows
    ↓
Metal + macOS
    ↓
Metal + iOS
    ↓
additional targets as justified
```

Windows is the reference desktop implementation. Android is the first device-platform validation and should test application lifecycle, surface recreation, touch/sensors, filesystem/storage, packaging/deployment, mobile GPU capabilities, memory/thermal constraints, and device profiles while continuing to use Vulkan.

Do not add platform checks to backend-neutral systems when a capability query or platform abstraction is appropriate.

## Guiding principle

GitHub should answer three questions without reading chat history:

1. What are ARC's major project spaces?
2. What concrete work is active or planned in each space?
3. Which issue does a PR advance or complete?

If a task exists only in a conversation and cannot be found from the repository or GitHub Issues, it is not yet part of the durable ARC roadmap.
