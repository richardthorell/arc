# AI project context service

ARC gathers editor and project state for AI requests through `AiProjectContextService`. The service is independent of the AI Chat panel and composes small `AiContextProvider` implementations into a versioned snapshot that other AI, agent, or tooling features can consume.

## Context schema

`AiProjectContextSnapshot` is versioned independently from provider payloads. Each collection records the active project GUID, capture time, native scene/world/frame revisions where available, the latest editor event sequence, per-section freshness, and an estimated prompt cost.

The built-in providers collect:

- project metadata and stable scene asset references from the active ARC project descriptor;
- active scene hierarchy through `scene.hierarchy`;
- current selection and component metadata through `entity.selected`;
- active/open editor documents through `workspace.documents`;
- editor/native diagnostics through `gateway.diagnostics`;
- viewport state through `viewport.state`;
- a bounded list of recent meaningful host changes.

Scene and selection payloads prefer persistent entity GUIDs. When a host object already has a stable `guid`, the transient `{ index, generation }` entity handle is removed from AI context. Project scene references likewise retain their asset GUID and path hint rather than relying on runtime indices.

## Freshness and invalidation

Providers may be cached briefly to avoid issuing the same native queries several times during one AI interaction. Native host events selectively invalidate affected providers. Project open/close events invalidate every provider; scene/entity/component changes invalidate scene and selection context; asset changes invalidate workspace/diagnostics; viewport changes invalidate viewport context. Failure/error events invalidate diagnostics.

Every collection still reads the active project snapshot first. A project GUID change clears all cached sections and recent changes even if an event was missed, preventing one project's state from leaking into another project's AI context.

Consumers can call `invalidate()` explicitly or request `collect({ forceRefresh: true })` when they require a fresh snapshot.

## Context budgets

Provider values are normalized into JSON-safe data before publication. Collection applies deterministic limits for recursion depth, array length, object key count, and string length. Object keys are sorted, cycles are replaced with a marker, and every section reports whether it was truncated.

The service estimates prompt cost from the serialized character count (`ceil(characters / 4)` tokens). This is intentionally a provider-independent estimate for budgeting and diagnostics; the final model adapter remains authoritative for tokenizer-specific accounting.

## Observability

`subscribe()` exposes collection lifecycle events:

- collection started/completed;
- provider completed, including live-vs-cached state and duration;
- explicit or host-driven invalidation.

Provider failures are isolated to the affected section and returned as `status: 'error'`; one unavailable context source does not prevent the rest of the snapshot from being collected.

## Integration

Use `createWindowAiProjectContextService()` in renderer features that need the production Electron/host bridges. Unit tests and non-UI consumers should construct `AiProjectContextService` with an injected `AiProjectContextEnvironment`, which keeps project snapshots, native queries, event delivery, and time fully controllable.

The context service is read-only. It does not grant agent capabilities or bypass `EditorAgentHarness`; any mutation still follows the approval, transaction, and capability rules in `docs/editor-agent-harness.md`.
