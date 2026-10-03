# AI context budgeting and compaction

ARC builds each chat request through `prepareAiContextBudget` before it reaches a provider. The planner owns conversation compaction and prompt-budget policy; provider adapters remain responsible only for transport and authoritative usage accounting.

## Model-aware budget

The planner uses `AiModelCapabilities.maxContextTokens` and `maxOutputTokens` when a model publishes them. Models without explicit limits use conservative ARC defaults. The context window is split into an input budget, an output reserve, and a safety margin so requests do not blindly consume the provider's entire context window.

Input is budgeted across four sources:

- recent conversation turns;
- a compacted summary of older turns;
- pinned and current explicit context references;
- automatic project/editor context from `AiProjectContextService`.

Recent turns retain the newest user/assistant exchange first. Older messages stay in the project-scoped conversation store for UI/history, but they are not resent on every request. Instead, omitted turns are folded into `AiStoredConversation.summary`, which is persisted by the existing conversation store and incrementally extended as the recent window advances.

Token estimates use the same provider-neutral `ceil(characters / 4)` approximation as project context plus a small per-message overhead. These estimates drive deterministic budgeting only; usage events from the provider remain authoritative.

## Explicit, pinned, and automatic context

Context origin is intentionally visible in `AiContextBudgetDecision`:

- `pinned` references are user-selected context that must survive conversation compaction;
- `explicit` references are attached to retained recent turns or supplied for the current request;
- `automatic` context is editor/project state collected without a direct attachment action;
- `summary` and `recent` identify compacted and verbatim conversation history.

Pinned references are serialized independently of transcript compaction. When their metadata is too large for the reference budget, ARC drops metadata before stable identity so the pinned GUID/asset identity remains represented.

Automatic editor context is added in priority order: project, selection, workspace, scene, diagnostics, viewport, and recent changes. Lower-priority sections are truncated or skipped when the automatic-context budget is exhausted. Every inclusion or rejection is recorded in diagnostics.

## Freshness and stale references

Before a chat request uses editor context, ARC collects a current `AiProjectContextSnapshot`. A cached snapshot is force-refreshed when:

- its active project does not match the chat's project;
- any section exceeds the configured maximum age; or
- an explicit/pinned reference carries scene/world/frame/event revision metadata that has drifted.

After refresh, stale stable references are accepted only if their stable identity can be resolved in the refreshed project context. A reference that still belongs to another project, points at a replaced world/entity, or otherwise cannot be re-resolved is omitted from the provider request and reported in `rejectedReferences`. This prevents old transient editor state from silently leaking into a new request.

References without revision metadata remain valid stable identities. The attachment picker can add freshness metadata when it has a concrete editor snapshot available.

## Diagnostics

`AiContextBudgetPlan.diagnostics` exposes the complete budgeting decision:

- model context size, input budget, output reserve, and safety margin;
- estimated input tokens;
- retained and compacted message counts;
- summary, pinned, explicit, and automatic token estimates;
- whether project context was unavailable, reused, or force-refreshed;
- per-source inclusion decisions and reasons;
- rejected stale references.

`AiChatPanel` accepts an optional `onContextBudget` callback for diagnostics/test consumers and an injectable `contextSource` for deterministic tests. Production chat creates the window-backed `AiProjectContextService` automatically when a project is active.

The planner never deletes conversation history and never grants editor capabilities. Context is read-only reference data; mutations still flow through `EditorAgentHarness` and its approval/transaction policy.
