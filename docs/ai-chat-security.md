# AI Chat security and data boundaries

ARC AI Chat is allowed to reason over editor/project context and eventually invoke editor tools, but it is not a privileged escape hatch around normal editor security. This document defines the mandatory boundary for provider adapters, project context, skills, diagnostics, persistence, and built-in agent execution.

The executable policy primitives live in `editor/src/common/aiSecurityPolicy.ts`.

## Principles

1. Provider credentials are transport secrets, not conversation data.
2. Project-scoped data may only be used while that exact project is active.
3. Automatically supplied project/editor context must be inspectable by the user before it can leave the editor.
4. Sensitive context requires explicit approval. Credentials and secrets are never eligible for provider context.
5. Skills describe behavior and select registered tools; they never grant permissions.
6. `EditorAgentHarness` remains authoritative for editor mutations, approvals, revisions, transactions, and capability discovery.
7. Arbitrary process execution, unrestricted filesystem access, project lifecycle operations, and build/package execution remain outside AI Chat until separately designed and approved.
8. Cancellation must stop provider work and prevent subsequent tool execution.
9. Diagnostics record operational metadata, not prompt/context/tool payload contents.

## Provider credential boundary

Provider credentials are owned by `AiProviderService` and stored through Electron `safeStorage`. The provider account snapshot intentionally exposes only connection state; it never returns the credential.

A provider adapter may obtain its provider credential only at the final transport boundary needed to construct authentication headers. Credentials must not be added to:

- `AiRuntimeRequest.messages`
- request metadata
- conversation persistence
- context items
- skills
- tool arguments/results
- diagnostic payloads

`assertAiRuntimeRequestSafeForProvider()` rejects secret-like metadata keys before a request is dispatched. This is a defense-in-depth check for ARC-owned metadata; ARC does not rewrite arbitrary user-authored prompt text because doing so would silently change user intent.

## Outbound project/context policy

Every ARC-owned context item intended for an external provider should carry enough policy metadata to evaluate `AiOutboundDataItem`:

- origin (`project`, `editor`, `skill`, `tool`, etc.)
- sensitivity
- whether it is project-scoped and, if so, the owning project GUID
- whether the user can inspect it
- whether sensitive data was explicitly approved

Rules:

- `credential` and `secret` data are always blocked.
- Project-scoped data requires the active project GUID and must match it exactly.
- Project/editor/skill/tool context must be inspectable.
- `sensitive` context requires explicit approval.
- User-authored prompt/conversation content is considered intentional input and is not heuristically rewritten by ARC.

Issue #681 should construct context through this policy rather than passing raw editor objects directly to providers.

## Project isolation

The stable project GUID from the ARC project descriptor is the security scope for project AI state.

Ephemeral context, active tool work, selected/pinned context, and cancellation state must be discarded when the active project changes or closes. Persisted conversations must be restored only for the matching project GUID.

Issue #680 owns the versioned project-scoped persistence implementation. The policy in this document remains authoritative even if the backing persistence mechanism changes.

## Skills are not authority

Built-in and project-local skills may request/select tools by name, but tool selection is an intersection with the registered runtime tool set. A skill cannot synthesize a new tool descriptor, declare a new harness capability, change a tool from read-only to mutating, or remove approval requirements.

`resolveAiSkillTools()` intentionally returns only already-registered tool descriptors.

Project skills are also project context: their instructions must obey project scoping and context inspection rules before being sent to a provider.

## Tool and harness boundary

Built-in AI tools must project existing `EditorAgentHarness` capabilities. `evaluateAiToolInvocation()` enforces the runtime-side invariants before an invocation reaches the harness:

- cancellation has not already occurred
- the tool is harness-backed rather than restricted
- the harness operation is currently advertised by capability discovery
- mutating tools preserve harness approval semantics

This check does **not** replace harness validation. The harness still owns the authoritative decision, revision validation, approval, transaction, and mutation semantics.

The following classes remain restricted and must not be exposed as generic AI tools:

- arbitrary process/script execution
- unrestricted filesystem access
- opening/replacing/closing projects
- build/package execution

New privileged capabilities require their own user-intent and lifecycle policy before they may be added to the harness and then projected into AI tools.

## Cancellation

An `AbortSignal` is part of the in-process AI runtime request. Provider adapters must propagate it to network requests/streams.

The agent/tool loop must also check the same cancellation state immediately before every tool invocation. A response that arrives after cancellation must not trigger a harness operation.

## Diagnostics and retention

AI diagnostics are session-operational metadata by default. Persistent diagnostics must not contain full prompt text, provider response text, context contents, tool argument/result payloads, or credentials.

Use `summarizeAiRuntimeRequestForDiagnostics()` for request-level tracing. It records counts, roles, tool names, safe metadata keys, conversation identity, and cancellation state without message or metadata values.

Any diagnostic object or provider error text that may contain remote/service data must pass through `redactAiDiagnosticValue()` / `redactAiDiagnosticText()` before logging or persistence. Known credential fields, bearer tokens, and API-key-like values are replaced with `[REDACTED]`.

## Persistence and export

Normal conversation history is user-private editor data and is not written into the project repository by default. Credentials are never part of the conversation schema.

An explicit future export/share feature must define its own review surface and must apply the same secret/context policies before material leaves the editor profile.

## Integration requirements for later milestones

- **#678 / #679 provider adapters:** validate ARC-owned request metadata, fetch credentials only at the transport boundary, propagate cancellation, and sanitize diagnostics.
- **#681 context service:** classify every context item, bind project data to the active project GUID, and expose user inspection.
- **#684 / #685 skills:** skills select registered tools/instructions but never authority.
- **#687 / #688 built-in agent:** project harness capabilities into registered tools and run the policy/cancellation guard before each invocation.
- **#689 approvals:** preserve `EditorAgentHarness` approval/transaction semantics and surface them in Chat rather than bypassing them.
