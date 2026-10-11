# Editor agent harness

ARC exposes editor automation through one transport-neutral harness and optional connection adapters.

```text
Built-in agent runtime ---------------+
                                      +--> EditorAgentHarness --> native editor host
HTTP / JSON-RPC / MCP / SSE gateway --+            |
                                                   +--> typed project asset workspace
```

`EditorAgentHarness` owns operation semantics, persistent-GUID resolution, scene revisions, approval, leases,
transactions, viewport capture analysis, and typed asset staging. `AiGatewayServer` owns only localhost discovery,
authentication, rate limiting, protocol adaptation, event streaming, OpenAPI, and capture artifact delivery.

A built-in agent must invoke `EditorAgentHarness` directly with a stable client ID. It must not loop back through the
localhost gateway or duplicate operation validation. External clients use the same harness through the gateway.

The built-in AI runtime additionally applies the data/tool policy in `docs/ai-chat-security.md`, but that policy is a
pre-flight guard rather than a replacement for harness authority. Skills and provider responses can request only tools
that project existing harness capabilities; they cannot grant capabilities or bypass approval/transaction semantics.

## Current capability surface

### Shared operation contract (#792, first slice)

`editor/src/common/editorOperationRegistry.ts` is the transport-neutral registration contract. Trusted editor
code registers Zod input/output schemas, an executor, a stable namespaced ID/version, permissions, required
capabilities, resource effects, scheduling/cache policy, transaction participation, and reference/creation metadata.
The harness supplies authorization; metadata is not an authorization grant. Project content and clients cannot
register executable code. Unknown permissions fail closed.

`agentOperationCatalog.ts` adapts the existing operations. The production built-in agent, MCP, and OpenAPI all
derive schemas from these registrations. Existing REST routes and MCP aliases remain supported; newly registered
operations receive `/api/v1/operations/<operation.id>` and an `arc_<namespace>_<operation>` MCP tool automatically.
Register extensions at startup. Registration removal/replacement changes the catalog revision; MCP clients should
refresh their tool list after catalog changes. Availability is checked again on every invocation.

Discovery is split into two bounded queries:

- `agent.operations.list`: summaries, optional domain filter, offset, and limit (default 50, maximum 100).
- `agent.operations.describe`: exact input/output schemas for up to 16 IDs.

Both return a `catalogRevision`. Pass it back on subsequent discovery requests to reject changed catalogs.
Version-aware invocations include the reserved envelope field:

```json
{
  "guid": "<entity-guid>",
  "operationContract": { "version": 1, "catalogRevision": "<discovered-revision>" }
}
```

The envelope is separate from operation arguments. Omitting it retains legacy version-1 behavior; later contract
versions require explicit negotiation. Revisions identify catalog registrations, not scene/document freshness or
approval. Authorization, live availability, and input validation run before execution; results are checked against
the registered output schema. An invalid-result error does not imply that an executor's effects were rolled back.

Contract errors include `unknown_operation`, `unsupported_version`, `stale_catalog`, `unavailable`,
`permission_denied`, `invalid_arguments`, and `invalid_result`. HTTP/RPC, MCP, and built-in tool IPC preserve these
codes. The built-in agent pins the envelope to the tool schema advertised for that provider request; it does not
silently upgrade a pending call when registrations change. Transport-name collisions are rejected at registration.
Do not automatically retry a
mutation after an uncertain completion.

Legacy host output snapshots deliberately retain an extensible object schema, rather than guessing native fields.
Their scheduling is conservative (serial, uncached), and their existing mutation semantics are marked `legacy`.
New domain adapters must provide precise output contracts. This slice does **not** claim cross-domain atomicity,
asset undo, read-set caching, semantic graph editing, or the revised agent budget. Those remain the next focused
#792 PRs: reads/transactions, domain operations (coordinated with #986), then agent execution/evaluations.

The former renderer-only registry import re-exports the shared contract; there is no second renderer registry.

### Existing operations

- Inspect and search the scene, components, spatial relationships, assets, history, renderer state, and diagnostics.
- Move, configure, pick, capture, sample, and compare the viewport.
- Create, duplicate, delete, rename, activate, tag, reparent, and snap entities to the floor.
- Set transforms, mobility, render layers, materials, and Flow graphs.
- Patch transforms, cameras, lights, mesh renderers, terrain, water, Flow, and world environments.
- Create material, Flow, and Slang shader assets from defaults or complete authored definitions.
- Commit or cancel one approved transaction and undo or redo committed scene edits.

Call `agent.capabilities` (MCP: `arc_agent_capabilities`) instead of hard-coding this list. Entity creation and
duplication return a persistent GUID so later operations in the same transaction can target the new entity.

The low-level `edit.apply/createAsset` path stages assets in memory until commit. Legacy `editor.applyBatch`
material creation currently materializes files within an approved active session so later operations can bind them,
and removes them on cancellation. Neither path overwrites existing files. Native scene changes remain
covered by the editor history transaction; newly created assets are not currently part of undo/redo after commit.

## Deliberate boundaries and remaining gaps

Saving scenes, opening or replacing projects, running scripts or processes, arbitrary file access, and build/package
execution remain unavailable to agents. These operations need separate user intent and lifecycle policies rather than
being added as generic harness commands.

The built-in agent/tool loop must check cancellation immediately before every harness invocation. A skill declaration
or provider tool call is never sufficient authority on its own: the operation must still exist in current harness
capabilities and any mutation must retain the harness approval/transaction path.

The next useful capability additions are prefab authoring/instantiation, selected component add/remove/reset with
GUID-first targeting, typed updates to existing material and Flow assets with rollback, terrain brush strokes, and
runtime play/pause/step controls. Each should be added to the harness first and then projected into transport schemas.
