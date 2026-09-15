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

## Current capability surface

- Inspect and search the scene, components, spatial relationships, assets, history, renderer state, and diagnostics.
- Move, configure, pick, capture, sample, and compare the viewport.
- Create, duplicate, delete, rename, activate, tag, reparent, and snap entities to the floor.
- Set transforms, mobility, render layers, materials, and Flow graphs.
- Patch transforms, cameras, lights, mesh renderers, terrain, water, Flow, and world environments.
- Create material, Flow, and Slang shader assets from defaults or complete authored definitions.
- Commit or cancel one approved transaction and undo or redo committed scene edits.

Call `agent.capabilities` (MCP: `arc_agent_capabilities`) instead of hard-coding this list. Entity creation and
duplication return a persistent GUID so later operations in the same transaction can target the new entity.

Typed assets are staged in memory and written below the project's primary content root only when the transaction
commits. Existing files are never overwritten, and cancellation leaves no files behind. Native scene changes remain
covered by the editor history transaction; newly created assets are not currently part of undo/redo after commit.

## Deliberate boundaries and remaining gaps

Saving scenes, opening or replacing projects, running scripts or processes, arbitrary file access, and build/package
execution remain unavailable to agents. These operations need separate user intent and lifecycle policies rather than
being added as generic harness commands.

The next useful capability additions are prefab authoring/instantiation, selected component add/remove/reset with
GUID-first targeting, typed updates to existing material and Flow assets with rollback, terrain brush strokes, and
runtime play/pause/step controls. Each should be added to the harness first and then projected into transport schemas.
