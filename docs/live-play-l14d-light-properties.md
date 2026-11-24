# Live Play L1.4d — Typed authoring light property edits

Tracking: [#1199](https://github.com/richardthorell/arc/issues/1199). Follows merged #1228 (L1.4a), #1232 (L1.4b), #1245 (L1.4c) and #1247 (L1.3 verified Windows startup).

## Scope

- The editor's native `scene.hierarchy` snapshot includes a **read-only** optional `light` value for every authored directional, point, spot, or area light. Values are taken directly from native ECS, not the renderer or untrusted UI state.
- L1.4 authoring observer establishes the exact per-entity light type and its initial intensity/color from the L1.3 immutable captured `.arcscene`. On native `component.changed` events (including Undo/Redo), it compares new native hierarchy values by authored entity GUID and queues **typed schema-v1** commands:
  - `arc.light.directional`, `arc.light.point`, `arc.light.spot`, `arc.light.area`
  - allowlist `intensity` (one finite, non-negative scalar) and `color` (three finite, non-negative floats)
- The existing bounded throttle, last-value commit, six-attempt retry, authenticated ACK, revision/idempotence and session/instance isolation are **reused**, not duplicated.
- The native Windows runtime applies changes at the frame boundary via the same `edit_mailbox`, after baseline activation, authored-GUID lookup, schema/property/length validation, and movable-mobility gating. Unrelated entities are untouched. A duplicate matching commit returns its prior applied revision.
- A light component being added/removed/retyped, invalid values, an unknown entity or unknown field requires an explicit new baseline/unsupported result rather than mutating a mismatched ECS component.

## Explicit limits and follow-up

- This PR does **not** add `range`, shadow settings, flags, color-temperature, light asset changes, or arbitrary component reflection. Those require distinct typed policies or C9 recooking.
- A static/stationary light is not mutated without authoring/runtime authority; this slice preserves the existing movable-only gate. L1.4e must define runtime-owned physics/gameplay transform authority, an explicit editor override/hold policy, and how to release it safely.
- `#1199` stays open until Windows sample Play demonstrates unsaved cube drag and visible light change, with measured event-to-visible-frame latency (<100 ms target); component unit/protocol tests are not a substitute for visible frame acceptance.
