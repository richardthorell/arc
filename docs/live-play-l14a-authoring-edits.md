# Live Play L1.4a — Native Authoring World transform capture

L1.4a is the **editor-to-game producer** for [#1199](https://github.com/richardthorell/arc/issues/1199).
It is separate from C9: these are small, typed authoring property deltas,
**not** cooked asset changes or source file uploads.

- Native host `component.changed` and successful Undo/Redo/cancel events trigger
  a bounded, coalesced `scene.hierarchy` query. The editor main process
  compares transforms by persistent **authored GUID**, not transient ECS index.
- The immutable L1.3a captured scene is the baseline. New/deleted entities and
  unsupported components are deliberately not transmitted yet.
- Position, quaternion rotation and scale use existing `entity.property.set`
  with `arc.transform` schema 1; revisions increase monotonically.
- During drag, intermediate updates are throttled; the latest value is emitted
  again as a commit after a quiet interval. Dropped sends retain the final
  value; successful ACKs retire only matching latest revisions.
- Nothing is sent before the native runtime explicitly ACKs the matching
  cooked scene baseline as **active**. Stop/project changes clear all state;
  stale queries, mismatched scenes, invalid values and oversized queues fail
  closed. No Save, cook, runtime asset access, or Editor Viewport PIE changes.

**L1.4b next:** actual native Development bridge handling; enqueue edits on
simulation-safe world boundaries, authority/expected-revision conflict policy,
ACK/rejection/resync and tests with running external Play World. Runtime
`entity.edit` capability is intentionally **not granted yet**: L1.4a captures
changes, but the native game is not prepared to apply them. C9 integration is
tracked independently in #1133.
