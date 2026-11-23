# Live Play L1.5d — Runtime hierarchy synchronization

Follows L1.5a [#1233](https://github.com/richardthorell/arc/pull/1233),
L1.5b [#1236](https://github.com/richardthorell/arc/pull/1236) and L1.5c
[#1239](https://github.com/richardthorell/arc/pull/1239). Tracks [#1200](https://github.com/richardthorell/arc/issues/1200) and epic #1194.

## Native runtime producer

- On the Windows Development game **frame thread** sample a bounded immutable
  hierarchy of up to **4,096 entities** at most four times a second. Larger
  scenes report `complete: false`; they are **not** silently truncated.
- Keep the existing **one-slot coalescing mailbox** between ECS and socket
  worker. The socket worker computes a delta against the **last successfully
  sent snapshot**, not the last sampled state. Spawn, despawn, reparent, and
  rename produce revisioned `hierarchy.changed` operations with the
  `baseRevision` required for safe replay.
- Send at most **64 changes** per update. For a larger change, unknown base,
  incomplete scene, or reordered-only sequence, send a paged snapshot instead.
- Send paged snapshots in a maximum of **64 pages of 64 entities**. Every page
  carries `worldRevision`, `pageIndex`, `pageCount`, and `complete`.
  Only finished page batches constitute a coherent hierarchy.
- No ECS pointers, project source paths, runtime memory, or editor save state
  cross the bridge. All traffic remains on the authenticated `entity.inspect`
  control channel and opt-in Development game target.

## Editor receiver

- A shared, pure hierarchy assembler in `editor/src/common` is used by both
  the Electron main process (for reload-safe snapshots) and React runtime
  hierarchy view.
- Assemble only consistent page batches from the same session, runtime
  instance, and world revision. Apply a change set only when
  `baseRevision === displayed worldRevision`. Preserve old rows while pages
  arrive; a gap marks the tree stale rather than inventing missing entities.
- Runtime-spawned identity stays separate from authored GUIDs. The Runtime
  Inspector remains read-only; its existing selected component polling
  provides values between structural updates.
- Standalone default in-editor Play, Authoring World data, Shipping builds,
  C9 cooked asset publishing, and content network transport are unchanged.

## Follow-up (L1.5e)

This slice deliberately **does not** implement active full-state recovery:
explicit `hierarchy.resync.request`, disconnected session resume, lost-page
timeouts and gap repair belong to L1.5e. A rejected base or incomplete batch
is marked stale until a valid complete snapshot arrives. Sequence/identity
validation remains the L1.1 control transport's responsibility.

## Verification

- Native portable diffs: coalesced revisions, renamed/reparented entities,
  spawned/removed identity, and >64-change full-snapshot fallback.
- Windows ECS fixture: coherent native scene sample with more than 128
  entities, no authoring edits or second ECS access thread.
- TypeScript assembler: 260 entities over out-of-order pages, atomic commit,
  valid/invalid base revisions, stale/different instance safety, structural
  mutations and strict wire bounds.
