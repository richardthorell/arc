# Live Play L1.4b — Native runtime transform edit application

Issue [#1199](https://github.com/richardthorell/arc/issues/1199), after
[L1.4a #1228](https://github.com/richardthorell/arc/pull/1228).

This PR implements the native Development side of the **existing** L1.1
`entity.property.set` control message. No C9 HTTP, asset recooking or
gameplay network replication is added.

## Thread ownership and safety

- `arc-live-play-dev` parses exact versioned property payloads and verifies
  the negotiated `entity.edit` capability. Only stable **authored entity GUID**
  references are accepted; no transient ECS entity index may cross the wire.
- The socket worker inserts typed commands into a bounded, FIFO 256-edit
  mailbox, and the Windows host drains it **on the game frame thread before
  `session->advance`**. Never mutate ECS on the network worker.
- The runtime searches running client-world entities by the persistent GUID,
  rejecting missing or ambiguously duplicated identities, absent transforms,
  non-movable objects, unsupported schema/properties/phases, stale per-property
  authoring revisions and future expected runtime revisions.
- Valid `position`, `rotation` and `scale` writes reuse the normal scene
  transform invalidation and hierarchy propagation. The game thread sends a
  bounded result back to the worker; only the worker writes the protocol ACK.
  Errors use existing L1 error codes, with no raw source or session secrets.
- Update/commit for one authoring revision may both apply, but duplicates and
  older revisions fail closed. Applied runtime revisions increment and are
  returned in `command.ack`.
- Overflow gets an explicit `busy` rejection; Stop clears the queues.
  `entity.edit` is negotiated, but **no edits are permitted until C9
  verifies, mounts and activates the exact unsaved-scene baseline**.

## Runtime baseline / ownership boundary

**C9 activation still happens in the other track.** The Windows host today
rejects pending offers as `unavailable`. Its new `live_baseline_active`
gate remains false, so no stale packaged startup world can be mutated by
an editor command. When C9 integration genuinely activates the scene, it
must set that gate at the same safe boundary and reject any queued edits on
Stop/release replacement.

For now, the permission model is conservative: static/nonmovable objects
cannot be overridden; unsupported gameplay/physics ownership changes are
rejected rather than silently implemented. A richer explicit hold/override
policy and runtime world arbitration remain follow-ups before claiming full
physics-authoritative Live Play.

This slice is unit-tested for decode, identity, schema bounds, revision
precision, mailbox FIFO/overflow and Stop cleanup. Actual Windows external
edits require the C9 initial-scene activation integration and an on-device
acceptance run.
