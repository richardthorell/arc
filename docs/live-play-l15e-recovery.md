# Live Play L1.5e — Reconnect and full hierarchy recovery

Follows [L1.5d #1241](https://github.com/richardthorell/arc/pull/1241),
[L1.5c #1239](https://github.com/richardthorell/arc/pull/1239), and
[L1.5b #1236](https://github.com/richardthorell/arc/pull/1236).
Completes the L1.5 inspection recovery work in [#1200](https://github.com/richardthorell/arc/issues/1200).

## Editor: authoritative repair

- The existing `session.resync.request` v1 control message now has a real
  native implementation. Only the **authenticated, negotiated**
  `entity.inspect` session can request it.
- Editor main owns a single hierarchy recovery coordinator (not the
  renderer). It asks for a new full snapshot on incorrect delta base,
  incomplete page timeout (>1.6 seconds), protocol sequence gaps and a
  reconnected session.
- Recovery is bounded to four requests per episode with a 2.4-second retry
  interval, and the transport enforces a 600 ms minimum send interval. Neither
  malformed wire data nor an untrusted renderer is allowed to command native
  ECS writes.
- A renderer reload rehydrates only main's completed hierarchy state.
  The previous verified tree is retained as stale until the entire new
  64-item-per-page batch arrives and passes L1.5d validation.
- Disconnection clears stale Inspector component values and pending requests.
  Incomplete renderer pages time out too; they do not appear as a complete tree.
  Live Play authoring edits remain disabled until a verified cooked baseline
  is reactivated. No stored edit command is automatically replayed.

## Native Development runtime: bounded reconnect

- The Windows Development bridge retries a temporarily lost loopback socket
  for up to 12 seconds without stopping simulation. On reconnect, it reuses
  the **inherited, process-private handoff** (never argv, environment, disk
  or logs), negotiates permissions again, and continues both sides' monotonic
  message sequences. No new handshake accepts replayed sequences.
- A successful resume clears outstanding edit and inspection mailboxes and
  invalidates the previous structural diff base. The socket worker requests a
  **forced snapshot** from the owning runtime frame thread; it never accesses
  ECS directly. The native Inspector publishes a new world revision even if
  the structure did not change, and the next 64-item pages replace the stale
  editor hierarchy atomically.
- An authenticated `session.resync.request` can also force that full
  snapshot without a socket interruption. Unknown reason, unauthorized
  capability, invalid acknowledgement or malformed command is rejected
  without expanding the permitted protocol.
- No gameplay networking, C9 content delivery, packaged startup scene
  activation, Authoring World saves, or Shipping target changes.

## Limitations

- A native game cannot reattach to a **different editor process/port** using
  a stale handoff. This scope covers transient disconnects to the original,
  still-running editor. Starting a new editor requires a fresh Live Play
  launch and protected handoff.
- If the original editor remains unavailable for more than 12 seconds,
  the Development game can continue standalone but its Live Play control
  worker stops. Resuming later requires a fresh launch.
- Snapshots are still capped at 4,096 entities; larger worlds correctly
  report `complete: false`.

## Verification

- Editor-side hierarchy recovery test covers page timeout, capped retry,
  reconnect invalidation, stale deltas and completed replacement.
- Editor loopback tests cover permission gating, frame-gap requests,
  throttling and reconnect negotiation.
- Cross-platform native socket fixture authenticates a second connection,
  exercises `session.resync.request`, preserves monotonic sequence
  numbers across the socket loss and rejects invalid authority.
- Windows opt-in ECS fixture verifies forced unchanged hierarchy snapshots
  advance their revision on the frame thread.
