# Live Play L1.4c — Reliable authoring transform commits and native ACK reconciliation

**Tracking:** [#1199](https://github.com/richardthorell/arc/issues/1199).
**Prerequisites:** merged [#1228](https://github.com/richardthorell/arc/pull/1228),
[#1232](https://github.com/richardthorell/arc/pull/1232), and
[#1243](https://github.com/richardthorell/arc/pull/1243).

## What is implemented

- Native editor-host authoring transform changes, including Undo/Redo, remain
  session-scoped and **never save the scene**. Intermediate drags still throttle
  to 33 ms; a final update is committed after a 120 ms quiet interval.
- Final commits now retry automatically on a refused send or missing ACK.
  The coordinator preserves the **same authored property revision and value**
  across retry attempts, correlates each request ID to its immutable revision,
  and bounds retry attempts to six. An unacknowledged value does not turn into
  a false success or wait forever.
- Native Windows Development applier recognizes an *identical* retry with the
  same GUID, property, authoring revision, phase and value. It returns the
  originally applied revision **without writing ECS again**. Conflicting
  duplicate values, old revisions and unsupported schemas remain rejections.
- ACKs are scoped to project/build/target/session/**instance**, so an old
  runtime cannot complete a new session's pending edits. ACKs for superseded
  drags cannot clear newer unsaved edits. On disconnect/Stop, pending edits
  are discarded rather than replayed to a different or reconnected game.
- Deleting or creating an authored entity is reported as requiring a new
  baseline. No unsafe create/delete/reparent transaction is manufactured
  from ordinary transform events.
- Producer diagnostics now retain last applied runtime revision, ACK
  round-trip, applied/rejected counts and explicit terminal failure.
  L1.5c already reports native ACK timing and rejection status separately.

## Property ownership

The live bridge remains an *allowlist*: authored Transform only, version 1;
movable entities only; validated stable authored GUID, finite numeric data,
expected revision and frame-thread application. No generic C++ field reflection
or arbitrary physics writes are enabled. **There is not yet a verified
physics/gameplay authority override/hold contract**, so these commands must not
be advertised as permanently overriding simulation-driven transforms.
A subsequent slice must define how runtime-owned (physics/gameplay) transforms
accept a temporary editor lease, or explicitly reject them via authoritative
ownership metadata. Guessing based on component names would be unsafe.

## Remaining before #1199 can close

1. [#1198](https://github.com/richardthorell/arc/issues/1198) and
   [#1140](https://github.com/richardthorell/arc/issues/1140): genuine C9
   verified cooked baseline activation and Windows attached Play orchestration.
   Today the native baseline offer still returns `rejected/unavailable`;
   **no end-to-end live transform update can honestly be claimed** until then.
2. Typed allowlisted value edits beyond Transform (e.g., light intensity,
   color and selected Active fields). The native editor hierarchy query
   currently exposes transform/name/active but not full light values, so
   light support needs an authoritative typed authoring query and a matching
   native schema implementation.
3. Explicit gameplay/physics transform ownership and opt-in override/hold;
   verified create/delete/reparent semantics (or documented reload-required
   behavior for unsupported structural edits).
4. External Windows game acceptance with real on-screen update and
   instrumented **editor-change-to-visible-frame** latency. The existing
   control RTT is not a substitute for measuring frame presentation.

## Verification in this PR

- Final commit resend on temporary transport refusal and lost ACK.
- Bounded terminal timeout and failure reporting.
- Wrong instance, late ACK, newer revision, disconnect, and stale replay.
- Native safe-frame idempotent ACK with same applied revision, and rejection
  when the retried payload has a conflicting value.
- Existing L1.4 native ECS and editor tests continue to run.
