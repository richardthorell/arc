# Live Play L1.5a — Native runtime hierarchy and component inspection

Parent [#1200](https://github.com/richardthorell/arc/issues/1200); this
follows merged L1.4a [#1228](https://github.com/richardthorell/arc/pull/1228)
and L1.4b [#1232](https://github.com/richardthorell/arc/pull/1232).

## What this PR builds

The **Development-only Windows native runtime** can now answer bounded,
read-only inspection requests using the existing L1.1 control connection:

- Negotiate `entity.inspect`. Parse exact `component.inspect.request`
  frames, with session/instance identity, replay checks and capability
  checks performed by the existing authenticated transport.
- Capture `hierarchy.snapshot` on the **runtime frame thread** at most 4
  times/second; send only when its structural identity, parent or displayed
  name changes. A bounded, coalesced mailbox transfers the newest immutable
  snapshot to the socket worker without touching ECS from the socket thread.
- Resolve **authored GUID** entities and **runtime-only spawned entities**.
  Runtime IDs encode world ID, world epoch, entity index and generation;
  the surrounding protocol envelope makes them session/instance scoped.
  Deleted or recycled runtime IDs cannot alias an unrelated live entity.
- Inspect one selected Transform (position, quaternion, scale), Name, Tag or
  Active component using **const ECS reads** and exact schema/version checks.
  Replies carry source request ID plus current runtime revision; unsupported,
  missing or ambiguous identities are explicitly rejected using existing L1
  error codes. No source file, engine pointer or gameplay memory is sent.
- Bounded 64-request/64-result mailboxes avoid unbounded traffic. Stop clears
  any unconsumed work and sends no further snapshots.
- The trusted editor-main control server and Windows session expose a
  permission-checked `sendInspectRequest` API. The user-facing runtime
  Hierarchy/Inspector UI belongs to **L1.5b**, not this PR.

## Intentional MVP limits

The existing v1 `hierarchy.snapshot` is capped at **128 entities** and
has no pagination cursor. Larger worlds are explicitly marked
`complete: false`; this PR never claims a truncated result is complete.
Full paging, revisioned structural diffs, high-frequency component
subscriptions, inspector UI, log panel and disconnect/stale UI are later
L1.5 slices.

This is inspection of the **actual running packaged game world**. Until
the separate C9 track activates the captured unsaved scene, the runtime
may still be showing its original packaged startup scene. The editor must
label it as a runtime view, not silently treat it as the latest Authoring
World. C9 editor HTTP/asset publication remains entirely in Track B.
The Development-only bridge and Windows entry are not linked into Shipping.

## Verification

Cross-platform network tests validate strict request parsing, rejection
of malformed IDs and unbounded queues, mailbox result handling and latest
snapshot coalescing. Opt-in Windows Development ECS tests verify authored
and runtime-spawned identities, dynamic entity removal, stable IDs, actual
component readings after a live edit, unsupported schemas and stale
entities. Protocol tests validate the correlation field.
