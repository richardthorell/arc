# ARC Live Play L1.1 — Editor/runtime control protocol (v1)

**Status:** Versioned contract and validation only. No network listener, native client
codec, ECS command dispatcher, or remote launch is introduced in this PR.

**Tracking:** [Live Play L1 #1194](https://github.com/richardthorell/arc/issues/1194),
[L1.1 #1196](https://github.com/richardthorell/arc/issues/1196).
Next: [L1.2 #1197](https://github.com/richardthorell/arc/issues/1197) implements transport
and the native Development-runtime adapter.

## Boundary

This **control plane** carries tiny, revisioned editor commands and runtime inspection
data. It is separate from:

- [C9 content plane #1133](https://github.com/richardthorell/arc/issues/1133) and
  [C9.1 #1134](https://github.com/richardthorell/arc/issues/1134): cooked snapshots,
  manifests, immutable content-addressed blobs, incremental generation feed, and
  verified VFS/overlay activation. `content.generation` below only reports an active
  generation; it cannot carry artifact bytes or URLs.
- The editor's existing **in-process EditorHost** command protocol.
- Gameplay networking, remote frame/screen streaming, raw project file transfer, or
  a remote shell.
- [PIE session architecture](multiplayer-pie-architecture.md): Authoring World and
  Play World stay separate. Nothing here automatically saves authored files or
  copies runtime-only changes back into them.

The source for the editor's executable v1 validation and discriminated message
schemas is `editor/src/common/livePlayProtocol.ts`. A future C++ adapter must pass
these same contract examples/negative cases in cross-language tests rather than
introducing a second incompatible wire schema.

## Runtime mode and trust boundary

The native game runs the ordinary project Runtime role in **Development** with an
explicitly enabled editor bridge. Never expose the control endpoint in Shipping
builds, and never enable the control plane for normal local Editor Viewport Play.

One editor-owned session grants explicit capabilities to a *single project, build,
target profile, session, and runtime instance*. The editor's main process/managed
worker owns the listener and session lifetime; the renderer does **not** accept
arbitrary inbound native commands.

The transport must complete short-lived per-session **authentication/pairing before
it dispatches any JSON frame**, and pass `authenticated=true` to the validator only
after that check. Credentials are delivered through an OS-protected channel/pairing
handshake, not JSON payloads, CLI parameters, URLs, process args, log output,
source control, or cooked manifests. Only localhost binds by default. Android USB
is expected to use ADB reverse; LAN pairing requires explicit opt-in and
appropriate transport security.

An untrusted peer's `hello.capabilities` are **claims, not grants**. Capabilities
actually accepted by the editor remain constrained by the trusted session grant
and the compatible local runtime implementation. An `hello.accepted` reply must
advertise only the intersection; the L1.2 handshake will enforce this statefully.

## Wire format

A JSON UTF-8 message, **at most 64 KiB** including the envelope. All top-level and
typed payload objects are strict; unknown members, unknown message types,
unsafe numerical values, and malformed identities are rejected.

Every message carries this envelope:

| Field | Type | Semantics |
|---|---|---|
| `protocol` | `"arc.live-play"` | Separate control plane from C9 |
| `protocolVersion` | `1` | Exact match required, no downgrade |
| `sender` | `editor` / `runtime` | Verified against authenticated peer role |
| `projectGuid` | 32-hex or hyphenated GUID | Matches active project |
| `buildId` | bounded token | Matches loaded runtime binary/manifest |
| `targetProfile` | bounded token | Matches selected cook/build target |
| `sessionId` | bounded token | Play session identity |
| `instanceId` | bounded token | Single native runtime/Play World identity |
| `messageId` | bounded token | Unique message and ACK correlation ID |
| `sequence` | positive safe integer | Monotonically increasing **per sender** for the same session/instance; preserved on reconnect |
| `type` | known type | Determines typed `payload` |
| `payload` | bounded typed object | No arbitrary JSON executable commands |

IDs are bounded ASCII tokens (`A–Z`, `a–z`, digits, `._:-`). Entity identity is
never an ECS index/pointer.

### Example: editor commits a transform

```json
{
  "protocol": "arc.live-play",
  "protocolVersion": 1,
  "sender": "editor",
  "projectGuid": "11111111-1111-1111-1111-111111111111",
  "buildId": "dev-build-42",
  "targetProfile": "windows-x64-vulkan",
  "sessionId": "session-1",
  "instanceId": "runtime-1",
  "messageId": "edit-42",
  "sequence": 42,
  "type": "entity.property.set",
  "payload": {
    "entity": { "kind": "authored", "entityGuid": "22222222-2222-2222-2222-222222222222" },
    "component": { "typeId": "arc.transform", "schemaVersion": 1 },
    "property": "position",
    "value": [2.0, 1.5, 0.0],
    "authoringRevision": 12,
    "expectedRuntimeRevision": 8,
    "phase": "commit"
  }
}
```

The game responds with a `command.ack` identifying `requestId: "edit-42"`.
An applied ACK supplies `appliedRevision`; a rejection supplies a typed
`error.code` and safe message. Command application is deferred to a safe
simulation tick and **never** executed on a socket thread. The editor may send
intermediate `begin`/`update` updates during a drag; the final `commit` must be
delivered and ACKed. `cancel` reversals and gameplay/physics conflicts require
the L1.4 authority policy.

### Example: remote runtime reports hierarchy

```json
{
  "protocol": "arc.live-play",
  "protocolVersion": 1,
  "sender": "runtime",
  "projectGuid": "11111111-1111-1111-1111-111111111111",
  "buildId": "dev-build-42",
  "targetProfile": "windows-x64-vulkan",
  "sessionId": "session-1",
  "instanceId": "runtime-1",
  "messageId": "snapshot-7",
  "sequence": 7,
  "type": "hierarchy.snapshot",
  "payload": {
    "worldRevision": 32,
    "entities": [
      {
        "entity": { "kind": "runtime", "runtimeEntityId": "spawned-9" },
        "parent": null,
        "name": "Enemy"
      }
    ],
    "complete": true
  }
}
```

Snapshots contain at most 128 entity entries in a single frame. Large scenes
need chunked snapshots/continuation and explicit completeness handling in L1.5;
the above `complete` flag must not be assumed true for intermediate chunks.

## Message vocabulary

| Direction | Type | Purpose |
|---|---|---|
| Both | `hello`, `hello.accepted`, `hello.rejected`, `heartbeat` | Version/schema/capability negotiation and liveness |
| Editor → runtime | `session.command` | Pause/resume/stop/step (granted `session.control`) |
| Editor → runtime | `entity.property.set` | Authoring override with expected runtime and authoring revision (granted `entity.edit`) |
| Editor → runtime | `entity.create`, `entity.delete`, `entity.reparent` | Typed operations, future allowlisted execution under `entity.edit` |
| Editor → runtime | `component.inspect.request` | Query an entity's bounded component state (granted `entity.inspect`) |
| Editor → runtime | `session.resync.request` | Recover after disconnect, sequence gap or overflow |
| Runtime → editor | `session.state`, `log.record` | Runtime state/ticks and bounded diagnostics |
| Runtime → editor | `hierarchy.snapshot`, `hierarchy.changed` | Revisioned runtime hierarchy |
| Runtime → editor | `component.inspect.result` | Requested runtime component property values |
| Runtime → editor | `content.generation` | Status of independently verified C9 content generation |
| Both | `command.ack` | Correlated applied/rejected command response |

Unknown types or arbitrary `process.exec`/unrestricted script evaluation are
not legal. Property values in v1 are bounded primitives, null or numeric vectors
(maximum 16 elements). Complex component serialization is intentionally out of
scope until its schema and safe native handlers exist.

**Entity references:** authored objects use `{kind:"authored",entityGuid}`,
where GUID is stable across editor/runtime. Transient gameplay-spawned entities
use `{kind:"runtime",runtimeEntityId}` *only* inside a bound
`sessionId`+`instanceId`; they are never persistent authoring identities.
Native component schemas use a stable `typeId` and explicit integer
`schemaVersion`, not host pointers or arbitrary JSON fields.

## Validation, ordering, resync and implementation notes

- Incoming frames first pass the byte cap, authenticated-connection check,
  strict message schema, full project/build/profile/session/instance identity,
  authenticated sender-role/direction, sender sequence and trusted permission
  grants **before** any dispatch. The pure `receiveLivePlayFrame()` function
  returns an accepted typed message and `nextSequence` or a rejection code;
  it does not itself mutate worlds or session state.
- Peers keep a separate last accepted sequence for each sender. Duplicate/stale
  `sequence` is rejected. The transport's session owner maintains an optional
  bounded message-ID/ACK result cache to make retransmission idempotent; it
  must not execute a repeated command.
- On reconnect, reconnect to the same authenticated identity and retain the
  sender sequence; request `session.resync.request` with the last received
  sequence. If the bounded history cannot satisfy it, send a fresh
  `hierarchy.snapshot` and explicitly reset the subscriber's revision view.
  Never replay edits into a **different** session or runtime instance.
- Source/Authoring World never becomes gameplay authoritative by accident.
  Runtime physics/gameplay remains owned by Play World; L1.4 defines supported
  editor override/hold and conflict semantics. Only a future explicit
  Keep Simulation Changes operation could write runtime values to authoring.
- A saved content change goes through C9 incremental cook, hash verification,
  atomic generation commit and safe resource rebind. **Unsaved** simple property
  changes use the control plane; the initial dirty scene baseline is handled
  separately by L1.3 #1198. The editor must not implicitly save documents.
- No network/transport implementation, native codec or restart/reload
  capability is implied by accepting a schema. L1.2 must add a native test
  client/codec and cross-language golden-frame tests before opening a socket.

### Expected failure codes

`unauthenticated`, `unsupported_version`, `identity_mismatch`,
`schema_mismatch`, `permission_denied`, `unsupported_operation`,
`unknown_entity`, `stale_revision`, `restart_required`,
`invalid_request`, `session_expired`, `resync_required`, `busy`.

Transport/parser-only errors also include `frame_too_large`, `invalid_frame`,
`wrong_direction`, and `replayed_sequence`.

## Next integration milestones

1. L1.2 #1197: authenticated connection, native Development client and
   session/heartbeat/ACK state machine.
2. C9 #1134, #1135, #1136, #1137: versioned cooked catalog, verified cache,
   trusted development endpoint and editor-hosted content service.
3. L1.3 #1198 + C9 #1138: initial unsaved snapshot overlay and explicit-save
   cook/publish generations.
4. L1.4 #1199: tick-safe transform/property edits, coalescing and ownership.
5. L1.5 #1200: runtime Inspector, hierarchy subscriptions and state diagnostics.
6. #1140 / #379 / #411 / #1142: one-click Windows Play orchestration,
   subprocess management and end-to-end acceptance; then Android #403.
