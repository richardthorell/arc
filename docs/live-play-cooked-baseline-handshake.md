# Live Play L1.3b1 — Cooked scene baseline control handshake

Tracking: [#1198](https://github.com/richardthorell/arc/issues/1198), [#1194](https://github.com/richardthorell/arc/issues/1194).

The **L1.3a** authored scene snapshot from #1211 remains an immutable,
session-private `.arcscene` input; it is **not** a cooked C9 release and is
never sent to the native game via control IPC.

This PR adds the bounded **metadata-only handshake** required for L1.3b:
an authenticated editor may offer a **C9-verified, target-compatible**
cooked release identity to the connected native Development runtime. The
native transport validates the envelope, authority, negotiated
`content.observe` capability, scene GUID, revision, digests and generation.
It returns the offer to the runtime caller without doing scene IO or ECS
mutation. Only the C9 loader (future integration) can decide that the
offered release is verified, mounted and activated. Its caller explicitly
reports `active` or `rejected` using the native client's
`send_baseline_status` after attempting that operation.

Messages share the L1.1 v1 envelope (project/build/profile/session/instance
and per-sender sequence) and do **not** contain paths, URLs, asset bytes
or authentication tokens.

`scene.baseline.offer` (editor -> native):
- `sceneGuid`: stable authored scene GUID
- `authoringRevision` and `authoringSha256`: captured unsaved authoring
  baseline revision and immutable authored bytes digest, for provenance
- `cookedGeneration` and `cookedReleaseSha256`: the C9-verified
  target-compatible cooked release and its monotonically scoped generation

`scene.baseline.status` (native -> editor):
- `requestId` correlates the exact offer
- `sceneGuid`, `cookedGeneration` and `cookedReleaseSha256` match it
- `status` is `active` (reason `none`) or `rejected` (reason
  `unavailable`, `incompatible`, `corrupt` or `unsupported`)

The server refuses offers before handshake or without negotiated
`content.observe`. Only one offer may be outstanding; an unsolicited,
duplicate or identity-mismatched status closes the peer. Disconnect/Stop
drops pending state. The native bridge rejects invalid metadata or
unsupported editor commands rather than silently executing them.
Golden wire fixture and TS/C++ transport tests cover the contract.

**Not yet wired into Play:** The editor has no verified ephemeral cooked
release to offer until C9 #1137/#995 and L1.3b2 authoring-to-cooked
materialization arrive. Accordingly, the existing Windows Play path
continues to connect without a scene-loaded guarantee. No fake "active"
ACK, raw scene path transmission, automatic save, or accidental C9
duplication is introduced here. L1.3b2 owns publishing the ephemeral
target-cooked generation, issuing this offer, waiting for confirmation,
and connecting game startup to its activated scene.
