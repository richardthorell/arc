# C9.1 — Remote cooked-content release contract (v1)

**Scope:** metadata and transport-independent validation only. This contract is
consumed by the implemented C9.2 HTTP client and the planned C9.4
editor-owned server. It does
**not** create a second GUID catalog, asset cooker, distribution manifest,
runtime update receiver or editor control channel.

## Publication and address model

A producer publishes a complete, immutable set of objects before atomically
publishing a small release descriptor. A plain static HTTPS host or a
development-only editor server can serve the **same bytes**.

| Resource | Suggested endpoint | Semantics |
|---|---|---|
| Current release | `releases/<project-guid>/<target-profile>/current.json` | Replace atomically; never cache as immutable |
| Immutable object | `objects/sha256/<64 lowercase hex characters>` | Exact bytes, declared SHA-256 and size |
| Dev change feed | `sessions/<session-id>/changes?after=<sequence>` | Optional, authenticated, monotonic event stream |

Paths are **relative to an independently configured, trusted HTTPS or
loopback origin**. The descriptor contains no arbitrary URLs, native paths,
secrets or executable modules. The transport must disable untrusted cross-origin
redirects and reject userinfo, unsupported protocols and unsafe origins. The
editor server defaults to loopback and requires authenticated session binding;
LAN access is opt-in, authenticated, and subject to resource limits. Production
CDN consumption does **not** require a live feed.

The `cookManifest` object is the existing `arc.cook-manifest` file, byte for
byte. `cookManifestName` is its original logical filename. Each entry in
`chunks` maps the **original** `.arcpak` filename referenced by that manifest
to an immutable content-hash object. C9.2 must parse/validate the cooker
manifest and prove every referenced chunk appears exactly once with consistent
length/hash before exposing the snapshot through the existing
`package_artifact_provider` or catalog/VFS. Do not rewrite artifact GUIDs,
schema IDs, dependencies or cooker-generated chunk offsets.

## Descriptor

```json
{
  "format": "arc.remote-content",
  "formatVersion": 1,
  "projectGuid": "11111111-2222-4333-8444-555555555555",
  "engineVersion": "0.1.0",
  "baseBuildId": "example-build",
  "target": {
    "platform": "windows",
    "architecture": "x86_64",
    "renderer": "vulkan",
    "cookProfile": "windows-x64-vulkan",
    "textureFamily": "bc"
  },
  "generation": 1,
  "complete": true,
  "cookManifestName": "windows-x64-vulkan.arccookmanifest",
  "cookManifest": {
    "path": "objects/sha256/aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
    "sha256": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
    "size": 1024
  },
  "chunks": [
    {
      "name": "startup-1234.arcpak",
      "path": "objects/sha256/bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
      "sha256": "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
      "size": 4096
    }
  ]
}
```

The digest/size values above illustrate shape only; they are not valid
checksums of a real cooker output. Android uses the same envelope with
`platform=android`, `architecture=arm64`, `renderer=vulkan`, an Android
cook profile, and `textureFamily=astc`. A build/target/profile mismatch must
fail closed. A headless server may declare no chunk entries, with a valid empty
cook manifest. Canonical publisher encoding is sorted-key UTF-8 JSON with no
insignificant whitespace, a final newline and finite numbers only. See
`tools/remote_content_contract.py` for the canonical publisher validator
and `engine/assets/src/common/remote_content.cpp` for the native validator.

## Snapshots, change feeds and rollback

Publish in this order: cook and verify all artifacts; upload immutable objects;
validate manifest/chunk closure; publish a complete descriptor atomically; then
advance the current-release reference and notify live subscribers. Clients
never read objects from a partially published generation.

A development feed references the published `generation` and uses its own
monotonic event `sequence`; these numbers must not be conflated. A subscriber
acknowledges the last applied sequence only **after** activating a complete
verified generation. Reconnect with `after=<last-ack>`. If the journal no
longer contains the next sequence, return `full-snapshot` and refetch the
current release; if the client claims a sequence newer than the server's, reject
and reauthenticate/reset. Duplicate events are idempotent. A change event may
include added/updated/deleted artifact GUIDs, but never contains raw source
bytes or runtime control commands.

Source selection uses ARC's existing provider priority: packaged Boot content
(available offline), verified persistent CAS cache, remote immutable content,
and the highest-priority *committed* developer/hotfix overlay. Tombstones stop
fallback; an unverified source must not replace last-good content. The existing
`runtime_update_receiver` verifies hashes, target/base-build identity and
atomic activation. C9.2 transports bytes into it; C9.6 owns active live reload
and safe handle generations. A failed publication/download or content update
must leave the last verified generation available. Cache policy may pin Boot
and current/last-good generations.


## C9.1c — Versioned development change-feed *response* (v1)

C9.1a/1b defined release bytes, immutable object addressing and a reconnect
decision. C9.1c completes the shared **wire response contract**, including
native C++ and Python validators and common fixtures. These fixtures are
transport-independent, and cannot replace future authenticated server or
real-device acceptance. A change feed is optional
and **development-only**; static CDN releases do not need to implement this
endpoint.

The editor's authenticated read-only endpoint is:

`GET sessions/<session-id>/changes?after=<last-acknowledged-sequence>`

The client must send its last **successfully activated** event sequence,
not the last event merely received or downloaded. `sequence` orders feed
notifications; `generation` identifies complete cooked release snapshots.
Each event refers to the **immutable, SHA-addressed canonical release
descriptor**; it never directs a client to race a mutable `current.json`
pointer. This also means events are replayable when a later generation has
already advanced the current pointer.

For example, a Windows/Vulkan/BC development client after ACK 4 receives:

```json
{
  "format": "arc.remote-content-feed",
  "formatVersion": 1,
  "projectGuid": "11111111-2222-4333-8444-555555555555",
  "engineVersion": "0.1.0",
  "baseBuildId": "build-abc",
  "target": {
    "platform": "windows",
    "architecture": "x86_64",
    "renderer": "vulkan",
    "cookProfile": "windows-x64-vulkan",
    "textureFamily": "bc"
  },
  "afterSequence": 4,
  "oldestSequence": 4,
  "newestSequence": 6,
  "currentGeneration": 10,
  "currentRelease": {
    "path": "objects/sha256/cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc",
    "sha256": "cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc",
    "size": 702
  },
  "mode": "replay",
  "events": [
    {
      "sequence": 5,
      "generation": 9,
      "release": {
        "path": "objects/sha256/dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd",
        "sha256": "dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd",
        "size": 700
      },
      "changed": ["aaaaaaaa-bbbb-4ccc-8ddd-eeeeeeeeeeee"],
      "removed": []
    },
    {
      "sequence": 6,
      "generation": 10,
      "release": {
        "path": "objects/sha256/cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc",
        "sha256": "cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc",
        "size": 702
      },
      "changed": ["44444444-5555-4666-8777-888888888888"],
      "removed": ["aaaaaaaa-bbbb-4ccc-8ddd-eeeeeeeeeeee"]
    }
  ]
}
```

The hashes here are illustrative. The runnable contract fixture is
`engine/assets/tests/fixtures/remote-content-feed-v1.json`. Android uses
**the same envelope and GUIDs** with `platform=android`,
`architecture=arm64`, `renderer=vulkan`,
`cookProfile=android-arm64-vulkan`, `textureFamily=astc`. Each
target/profile has its **own compatible cooked descriptor and objects**;
a Windows client must reject an Android target's response, and vice versa.

### Cursor, atomic publication and reset rules

- `afterSequence` must **exactly match** the requested last ACK. The
  authenticated project, engine/base-build and all five target fields must
  match the session runtime.
- Journal `oldestSequence >= 1`, `newestSequence >= oldestSequence`
  and `currentGeneration >= 1` describe the published head. A server with
  **no first complete snapshot** must answer not-ready, not invent generation 0.
- For `mode=replay`, every sequence from `afterSequence+1` to
  `newestSequence` must be present exactly once and in order. Generations
  strictly increase; the final event's generation and descriptor SHA/length
  must equal `currentGeneration/currentRelease`. Changed/removed GUID lists
  are bounded, duplicate-free and disjoint. The event is an advisory change
  summary: clients still validate the full immutable descriptor and its
  cooked dependency graph before activation.
- At most **256 events** and **8 MiB** of feed metadata may be replayed.
  If the next sequence has been trimmed or more than 256 events are needed,
  respond `mode=full-snapshot` with `events=[]` and a verified
  `currentRelease` reference. Clients fetch/verify this immutable descriptor,
  perform a full reconciliation, and ACK `newestSequence` **only after**
  successful activation. A failed activation keeps the previous ACK.
- If the client is already at `newestSequence`, respond
  `mode=up-to-date` with an empty event list. If its cursor is **ahead**,
  reject/regenerate the session rather than pretending no changes occurred.
- Snapshot/reset/up-to-date responses may **not** contain partial replay
  events. Never expose an incomplete cook manifest, a partially staged object,
  or a notification before its SHA-addressed descriptor and all referenced
  immutable cooked objects have been published.

C9.4d now implements the actual authenticated **polling** server and
bounded, per-target journal. The C9.1c base route accepts
`?after=<cursor>`; when multiple target profiles share the same editor
session, clients must also supply `&profile=<cook-profile>` to avoid a
cross-target feed mix-up. A complete committed release descriptor is served
by immutable SHA under `objects/sha256/` *before* any journal event is
visible. C9.5 supplies explicit changed/removed GUID summaries only after a
new successful cook/publish; C9.6 owns verified native update activation
and ACK advancement. The transport-neutral C9.1 wire contract is unchanged.

### Static CDN and offline compatibility

A published release on S3/CDN or a static file server needs only the mutable
`releases/<project-guid>/<target-profile>/current.json` pointer and immutable
`objects/sha256/<digest>` files. It has **no feed** and no session endpoint;
clients poll/discover current release per C9.2 and apply the same target,
generation and SHA validation. Existing packaged Boot and verified persistent
CAS can be used offline. Remote/hotfix overlays retain their existing
priority, with tombstones and last-good rollback handled by the ordinary
catalog/VFS/receiver, not by feed-provided raw asset paths.

The feed may name **only** C9 GUID changes and immutable descriptor hashes;
tokens, signed URLs, authoring filesystem paths and executable commands
are forbidden. Authentication and rate limits are enforced by C9.4, not
by this transport-neutral parser. LAN listening requires deliberate,
authenticated pairing; development loopback and USB ADB reverse are preferred.

## Range requests and trust boundaries

Immutable objects support `GET`, `HEAD` and optional `Range: bytes=start-end`;
responses must be checked for exact range boundaries, total length, content
length, SHA-256 of complete objects, strong ETag consistency and maximum byte
budgets. A `200` response to a range request must be explicitly handled, never
misinterpreted as a partial response. Signed-URL handoff is allowed only from a
trusted authenticated origin to an allowlisted HTTPS host with bounded expiry;
no access token is persisted in the descriptor or logged. C9.2 owns these
network checks; this first PR validates *metadata*, not HTTP responses.

## Completion / downstream implementation owners

- **C9.1a — PR #1206:** canonical immutable release descriptor and optional
  change-feed recovery decision.
- **C9.1b — PR #1208:** fail-closed native descriptor parsing, C++/Python
  fixture parity, target and generation validation.
- **C9.1c — this PR:** bounded change-feed response schema, SHA-addressed
  immutable release events, replay/reset validation, Windows/Android examples
  and cross-language negative tests. **Completes the C9.1 contract scope.**
- **C9.2** (#1135): existing HTTP transport/cache and VFS acquisition; any
  outstanding transport diagnostics belong to C9.2h/C9.9.
- **C9.3–C9.4** (#1136/#1137): runtime source overrides and authenticated
  editor-owned service implementation.
- **C9.5–C9.6** (#1138/#1139): incremental publisher and safe in-game
  application of subscribed cooked updates.
- **C9.9** (#1142): external Windows/Android and fault-injection acceptance.

The authenticated Live Play control/inspection plane (#1194) remains separate
from this cooked-data delivery protocol.
