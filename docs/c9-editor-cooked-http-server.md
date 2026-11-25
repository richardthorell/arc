# C9.4a — Editor-hosted, session-scoped cooked content HTTP server

Tracking: [Runtime Content #1137](https://github.com/richardthorell/arc/issues/1137),
[C9 #1133](https://github.com/richardthorell/arc/issues/1133),
[L1.3 #1198](https://github.com/richardthorell/arc/issues/1198).

## Scope of this first C9.4 slice

The editor main process now owns an instance of EditorCookedContentServer
for **external Windows Standalone Play only**. It starts after L1.3
captures a private unsaved scene, and closes **before** that private staging
directory is deleted on Stop, failure, project switch/close or shutdown.

The HTTP service is **not** a file server. A trusted main-process caller
must first supply a complete LivePlayRemoteRelease returned by the existing
L1.3 verified cook/immutable publisher. The service independently checks
project/runtime-build/session identity, approved target, generation, descriptor
hash and object inventory, SHA-256 and sizes, and rejects symlinks and
non-private paths. Candidate verification completes before any new catalog
becomes visible; rejection keeps the previous published generation.

Supported C9.1 data-plane GET paths:

- /releases/<project-guid>/<cook-profile>/current.json — exact verified
  mutable current descriptor for that profile; no-store.
- /objects/sha256/<64-lowercase-hex-digest> — only SHA-addressed objects
  referenced by an approved complete descriptor. Full GET up to 256 MiB,
  otherwise bounded explicit ranges with HTTP 206/Content-Range, ETag and
  If-Match. Client runtime must SHA-verify all acquired objects as in C9.2.

The server binds strictly to 127.0.0.1, uses a private ephemeral port by
default, checks HTTP authority and strict unescaped paths, caps open
connections, header count, request rates and range bytes, and rejects other
HTTP methods. It requires a random expiring session Bearer token, which is
issued **only to trusted editor main code**; it never appears in a URL, argv,
log, source-controlled project settings, or renderer-facing IPC. The C9.4b
native pairing/handshake needs to transport that token over protected IPC,
not process arguments.

The server snapshot contains safe diagnostics: origin, project/session,
profile generation list, connected clients, active transfers, served bytes,
requests and rejects — never a secret.

## Important behavior and boundaries

- **Editor Viewport Play** still starts no HTTP server.
- L1.3 authoring capture remains unsaved and is never exposed as an HTTP
  resource. The published objects are immutable cooked artifacts.
- C9.4a established listener lifecycle; **C9.4b** now invokes
  prepareLivePlayCookedStaging and prepareLivePlayRemoteRelease from the
  trusted editor Run workflow before using the server's publish() method.
  An incomplete cook is never discoverable.
- The existing C9.3 Development endpoint override is not yet passed
  into the runtime, and runtime HTTP pairing and verified scene activation
  remain deferred to subsequent C9.4/C9.6 work. The baseline-active guard
  remains closed.
- Multiple target profiles may be served from one session; C9.4b selects a
  unique compatible texture-family from the trusted cook-profile policy and
  rejects ambiguous matches until device-specific pairing (C9.4c/C9.7).
- No authenticated LAN / remote exposure in this slice. Android USB should
  use ADB reverse to loopback after its authenticated native request path is
  integrated. Shipping must not accept a Development override.

## C9.4b — Verified initial Live Play publication

The editor's trusted standalone Play handler now invokes the complete
L1.3 cook/package/verify and immutable-release assembler before starting
the standalone process. C9.4b resolves the cooker from the selected SDK,
rechecks the captured project and session identity, and uses the selected
profile's platform/architecture/renderer/texture policy to select one
unambiguous target-compatible release variant. It then promotes only that
verified complete descriptor through the C9.4a server.

A session-owned AbortController cancels the private cooker and prevents
promotion if Play is stopped or the active project changes. Shutdown waits
for the cooker before removing the private authoring scene staging. Failed
cooks, invalid target families, changed scene artifacts and rejected releases
never appear as active HTTP generation. No global project-side "current"
file is written or modified.

The server is now populated with its first verifiable generation, but the
native runtime still does **not** receive its endpoint or pairing credential.
That protected handoff is C9.4c. C9.5 handles new post-start edits and
further generation commits, while C9.6 owns native verified scene activation.
The baseline active guard therefore remains closed here.

## Follow-up C9.4c and later

Deliver per-session authorization over protected native IPC, connect
Windows/Android clients (including ADB reverse for USB devices), and handle
explicit target pairing. C9.4d adds the authenticated bounded sequence/change-feed replay and reset
endpoint described below; C9.4e finalizes diagnostics and security/acceptance. Issue #1137 remains open
until those are complete.

## Test coverage

editor/src/main/editorCookedContentServer.test.ts uses real loopback HTTP
and session-private files: start/stop and port reuse, unauthenticated and
path-traversal rejection, exact C9 release and object bytes, ETag/range bounds,
cross-project isolation, failed-generation last-good retention, symlink and
post-publication modification rejection, and concurrent target clients.

C9.4b integration coverage: editor/src/main/livePlayContentPublication.test.ts
runs cook → verify → package → freeze → publish → authenticated catalog/blob GET,
plus cancellation, stale project, invalid artifact, identity and target-family
rejection tests using private filesystem fixtures.

## C9.4d — Authenticated generation change-feed

The C9.1c [versioned feed contract](remote-content-protocol.md) is now served
by the editor-owned HTTP listener. All requests require the existing expiring
Bearer token, strict IPv4 loopback authority, the selected session identity and
the same request-rate/size bounds as immutable cooked content.

**Polling route** (no SSE or WebSocket in this slice):

`GET /sessions/<session-id>/changes?after=<last-ACK-sequence>`

When one target is published, it can be omitted. If multiple cook profiles
share a session, select the intended target explicitly:

`GET /sessions/<session-id>/changes?after=<last-ACK-sequence>&profile=<cook-profile>`

Each target gets an independent, monotonic **event sequence**. Its published
`generation` is a separate monotonically increasing identifier; a reconnect
ACK must refer to the last **successfully activated** event, not the last
event merely received. The server keeps up to 256 complete events (bounded
`journalLimit` may be lowered in tests). A client at the current cursor
gets `up-to-date`; a client whose next event has been trimmed or whose replay
would exceed 256 events / 8 MiB gets `full-snapshot` and a verified current
immutable descriptor reference. A cursor ahead of server state receives HTTP
409, not a false success. Before any complete publication, the endpoint
returns 503; a missing target/profile returns 404, and an ambiguous target
returns 400. Malformed or extra query parameters are rejected.

Every successfully published SHA-verified release descriptor is itself exposed
at `/objects/sha256/<descriptor-sha>`, so reconnecting clients can fetch the
**exact historical** snapshot identified by the event rather than race a
changed `current.json`. The immutable descriptor is indexed alongside its
fully verified cook manifests/chunks **before** its event enters the journal.
Rejected cooks, unchanged/incorrect generations, cancellations and server
shutdown never advertise a partial generation. Journal and object inventory
are cleared on session stop. Events carry only validated sorted, disjoint
asset GUID lists and immutable descriptor hashes, never raw authoring files
or secret credentials.

The initial baseline event lists its registered scene-asset GUID.
Subsequent revisions must supply trusted explicit `changed`/`removed`
asset GUID sets when calling `publish`; C9.5 is responsible for creating
those after successful save/cook. This server does **not** infer changed
asset GUIDs from manifests or claim native scene hot reload (C9.6). The
C9.1c native parser already validates response continuity and target identity;
runtime feed polling/application is a separate consumer integration.

Coverage: `editor/src/main/editorCookedContentServer.test.ts` checks the
authenticated initial event, immutable old descriptor fetch, per-profile
journals, exact replay, up-to-date, truncated-history reset, cursor regression,
malformed URL/cursor, failed publication retaining last-good, and shutdown.

## C9.4e — Diagnostics, file-descriptor hardening, and acceptance

The editor's Live Play preload bridge now provides a **read-only**
`window.arc.livePlay.contentServerSnapshot()` call. The main process returns
`null` without an active server/project match. Otherwise it returns
`EditorContentServerSnapshot` with the loopback origin/port/bind mode,
project/session identifiers, published profile/generation and per-target feed
cursors, active clients/transfers, connection and request-rate limits, object
index hits and responses, release/feed responses, full-snapshot resets, total
requests/rejections/served bytes, failed publications, and the most recent
**bounded failure code**. No bearer tokens, authorization header values,
private paths, raw network errors or original request URLs are returned.
Metrics are main-process-owned and reset for a fresh server instance.

The C9.4e object serving path opens a **single pinned file descriptor** and
validates the opened object's owner-independent file identity and saved
size/inode/timestamps against the publication record, rejecting symlinks
(including paths swapped after the earlier publication check). The same
verified descriptor is handed to Node's read stream with `autoClose`. A
rename/replacement after validation therefore cannot make that response
stream a different file; transfer abort and normal completion close the
descriptor. A later request against a modified path is rejected.
Immutable object hashes remain verified during cook publication and by the
native C9.2 receiver on download; this server's metadata check does **not**
replace the recipient's SHA verification.

The session token is SHA-256 stored and expires after 15 minutes; reissuing
it clears and immediately revokes the previous hash. Bind remains **strictly
127.0.0.1** with no LAN control: paired Android uses explicit-serial
`adb reverse`. This is a deliberate default-deny posture; enabling LAN is
**not** implemented here and would require a separate user-authorized
security design. The server caps connections (16), headers (64), requests
(100 per second), URL size, object transfers (256 MiB each), and feed
replay (256 events / 8 MiB). A lower rate limit is injectable only to
exercise deterministic acceptance tests.

The C9.4e editor tests cover loopback bind/port collision/reuse, token
rotation and unauthorized access, simultaneous independent project listeners,
cross-session/cross-target isolation, rate-limit rejection, safe diagnostic
counters and no-secret serialization, malformed/modified objects, an
on-disk pathname swap during a pinned file transfer (non-Windows), and
server stop/restart with no stale credentials. Existing C9.4a–d tests cover
range/ETag, tampered publish, symlink escapes, verified catalog, replay
and rollback.

C9.4 is a **development-only cooked-content service**, not a remote
editing API or shipping server. A successful C9.4e PR can close #1137.
Later C9.5 authors additional post-edit generations, and C9.6/C9.7
consume/activate remote changes; they are not silently considered complete
by the server's acceptance tests.
