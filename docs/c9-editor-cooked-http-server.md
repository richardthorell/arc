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
explicit target pairing. C9.4d adds sequence/change-feed replay/reset;
C9.4e finalizes diagnostics and security/acceptance. Issue #1137 remains open
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
