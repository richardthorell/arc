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
- C9.4a wires **listener lifecycle**, but does not yet invoke
  prepareLivePlayCookedStaging / prepareLivePlayRemoteRelease from the
  editor's Run workflow. The server is empty until its trusted publish()
  method is called. This prevents an unfinished cook from being discoverable.
- The existing C9.3 Development endpoint override is not yet passed
  into the runtime, and runtime HTTP pairing and verified scene activation
  are **not** in this slice. The native baseline-active guard remains closed.
- Multiple target profiles may be served from one session; ambiguous
  texture-family variants require an explicit family selection. C9.4b/C9.7
  will map selected device/profile to the required variant.
- No authenticated LAN / remote exposure in this slice. Android USB should
  use ADB reverse to loopback after its authenticated native request path is
  integrated. Shipping must not accept a Development override.

## Follow-up C9.4b

Connect the existing L1.3 cook-and-publish pipeline to publish(), pass
the trusted non-secret C9.3 origin to the Development runtime, and deliver
per-session pairing authorization via the protected native handoff. Extend
conditional/range acceptance, lifecycle faults, and server diagnostics.
C9.4 remains open until authenticated Windows/Android clients and change-feed
replay/reset have been validated.

## Test coverage

editor/src/main/editorCookedContentServer.test.ts uses real loopback HTTP
and session-private files: start/stop and port reuse, unauthenticated and
path-traversal rejection, exact C9 release and object bytes, ETag/range bounds,
cross-project isolation, failed-generation last-good retention, symlink and
post-publication modification rejection, and concurrent target clients.
