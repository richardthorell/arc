# ARC HTTP — shared native HTTP foundation

## Purpose

\`arc-http\` is an **engine-level C++ HTTP module**, shared by native game
runtimes, standalone tools and native editor-host code. It is not editor-only
and is intentionally distinct from \`arc-network\` (gameplay replication and
the separate Development Live Play control bridge).

This first PR extracts the existing, working C9.2 HTTP transport from
\`arc-io\` without changing the wire format or request safety behavior:

| Module | Responsibility |
| --- | --- |
| \`arc-http\` | HTTP request/response types, bounded fetch on IO jobs, WinHTTP and Linux libcurl backends, strict HTTP transport validation |
| \`arc-io\` | Async file and VFS operations; no HTTP backends or TLS linkage |
| \`arc-assets\` | C9.1 release compatibility; SHA-256/CAS ingestion; generation closure and mounting |
| \`arc-network\` | Gameplay replication and optional Development Live Play TCP bridge |
| Editor (Electron main) | Future C9.4 HTTP server hosting the C9 content wire protocol |

Native users include \`<arc/http/content_http.h>\` and use the \`arc::http\`
namespace. \`arc-assets\` links \`arc-http\`, which links \`arc-io\` and
\`arc-jobs\`. The installed CMake target is \`ARC::HTTP\` and the installed
Linux SDK resolves its libcurl dependency.

### Portable HTTP contract

For now the concrete client request profiles in \`arc-http\` are the already
verified C9.2 immutable-content GET and mutable-release GET. These are
transport-level profiles, **not** new asset databases, game networking APIs,
or a generic arbitrary-URL browser. We should expose generic client/server
primitives as additional APIs of \`arc-http\` when required rather than
loosening the safe C9 resource/path restrictions.

WinHTTP and libcurl retain strict OS/system TLS and hostname verification,
no automatic redirects, hard response limits, cancellation and
verified status/content/range framing. Android TLS is a separate platform
follow-up; the NDK does not bundle libcurl.

### Reusing it with the editor-hosted server (C9.4)

Electron's main process already runs HTTP servers using Node's \`node:http\`
(e.g. the AI gateway). A C++ static library cannot be directly used by the
TypeScript main process without a native addon/bridge. Instead, we reuse
the **same HTTP wire contract** and the native client:

- Editor main can host a Node HTTP server implementing
  \`GET /releases/<project-guid>/<cook-profile>/current.json\`.
  It returns a complete, compatible C9.1 release descriptor, with no-store
  freshness semantics and bounded JSON payloads.
- Editor main can serve immutable
  \`GET /objects/sha256/<lowercase-digest>\` cooked-object bytes.
  It must honor exact full-object and ranged responses, Content-Length,
  Content-Range, strong ETag and If-Match, and immutable hashes.
- The **native runtime** (Windows / future Android) fetches both endpoints
  using \`arc-http\`, then \`arc-assets\` verifies SHA-256 and atomically
  mounts the complete cooked generation.
- Both sides implement the existing C9.1 schema and C9.2 request/response
  rules. Cross-language fixture tests should share protocol vectors from
  the existing C9 fixture files, not invent another editor-specific format.

For local editor/runtime sessions, use explicit loopback-only HTTP in
Development builds (including Android over ADB reverse). For physical LAN
and production use, require authenticated TLS with a trusted certificate;
never silently relax the current loopback-only plaintext restriction.

**Do not** tunnel cooked assets through the Live Play TCP control channel,
repurpose the existing AI gateway as a content server, or include gameplay
replication in \`arc-http\`. Live Play can advertise release versions and
request updates over its authenticated control plane; asset payloads flow
over the separate C9 HTTP endpoints.

### PR boundary / follow-ups

This extraction moves the C++ code, its native tests, dependency graph,
WinHTTP/libcurl platform linkage, and SDK export. It **does not**
implement the editor C9.4 content server, add an HTTP server implementation,
or introduce Android TLS. Those are separate PRs built upon this foundation,
followed by publish-on-edit and live resource refresh.
