# C9.2 — Cooked-content HTTP transport implementation plan

The first C9.2 PR introduces the **native HTTP request/response safety boundary** in
`arc::io` (`content_http.h`). It is distinct from the existing C9.1
release parser in `arc::assets`: C9.1 validates a publication descriptor,
while C9.2 validates bytes actually obtained from an HTTP transport.

## C9.2a — fail-closed validation (this PR)

- Accept only HTTPS origins; allow development loopback HTTP behind an
  explicit non-default option. No userinfo, origin paths, fragments, invalid
  ports, embedded URLs, or arbitrary authority redirects.
- Fetch only C9.1 immutable `objects/sha256/<digest>` paths.
- Preflight object-size limits and request interval bounds without integer
  overflow; reject inconsistent range start/length.
- Full fetch requires HTTP 200 and the **exact declared object size**.
  Byte-range fetch requires HTTP 206, matching Content-Range start/end/total,
  declared Content-Length, and exact response body length.
- Reject HTTP redirects instead of following them; require valid strong ETags,
  and ensure an If-Match value agrees with the returned ETag.
- Use structured error categories and deterministically test each safety
  decision without Internet access.

These checks apply **after** a backend has received bytes; they do not
replace native TLS verification, hash verification, streaming quotas or the
existing `runtime_update_receiver` atomic publication checks.

## C9.2b — native Windows HTTPS request executor (this PR)

Implement cancellable, asynchronous I/O on ARC job-system IO workers. The first
backend uses Windows WinHTTP, with OS CA validation, automatic redirects
disabled and strict body-length checking. Other platforms explicitly return
\`transport_unavailable\` rather than bypassing HTTPS or supplying fake bytes.
Remaining backend and hardening requirements:

1. Trust the OS CA store; verify hostname/certificate and never disable TLS.
2. Disable automatic redirects; validate origin **before** DNS/connect.
3. Support GET and explicit Range/If-Match headers; the Windows backend uses
   bounded WinHTTP connection/read timeouts and observes cancellation between
   blocking calls and read chunks. HEAD, interruption of a blocked OS call,
   and bounded retries/backoff remain follow-ups.
4. Bound all received bodies **during** transfer (not only after receipt) to
   prevent unbounded buffering. Report 401/403/404/429 distinctly.
5. Avoid blocking render/main threads. Add local WinHTTP HTTP/TLS fixtures
   and fault injection for stalls, cancelled transfers, truncation and malformed
   ranges as follow-up acceptance. Production non-loopback requests remain HTTPS-only.
6. Add native Android/Linux TLS implementations using the same interface and
   strict response validator, without editor-only dependencies.

## C9.2c — content-aware cache/VFS integration

Use C9.1 metadata validation, then download the *existing cook manifest*
and chunk objects. Verify SHA-256 plus sizes and complete manifest-to-chunk
closure, stage under persistent CAS, and activate through ARC's existing
`package_artifact_provider`, `cooked_asset_catalog`,
`cas_overlay_provider` and `runtime_update_receiver`. Preserve offline
last-good versions and reject incompatible target/base build. Never expose URLs
to game-side GUID consumers.

## C9.2d1 — cache-first generation acquisition (this PR)

After a release descriptor has passed C9.1 target/project/build/generation
validation, \`acquire_remote_generation\` ties the previous slices together:

1. Validate the bounded object table and byte budget before issuing requests.
2. Check each declared immutable cook manifest/chunk object in the existing
   verified derived-data CAS. Offline mode requires all objects locally.
3. Fetch only missing objects (or fetch all for diagnostics) via the C9.2b
   HTTPS client, running on dedicated IO workers. The coordinating job runs on
   a general worker, with one request in flight to bound memory.
4. Ingest complete responses only through C9.2c1 SHA-verified CAS persistence.
5. Re-run C9.2c2 whole-generation manifest/chunk/artifact closure before
   reporting a successful acquisition.

This is an *acquisition*, not an activation. The existing
\`cooked_asset_catalog::mount_remote_generation\` call is explicitly separate
and must happen only after acquisition succeeded. A failure or cancellation
never changes the currently mounted generation. Tests inject deterministic
IO-worker HTTP responses while exercising the same request validation and CAS
integrity checks.

Not yet included: fetching the mutable \`current.json\` release descriptor,
parallel/coalesced downloads, retry/backoff, quota and generation pinning,
native Android/Linux HTTPS backends, or automatic publish/activation. Future
C9.2 work handles these independently.


## C9.2d2 — mutable current release discovery (this PR)

`discover_remote_release` closes the gap between C9.1 descriptor
parsing and C9.2d1 immutable generation acquisition:

1. Request only `releases/<project-guid>/<target-profile>/current.json`.
   The GUID/profile path components and origin are validated before networking.
2. Windows WinHTTP uses the OS TLS/certificate store, disables redirects, and
   sends no-cache request headers. The read has a strict 8 MiB maximum, even
   if Content-Length is missing (chunked transfer); HTTP failures are
   structured. Only HTTPS or explicitly allowed plaintext *loopback* is valid.
3. Revalidate status, redirection and body framing regardless of the fetch
   executor, then parse with the existing C9.1 descriptor validator. Exact
   project/engine/base-build/cook-target identity, complete generation, and
   monotonic `minimum_generation` are mandatory.
4. Return a parsed release candidate for C9.2d1 acquisition. Discovery alone
   cannot install, stage or mount content, and stale/bad responses leave the
   existing catalog untouched.

The mutable document is **never** inserted into the immutable object CAS.
Native Linux/Android HTTP backends, poll scheduling, retries/backoff, pinning,
cache policy, ETag-based 304 refresh and automated activation are separate PRs.

## C9.2e1 — Linux system-TLS HTTP transport

Linux now uses libcurl (when the platform development package is available)
for both immutable `objects/sha256/<digest>` and mutable release discovery.
ARC continues to perform the existing C9.2 preflight and response framing
checks and C9.2c SHA-256/CAS validation; libcurl does not change asset identity.

- Enforces the system CA trust store and hostname verification, with HTTPS
  required except for explicitly enabled **loopback-only** HTTP.
- Does not follow redirects or allow non-HTTP(S) protocols; protects against
  unexpected URL changes, content encodings, malformed/oversized headers and
  bodies larger than the declared immutable object or 8 MiB release bound.
- Runs on ARC IO workers and responds to cancellation, timeouts, Range and
  If-Match headers. Release lookups send no-cache headers.
- Tests exercise real local HTTP transfers for full objects, byte ranges,
  current-release requests and failure modes without Internet access.
- CMake detects the system libcurl development package; installed Linux SDK
  configs restore the link dependency via `find_dependency(CURL)`. Linux CI
  installs the package explicitly. Missing development libraries leave the
  backend unavailable rather than silently falling back to insecure HTTP.

Android is **not** covered by this backend: the NDK does not ship libcurl.
Android TLS transport, shared retry/coalescing policies, cache quota and
end-to-end device acceptance remain separate C9.2 follow-ups.

**Ownership:** #1135. The editor content server is #1137; Dev-only endpoint
overrides are #1136; publish-on-edit is #1138; runtime hot reload is #1139.

## Shared engine HTTP extraction

As of C9.2 module extraction, the transport and its platform backends live in
\`engine/http\` / \`arc-http\` (public \`arc::http\`, \`<arc/http/content_http.h>\`),
not in \`arc-io\`. All existing C9.2 response validation, TLS requirements,
immutable byte framing, cancellation and cache identity stay in place.

\`arc-assets\` depends on \`arc-http\`. The network transport is no longer
coupled to VFS or packaged runtime hosting. \`arc-io\` remains responsible for
filesystem, async file services and VFS. See \`docs/arc-http.md\` for the
editor/runtime integration boundary.
