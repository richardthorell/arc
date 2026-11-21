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


**Ownership:** #1135. The editor content server is #1137; Dev-only endpoint
overrides are #1136; publish-on-edit is #1138; runtime hot reload is #1139.
