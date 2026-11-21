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

## C9.2b — actual HTTPS request executor (next PR)

Implement cancellable, asynchronous I/O on ARC job-system IO workers with an
appropriate supported platform backend (e.g. Windows WinHTTP and an Android
TLS implementation). Backend requirements:

1. Trust the OS CA store; verify hostname/certificate and never disable TLS.
2. Disable automatic redirects; validate origin **before** DNS/connect.
3. Support GET/HEAD, explicit Range and If-Match headers, connection/read
   timeouts, request-level cancellation and bounded backoff.
4. Bound all received bodies **during** transfer (not only after receipt) to
   prevent unbounded buffering. Report 401/403/404/429 distinctly.
5. Avoid blocking render/main threads. Test using a local HTTP fixture and
   fault injection for stalls, cancelled transfers, truncation and malformed
   ranges. Production non-loopback requests remain HTTPS-only.

## C9.2c — content-aware cache/VFS integration

Use C9.1 metadata validation, then download the *existing cook manifest*
and chunk objects. Verify SHA-256 plus sizes and complete manifest-to-chunk
closure, stage under persistent CAS, and activate through ARC's existing
`package_artifact_provider`, `cooked_asset_catalog`,
`cas_overlay_provider` and `runtime_update_receiver`. Preserve offline
last-good versions and reject incompatible target/base build. Never expose URLs
to game-side GUID consumers.

**Ownership:** #1135. The editor content server is #1137; Dev-only endpoint
overrides are #1136; publish-on-edit is #1138; runtime hot reload is #1139.
