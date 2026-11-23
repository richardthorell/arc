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
`transport_unavailable` rather than bypassing HTTPS or supplying fake bytes.
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
validation, `acquire_remote_generation` ties the previous slices together:

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
`cooked_asset_catalog::mount_remote_generation` call is explicitly separate
and must happen only after acquisition succeeded. A failure or cancellation
never changes the currently mounted generation. Tests inject deterministic
IO-worker HTTP responses while exercising the same request validation and CAS
integrity checks.

Not yet included: fetching the mutable `current.json` release descriptor,
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
`engine/http` / `arc-http` (public `arc::http`, `<arc/http/content_http.h>`),
not in `arc-io`. All existing C9.2 response validation, TLS requirements,
immutable byte framing, cancellation and cache identity stay in place.

`arc-assets` depends on `arc-http`. The network transport is no longer
coupled to VFS or packaged runtime hosting. `arc-io` remains responsible for
filesystem, async file services and VFS. See `docs/arc-http.md` for the
editor/runtime integration boundary.

## C9.2f1 — Demand-loaded verified cooked chunks

`prepare_lazy_remote_generation(...)` is a metadata-first alternative to
`acquire_remote_generation(...)`. Given a previously C9.1-validated release,
it fetches (or reuses) only the SHA-verified `.arccookmanifest`, checks the
target/build, chunk declarations, artifact mapping, non-overlapping ranges
and the declared bounds, and returns an **unmounted** lazy provider.
The caller explicitly publishes the metadata at a safe runtime boundary via
`cooked_asset_catalog::mount_remote_provider(provider)`.

The normal cooked Asset Database and VFS resolve logical GUID artifacts
without exposing URLs. On the first read of an uncached artifact, a general
worker coordinates an IO-worker `arc-http` GET for its **entire** containing
immutable `.arcpak` object; existing SHA-256/response validation and
persistent CAS ingestion run before any bytes are returned. The provider
then checks ARC package magic, declared offsets and the addressed artifact
hash from the cook manifest. Later reads use the verified cached object.
Offline-only metadata mounting returns a typed cache miss on uncached reads.

**Security and availability boundaries:** the lazy provider reports
metadata availability before all referenced chunk bodies are verified.
Corrupt, cancelled or offline reads fail closed and never leak unverified
bytes, but lazy mounting does not automatically switch the VFS to a
lower-priority last-good provider on read failure. A runtime may retain
and explicitly reactivate that provider. The original fully verified
acquisition/mount path remains unchanged for callers needing all-or-nothing
generation activation. This is demand-loading by **chunk**, not a byte-range
HTTP streaming optimization: whole-object SHA-256 must pass first.
Concurrent duplicate coalescing, retries, transfer budgets, persistent
quota/pinning and runtime startup/orchestration are follow-on C9.2f2/g
and C9.3 tasks.

The new Catch2 coverage uses actual cooker-generated ARC package fixtures and
an injectable IO-worker fetcher; no external service or credentials are needed.

## C9.2f2 — Coalesced immutable object transfer and bounded retries

The shared `remote_object_download_coordinator` provides an explicit,
reusable transfer-policy boundary for `arc-assets`. A coordinator can be
shared across lazy remote providers and eager generation acquisition, provided
the callers use the same ARC job scheduler and derived-data cache instance.

- Every SHA-256 CAS miss is keyed by immutable object hash: concurrent
  requests for the same object join one in-flight leader, then verify the
  completed CAS entry. A cancelled follower returns immediately without
  cancelling an unrelated leader.
- Downloads are restricted by both a concurrent-request cap (default 4)
  and an aggregate declared in-flight response-byte budget (default 512 MiB).
  These settings are configurable via `remote_download_limits` when
  constructing a shared coordinator.
- Only **transport timeouts, connection failures and rate limiting** are
  retried, at most 3 attempts by default with bounded exponential backoff.
  Authorization failures, missing resources, redirects, TLS/origin validation,
  invalid response framing, digest mismatch and CAS failures are never
  retried. Cancellation ends queued, waiting and retrying requests.
- The coordinator runs on general worker threads and schedules HTTP jobs on
  IO workers. Verified full immutable objects are ingested into the existing
  persistent CAS before any waiter receives success.
- Eager generation acquisition remains **serial** for its existing
  all-or-nothing verification/activation semantics, but uses the same retry
  boundary. Lazy provider reads use bounded parallel GETs across distinct
  chunks while coalescing duplicate-chunk requests.

The coordinator does not implement global cache eviction, production CDN
credentials, or resumable partial-object SHA verification; those remain
separate tracks. Keeping `remote_object_download_coordinator` scoped to
an explicit cache and job system avoids a process-wide singleton or
cross-project transfer leak.

## C9.2g1 — Persistent remote cache retention and quota pruning

C9 uses the existing `derived_data_cache` filesystem CAS; no second asset
cache, cooked registry or blob identity is introduced. The DDC already
provides a configurable size budget, threshold/target-based LRU-ish
last-access pruning and temporary/action expiry.
The cleanup operation is explicit (`cache.prune()` on a worker); downloading
an object does not synchronously trigger disk scanning or eviction on an IO
worker. The runtime maintenance/promotion policy will schedule cleanup after
activation in C9.2g2. C9.2g1 adds **persistent
scoped SHA-256 retention** so the active and previous verified generation's
object references cannot be evicted when the process exits.

- `retain_remote_content_generations(cache, active, last_good)` forms the
  union of the two C9.1 immutable cook-manifest and package-chunk closures.
  It rejects malformed SHA/path/size references, incompatible project/runtime
  base/target/profile identity, and a last-good generation that isn't older.
  Shared object hashes are retained once; retention scopes are stable for a
  project + engine/runtime base build + target/profile, not for each generation.
- `derived_data_cache::set_persistent_pins(scope, hashes, error)` replaces
  a scope in one bounded, versioned `persistent-pins.json` snapshot under
  the DDC root. Updates write a temporary file and atomically replace the
  prior snapshot; Windows uses an overwrite-capable OS rename. The cache
  restores all pin scopes when it opens, **before any prune can execute**.
  Prune excludes the union of volatile and persisted pins, even when pinned
  data alone exceeds the normal quota. Stale temporary downloads remain
  subject to the DDC's configured temporary expiry.
- Malformed, unreadable or oversized retention metadata disables eviction
  rather than risking loss of the last-good generation. Errors are available
  from `persistent_retention_ready()` and failed update results; repair is
  explicit, never a silent reset. Read-only caches cannot mutate snapshots.
  Different project/target scopes cannot unpin one another; closing or
  retiring a project may explicitly call `clear_remote_content_retention`.
- On process restart the DDC restores **which immutable digests to retain**;
  it does **not** interpret pin metadata as a trusted release, claim a lazy
  chunk has already been downloaded, or automatically activate a scene.
  A runtime must still validate the C9.1 release, load and SHA-verify each
  required CAS object, then activate via the existing catalog/update receiver.
  This runtime activation and reconnect policy is the next C9.2g2/C9.3 slice.

Call retention updates and quota pruning off the render/main thread. A single
DDC cache root should have one retention-metadata writer at a time; multi-process
writer coordination is not introduced here. No credentials, authoring paths,
source artifacts or editor-only network code are persisted. The tests cover
restart/offline recovery, active/last-good rotation, independent scopes,
corrupt metadata fail-closed behavior, hash tampering, bounded eviction and
expired temporary-file cleanup.

## C9.2g2 — Runtime-safe remote generation activation and last-good recovery

The `remote_runtime_generation_session` connects verified C9.2 cooked releases
to the ordinary `cooked_asset_catalog` and existing per-artifact
`runtime_update_receiver` without introducing a second VFS or content cache.

**Strict release activation.** Acquire/cache all C9.1 immutable objects with
`acquire_remote_generation(...)`, then on a serialized simulation-safe
boundary call `session.activate_verified(release)`. It requires exact
project GUID, engine version, runtime base-build and target/profile identity,
and a strictly increasing generation. The strict provider factory verifies
every cached chunk and addressed artifact hash before mounting.
The session persists retention for the candidate and previous active release,
mounts the verified provider at higher VFS priority, retains the prior active
as last-good, and retires the older provider. Failed validation, retention
or mount leaves the existing base visible.

**Explicit rollback.** `session.rollback()` unmounts the current remote base
and exposes the previous SHA-verified last-good provider while keeping its
object closure protected from cache pruning. The maximum accepted generation
stays monotonic even after rollback, so stale releases cannot be replayed.
Rollback refuses to proceed if an active live artifact overlay might shadow
the older base, rather than silently mixing incompatible generations.

**Live updates stay a separate verified layer.** Optional
`session.attach_overlay(private_root)` uses the existing
`cas_overlay_provider` and `runtime_update_receiver` at reserved high
priority over the remote cooked base. `session.ingest_overlay(update, blobs)`
delegates artifact hash, target/build checks, ordered sequence validation,
tombstones and atomic overlay manifest commit to that receiver.
C9 remote package chunks are **not** treated as raw overlay blobs.
The cooked Asset Database and VFS publish provider change events downstream.
Actual GPU resource residency refresh and scene reload-versus-restart policy
are C9.6 (#1139).

**Lifecycle and recovery.** Activation, rollback and overlay attach are
safe-frame serialized runtime operations, never network/render callbacks.
Cache prune/rotation must share maintenance ownership, preventing concurrent
quota sweep from deleting an unpinned candidate during promotion.
The session maintains at most active and previous remote bases normally;
failed removal of an older low-priority mount is counted in diagnostics.
Persisted C9.2g1 SHA retention is still only an eviction barrier: on restart,
the runtime revalidates compatible release metadata and CAS bytes before mount.

This slice does not add C9.3 launch overrides, C9.4 editor HTTP service,
automatic scene-ready handshake, CDN signing, automatic persisted-release
reconstruction, live GPU hot reload or remote feed replay.

The session does not change normal Editor Viewport Play, issue a build, or
silently start a content server. External Play orchestration remains owned by
#1140 and can activate this verified provider only after the cook/publish path
and runtime identity checks have completed.

## C9.2h — Transfer diagnostics, integrity and failure acceptance

The C9.2 shared `remote_object_download_coordinator` now exposes
`metrics()`, a **thread-safe cumulative snapshot**. All counters are scoped
to the coordinator instance and intentionally contain no origin URLs, paths,
hashes, authorization tokens or other sensitive content. A runtime or editor
can keep a shared coordinator in its session owner and poll metrics for
structured diagnostics; callers who use the default temporary coordinator can
still access per-operation errors and eager acquisition's existing byte/object
counts. Collection does not require a logger or introduce a second transport.

The snapshot includes:

| Field | Meaning |
|---|---|
| `calls`, `cache_hits`, `coalesced_calls` | Logical demand, verified CAS reuse and followers sharing an in-flight immutable object |
| `network_attempts`, `retry_attempts` | Actual leader attempts and retries, not duplicated follower requests |
| `completed_downloads`, `downloaded_bytes` | Whole SHA-verified immutable transfers committed to the CAS |
| `failed_calls`, `cancelled_calls`, `integrity_failures` | Structured failure totals; no raw HTTP response contents or secrets |
| `elapsed_milliseconds` | Sum of completed logical call durations, including cache, waits and retries |
| `active_requests`, `peak_active_requests`, `queued_requests`, `peak_queued_requests` | Current/peak network concurrency and waiting pressure |
| `active_declared_bytes`, `peak_declared_bytes` | Network in-flight object-size budget use; these are declared limits, not OS resident-memory measurements |

The coordinator updates request counts at the real IO dispatch boundary;
corrupt framing, SHA mismatch, untrusted status, invalid URL, authorization
failure or a cancelled wait cannot masquerade as a successful transfer. Cache
integrity remains verified before every read. A forced refresh does **not**
delete a previously verified CAS entry when a remote object becomes 404.
Failed/partial body reads are discarded without writing a partial blob, so the
next request starts a **clean whole-object download**. Partial HTTP byte
ranges remain a transport capability, but demand-loaded `.arcpak` chunks
are ingested only after the entire object passes SHA-256. Arbitrary partial
byte ranges are never published as verified CAS entries.

### Acceptance and ownership

Existing C9.2a/e1/e2 native HTTP tests cover 206 Content-Range boundaries,
ETag/If-Match, redirect rejection, local real-server Linux HTTP, oversize
and 404/cancellation behavior; C9.2c/d/f/g tests cover CAS persistence,
whole-generation VFS mounts, cache-first/offline/refresh, concurrency,
transient retry, corruption, active/last-good protection and rollback.
C9.2h adds diagnostics assertions and regression gates for no retry of
truncated bodies, clean recovery, stale verified cache retention on 404,
cancelled queue admission and in-flight limits.

This closes the **transport/cache contract and implementation** for C9.2
issue #1135. C9.3 selects the per-platform Development endpoint and writable
cache root; C9.4 hosts the authenticated editor service; C9.5 publishes
ordered changes; C9.6 subscribes/ingests verified update batches and updates
GPU/scene state; C9.9 performs real external-runtime Windows/Android E2E
security/performance acceptance. C9.2 itself cannot claim that those higher
level sessions, physical GPU device tests, or Shipping CDN release channels
are finished.
