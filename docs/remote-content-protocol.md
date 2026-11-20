# C9.1 — Remote cooked-content release contract (v1)

**Scope:** metadata and transport-independent validation only. This contract is
consumed by the upcoming C9.2 HTTP client and C9.4 editor-owned server. It does
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
`tools/remote_content_contract.py` for the initial metadata validator.

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

## Range requests and trust boundaries

Immutable objects support `GET`, `HEAD` and optional `Range: bytes=start-end`;
responses must be checked for exact range boundaries, total length, content
length, SHA-256 of complete objects, strong ETag consistency and maximum byte
budgets. A `200` response to a range request must be explicitly handled, never
misinterpreted as a partial response. Signed-URL handoff is allowed only from a
trusted authenticated origin to an allowlisted HTTPS host with bounded expiry;
no access token is persisted in the descriptor or logged. C9.2 owns these
network checks; this first PR validates *metadata*, not HTTP responses.

## Follow-up implementation sequence

1. **C9.1b** — native C++ descriptor parser/validator and cross-language
   fixtures that the runtime client uses (if needed by C9.2).
2. **C9.2** (#1135) — network transport, cache, VFS integration and strict HTTP
   trust behavior.
3. **C9.4** (#1137) — editor-hosted read-only publication service; optional
   authenticated feed.
4. **C9.5/C9.6** (#1138/#1139) — incremental publish and runtime activation.

The authenticated Live Play control/inspection plane (#1194) remains separate
from this cooked-data delivery protocol.
