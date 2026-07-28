# ARC Assets

`arc-assets` owns durable asset identity and runtime asset state. Public consumers include
`<arc/assets/assets.h>` and use `asset_reference` in persisted data. References are
GUID-authoritative; their normalized project-relative path hint exists for diagnostics and
legacy repair, not identity.

Recognized source files receive checked-in `<source>.arcmeta` sidecars. The sidecar stores the
asset GUID, stable type/importer IDs, canonical import settings, and persistent subasset keys.
Removed subassets remain tombstoned so a later reappearance can reclaim the same GUID.

`.arc/cache/assets.db` is a rebuildable private SQLite cache. Derived artifacts are addressed by
SHA-256 under `.arc/cache/derived/<profile>/<prefix>/<hash>/`. Neither location belongs in source
control. A corrupt or incompatible registry is preserved with a `.corrupt-*` suffix and rebuilt
from sidecars.

Importers are registered by their owning module. They return a validated payload, direct
dependencies, subassets, diagnostics, and derived artifacts. ARC validates dependency cycles and
artifact hashes before atomically publishing a generation. A failed hot reload leaves the
previous payload live.

Runtime code requests typed asynchronous loads. Concurrent requests share the import generation;
strong handles follow atomic generation publication, while pins prevent pressure eviction.
Source monitoring currently uses portable debounced metadata polling and can be replaced by a
platform watcher without changing the public manager contract.

## Derived data and cooking

`derived_data_cache` stores immutable SHA-256 blobs under
`.arc/cache/cas/sha256/<prefix>/<hash>` and maps deterministic build keys to artifact lists under
`.arc/cache/actions`. Build keys include source and dependency hashes, importer and processor
versions, artifact schemas, canonical settings, the complete target profile, and compiler
fingerprints. Cache reads always verify content before publication to a consumer.

`arc-cook` is a renderer-free command-line entry point for dependency-closed incremental cooks:

```text
arc-cook cook --project <root> --root assets/materials/default_phong.arcmat
arc-cook package --project <root>
arc-cook verify --project <root>
arc-cook cache stats --project <root>
arc-cook cache verify --project <root>
arc-cook cache prune --project <root>
```

The checked-in `arc.cook.json` defines default roots and Windows/Linux Vulkan profiles. Package
generation emits a versioned `.arccookmanifest` plus content-named `boot` and `startup` chunks.
Mounted packages resolve artifacts by asset GUID and schema and verify every ranged payload.

### Shared cache contract

The engine-facing `shared_cache_backend` is transport-neutral. A conforming HTTP service exposes:

- `HEAD`, `GET`, and immutable `PUT /v1/blobs/sha256/{hash}`
- `GET` and immutable `PUT /v1/actions/{buildKey}`

Clients send a bearer token, accept ranged blob reads, require an ETag matching the SHA-256 name,
and verify the declared length and payload hash locally. A mismatch is a corrupt miss, never a
usable hit. Pull-request clients must be read-only; only trusted main/release automation may
publish action records. `filesystem_shared_cache` is the deterministic team-drive and test
implementation of the same semantics.
