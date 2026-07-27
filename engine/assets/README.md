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
