# ARC game distribution manifest (D0 / v1)

The distribution manifest describes a staged, runnable game or server. It is not an ARC SDK installation manifest, an arc-cook manifest, or a second asset catalog. The public contract and parser/serializer live in arc/project/distribution_manifest.h, and the conventional output filename is arc-distribution.json.

## Boundaries

- .arcproject owns the project GUID, engine version, startup scenes, target and cook profile.
- arc-cook owns asset dependencies, bundles, variants, cook manifests and content integrity. The distribution records only the relative paths to the staged .arccookmanifest files.
- The distribution manifest owns application role, entrypoint, runtime configuration, finalizer identity and a complete staged file inventory, including executable/native libraries, configuration, content manifests and package chunks.
- The common assembler (later #1095) may resolve absolute host paths **in memory**, but must serialize only distribution-root-relative paths. Platform finalizers handle native directory layouts, archives, metadata, signing and deployment.
- A platform host supplies the logical distribution root to shared runtime startup (#1094). Desktop can locate the root near the executable; Android may provide an APK asset provider rather than a filesystem directory.

## Example (Windows client)

The hashes below are illustrative, not real file digests.

```json
{
  "format": "arc.distribution",
  "formatVersion": 1,
  "projectGuid": "11111111-1111-1111-1111-111111111111",
  "engineVersion": "0.1.0",
  "sdkVersion": "0.1.0",
  "target": {
    "id": "windows-x64-vulkan",
    "platform": "windows",
    "architecture": "x86_64",
    "renderer": "vulkan",
    "configuration": "Shipping",
    "cookProfile": "windows-x64-vulkan"
  },
  "role": "client",
  "finalizer": "windows-directory",
  "entrypoint": { "kind": "executable", "value": "bin/Game.exe" },
  "startupScenes": ["22222222-2222-2222-2222-222222222222"],
  "runtimeConfig": "config/runtime.json",
  "content": {
    "cookManifests": ["content/windows-x64-vulkan.arccookmanifest"]
  },
  "files": [
    { "path": "bin/Game.exe", "kind": "executable", "size": 100, "sha256": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa" },
    { "path": "config/runtime.json", "kind": "configuration", "size": 20, "sha256": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa" },
    { "path": "content/windows-x64-vulkan.arccookmanifest", "kind": "content", "size": 50, "sha256": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa" }
  ]
}
```

Android uses the **same schema** with platform android, architecture arm64, renderer vulkan, an Android finalizer and an android-activity entrypoint (qualified component name rather than a filesystem path). Dedicated servers use role server, renderer none, and a native executable; startupScenes may be empty.

## Stability and validation

Only formatVersion 1 is supported. A future version must define an explicit migration rather than silently interpreting unknown fields as an older version.

All staged paths must be package-root-relative, use forward slashes, and exclude traversal, absolute paths, drives, backslashes, control characters and noncanonical segments. Duplicate destinations are rejected case-insensitively. Required referenced files must appear in the inventory. Every file has a size and a lowercase SHA-256 digest.

Serialization sorts file records and cooked-manifest references, preserving startup scene order. The manifest never includes machine build paths, checkout/SDK roots, timestamps or secrets. Signing that changes bytes must precede final file hashing or be detached. This contract does not itself perform hash verification against the filesystem.

D0 PR 1 provides types, serialization and validation only. PR 2 introduces the planner/finalizer interface. Shared runtime startup belongs to #1094, native staging to #1095, relocation CI to #1096 and content variant/bundle semantics to #994/#995.
