# Packaged runtime launch contract (D1.1, issue #1094)

A packaged game and a dedicated server must enter ARC through the same startup contract. The platform host owns process entry, native surfaces, input, writable user storage and the distribution locator. ARC owns distribution validation, cooked-content mounting, project registrations, startup worlds and shutdown.

D1.1 established the **platform-neutral launch contract and deterministic validation**. D1.2 adds manifest discovery, identity checks and staged-file integrity verification. Neither slice starts a game, mounts content, loads a scene or changes an existing native host/template; those integrations remain in D1.3–D1.8.

## API and authority

`arc/project/runtime_launch.h` adds:

- `runtime_launch_descriptor`, carrying an existing D0 `distribution_manifest` and `runtime_launch_host`. The manifest remains the **single authority** for project GUID, engine/SDK version, role, target profile, startup scene GUID order, runtime configuration path and cooked package manifest references; D1 does not invent a second schema.
- `runtime_distribution_location` with three mutually exclusive source modes: absolute executable path (native host discovers the release root in D1.2), explicit absolute directory root (CI/relocated distribution), or a borrowed `io::virtual_file_system` plus virtual root (Android/package resources). The contract does not call `current_path()` or read an SDK/source tree.
- `runtime_launch_host` with the writable user-storage root, capabilities and optional device profile. Those machine-specific values stay out of the serialized D0 manifest.
- A stable ordered list of `assets::content_bundle_id` startup bundles. IDs use existing C3 identities (`Engine`, `Boot`, `Startup/Common`); the D1.3/C4 stages decide mounting and residency.
- `validate_runtime_launch()` and `prepare_runtime_launch()`, which return structured `project_error` rather than starting an empty game or performing I/O.

Native validation reuses D0 for safe staged paths and supported target/role policy, then checks requested role, required client startup scene, actual host platform, the exclusive distribution locator, host-owned writable storage, and duplicate/empty bundle IDs. A dedicated server can intentionally omit startup scenes, use renderer `none`, and run without a window. Android can supply an already mounted virtual distribution with no native executable or surface at startup.

## Distribution discovery and boot validation (D1.2)

`arc/project/runtime_boot.h` adds `discover_runtime_boot(runtime_boot_request)`. The platform host supplies a native executable, a native distribution directory, or a pre-mounted virtual root, plus the **compiled-in** engine/SDK version, architecture, renderer and build configuration. Optional expected project GUID, target ID and cook profile protect project-specific binaries from the wrong distribution. No version/profile values are inferred from the manifest itself.

- **Executable:** search the executable's parent and up to eight ancestor directories for `arc-distribution.json`. The first found manifest must parse and declare the exact canonical executable as its entrypoint; an invalid nearby manifest is not silently bypassed. Never inspect the working directory or an installed SDK. The resolved root is canonical and is returned separately from the original host executable path.
- **Directory:** use the supplied absolute directory as the only root, allowing relocation integration tests without any source paths.
- **Mounted VFS:** read `arc-distribution.json` and staged inputs under the existing `io::virtual_path` root. This works with an Android-like package provider and never requires an Android native executable or window surface.
- **Fail closed:** D0 owns JSON format/version, role, staged path safety and inventory linkage; D1.1 owns host/role validation. D1.2 rejects incompatible engine/SDK/architecture/renderer/configuration and bound project/target identities. Missing and unreadable bytes, escaped native symlinks, invalid entrypoint identity and bad SHA-256 produce structured `project_error` with an actionable field and path or virtual name. Manifest reads are capped at 16 MiB.
- **Startup integrity (default):** verify all staged native inventory entries exist, remain beneath the root and match their declared sizes; SHA-256-check the runtime configuration and cooked manifest files. For mounted VFS providers, verify only configuration and cooked manifests: native libraries may be loaded by the platform's own package loader, not the virtual content provider. This avoids hashing potentially large content payloads on every launch.
- **Full integrity (explicit):** `runtime_boot_integrity::full` checks the SHA-256 of every inventoried native file; for a mounted provider, all inventory bytes must be exposed by the VFS. Use for test/release auditing, rather than every normal game startup. ARC's existing `assets::hash_file`/`hash_bytes` SHA-256 implementation is reused.

Discovery and verification are **read-only**. D1.2 does not verify the runtime-config schema, inspect asset dependencies, mount C4 bundles, initialize engine services, create a window or start a gameplay world. Those belong to D1.3–D1.7.

`engine/project/tests/runtime_boot_tests.cpp` covers relocations, exe-identity checks, no-scene servers, malformed/unsupported manifests, incompatible host metadata, missing/corrupt content, native symlink escape, startup/full verification and mounted nested Android assets.

## Subsequent implementation

- **D1.3:** mount referenced cooked manifests through `cooked_asset_catalog`, then integrate the C4 bundle mount/activation API (#992). Mount does not imply resident assets.
- **D1.4:** register project runtime/server modules, realize cooked startup scenes in explicit framework worlds and report absent scene/content instead of accepting the framework's implicit default world.
- **D1.5–D1.7:** wrap the validated launch in a running session with shared lifecycle/shutdown, and adapt existing Windows and headless host entries. Do not move native OS/surface code into `engine/project`.
- **D1.8:** relocate packaged client/server fixtures and exercise repeated start/stop, failure cleanup, missing content, cancellation and native host parity. GPU rendering golden tests belong to their own track.

The tests in `engine/project/tests/runtime_launch_tests.cpp` exercise role/target validation, scene/bundle ordering, relative-path rejection, explicit distribution roots, and a VFS-backed Android contract without requiring Android NDK on the native test runner.
