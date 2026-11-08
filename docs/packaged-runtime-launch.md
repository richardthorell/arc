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

## Cooked catalog bootstrap (D1.3)

`arc/project/runtime_catalog.h` adds `bootstrap_runtime_catalog(runtime_catalog_request)`, consuming the already-validated `runtime_boot_context` from D1.2. The resulting `runtime_catalog_session` owns any mounts created during bootstrap and exposes the existing `assets::cooked_asset_catalog`, ordered startup scene GUIDs and requested bundle IDs.

- **Native staged content:** a host-owned `io::async_file_service` and `io::virtual_file_system` mount the D1.2 canonical distribution root under `package://runtime` (configurable). The source remains executable/CWD/SDK-independent.
- **Already-mounted content:** an Android-style host supplies its existing VFS and virtual distribution root. The session does **not** take ownership or unmount the external source. The native and mounted paths use the same cook manifest validation, catalog and scene lookup.
- **Cook manifests:** parse every D0 `content.cookManifests` reference and reject incompatible target platform, architecture, renderer and development/shipping configuration. The current `cook_target` supports Windows, Linux and macOS; Android/iOS cooked variants must wait for C6 (#994). A Windows cook is never silently treated as Android data. D0 cook-profile IDs are not conflated with cook-target display names; full C6 variant/quality selection remains #994.
- **Content Bundle roots:** require each explicitly requested D1.1 `startup_bundles` ID (e.g. `Boot`, `Startup/Common`) to be advertised by at least one staged cook manifest. This is a mount/availability check only; no bundle is pinned, prefetched or made resident.
- **Scenes:** resolve D0 ordered startup scene GUIDs through `catalog.database()` and require a cooked `scene` representation. Dedicated servers may have no startup scene. Actual deserialization, world creation and project registration are D1.4.
- **Failure and teardown:** package providers are mounted in deterministic manifest order. Any invalid manifest, missing chunk, missing bundle or scene destroys the incomplete catalog and its mounts; the native distribution provider is then unmounted. A successful session follows the same order at shutdown. Externally supplied VFS mounts remain untouched.

Mounting reads cook metadata and locates chunk ranges only. `asset_residency::metadata_only` is preserved until a downstream loader explicitly requests artifact bytes. **C4 (#992) remains responsible for asynchronous bundle load, prefetch, pin, unload and budget policy**; D1.3 deliberately does not duplicate that lifecycle.

`engine/project/tests/runtime_catalog_tests.cpp` exercises client/server startup metadata, scene type checking, missing bundles/chunks, target mismatch, rolled-back native mounts, an externally owned mounted VFS provider, and metadata-only artifact resolution.

## Packaged project registration and startup worlds (D1.4)

`arc/project/runtime_application.h` exposes `prepare_packaged_application`, the shared
adapter from D1.2's validated `runtime_boot_context` and D1.3's mounted
`runtime_catalog_session` to `framework::application`.

- **Preflight:** for each D0-ordered startup scene GUID, find the active cooked
  scene-schema representation via the catalog's logical database and read its
  artifact from the existing VFS. Verify its actual byte length and ARC SHA-256
  against the cooked package manifest, with a 128 MiB per-scene and 256 MiB
  aggregate startup budget. Missing, incompatible or corrupt startup scenes
  fail before `framework::runtime` is constructed. Only requested startup
  scenes are read; mounting the catalog still does not load unrelated assets.
- **Game registrations:** the platform/game integration supplies in-process
  `packaged_game_registrations` callbacks for runtime services, modules and
  scene materialization. This delegates to the existing
  `framework::application::register_services/register_modules/register_worlds`
  lifecycle; there is no editor process, source-path inspection, implicit plugin
  guessing or second module registry. The app does not load native libraries
  itself. D1.6/D1.7 must connect the host's validated game-module loader and
  project-specific scene materializer to these hooks.
- **Explicit worlds:** one `framework::runtime_world` per ordered startup
  scene; no placeholder systems and no implicit default scene. The materializer
  receives verified cooked bytes and the specific world to populate. If any
  materializer returns failure, previously created worlds are rolled back,
  and a `packaged_runtime_startup_error` retains the structured
  `project_error`. The startup stage does not attempt to interpret arbitrary
  game-specific component types using editor-only scene codecs.
- **Server:** server role enforces a headless simulation (no window or
  presentation), and explicitly creates a single server world if the manifest
  intentionally declares no startup scenes. A server-world initialization
  callback is optional; a server with startup scenes uses the same checked
  materialization path as clients.
- **Ownership:** the returned packaged application owns its bounded startup
  scene buffers and registration callbacks. The host retains ownership of the
  cooked catalog/VFS, and must keep both alive through runtime shutdown.

`engine/project/tests/runtime_application_tests.cpp` covers real framework
startup/shutdown, ordered client worlds and module callbacks, scene-less
headless server startup, corrupt packaged scene bytes, missing materializer,
GUID ordering mismatch, and rollback on client/server world setup failures.

## Shared packaged runtime session and teardown (D1.5)

`arc/project/runtime_session.h` adds `start_packaged_runtime(packaged_runtime_request)`, the
single shared startup path for client and dedicated server. It calls D1.2 manifest discovery
and integrity validation, D1.3 cooked-catalog mounting, D1.4 project registration and
verified scene preflight, and then `framework::runtime::start()`. All steps run on the
host's runtime thread. Startup either returns a fully running
`packaged_runtime_session` or a structured `project_error`; no partial session
escapes to host code. Project module and scene implementations remain host-provided
callbacks and do not import editor code.

- **Ownership:** the session owns its boot context, cooked catalog mounts,
  packaged application with scene buffers and the framework runtime. The host
  still owns its process, VFS, async file service and any pre-mounted package
  providers. The host keeps those services alive through session destruction.
- **Teardown:** stop framework worlds, project modules and services first;
  destroy the framework runtime; release the application and scene buffers;
  unmount cooked artifact providers; finally release only the native
  distribution mount that this session created. Shutdown and the destructor
  are idempotent; a shutdown callback exception is reported without skipping
  lower-level catalog cleanup. A failed startup rolls back the same owned state.
- **Host loop:** the host supplies measured deltas to `advance(delta_seconds)`
  and forwards platform-neutral `framework::event` inputs to `dispatch`.
  `request_stop()` stops the loop without releasing callback-owned resources
  during that callback; hosts subsequently call `shutdown()`.
  Frame/world exceptions and faulted simulation worlds cause structured errors
  and clean teardown.
- **Mobile/desktop lifecycle:** `pause()` and `resume()` suspend updates
  independently of debug time controls. Clients also call `surface_lost()`
  and `surface_ready(width, height)`, which convey focus/resize via the
  existing framework event contract and keep the simulation suspended until
  *both* host pause and missing surface conditions are cleared. ARC does
  not allocate, own or recreate the OS window/swapchain in this layer.
  Headless servers reject surface callbacks, but support host-level pause.
- **Limits:** this is a synchronous lifecycle contract, not background work,
  a new renderer, a native module loader or a replacement for C4 (#992)
  bundle load/prefetch/pin/unload ownership. D1.6/D1.7 adapt the concrete
  server/Windows host entrypoints. D1.8 expands relocated package and
  cancellation coverage.

`engine/project/tests/runtime_session_tests.cpp` covers repeated
headless startup/shutdown with a shared VFS, client pause/surface transitions,
close requests, corrupt/missing content, callback failures, and shutdown
exceptions with no leaked mounts.

## Subsequent implementation

- **D1.6–D1.7:** adapt existing dedicated-server and Windows native host entries to the shared runtime session and bind packaged native game-module/scene materialization callbacks. Do not move native OS/surface code into `engine/project`.
- **D1.8:** relocate packaged client/server fixtures and exercise repeated start/stop, failure cleanup, missing content, cancellation and native host parity. GPU rendering golden tests belong to their own track.

The tests in `engine/project/tests/runtime_launch_tests.cpp` exercise role/target validation, scene/bundle ordering, relative-path rejection, explicit distribution roots, and a VFS-backed Android contract without requiring Android NDK on the native test runner.
