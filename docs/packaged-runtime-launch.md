# Packaged runtime launch contract (D1.1, issue #1094)

A packaged game and a dedicated server must enter ARC through the same startup contract. The platform host owns process entry, native surfaces, input, writable user storage and the distribution locator. ARC owns distribution validation, cooked-content mounting, project registrations, startup worlds and shutdown.

D1.1 is **only the platform-neutral launch contract and deterministic validation**. It does not start a game, discover a distribution root, open a manifest, mount a package, load a scene or change an existing host/template yet. Those are D1.2–D1.8 and must retain the same contract.

## API and authority

`arc/project/runtime_launch.h` adds:

- `runtime_launch_descriptor`, carrying an existing D0 `distribution_manifest` and `runtime_launch_host`. The manifest remains the **single authority** for project GUID, engine/SDK version, role, target profile, startup scene GUID order, runtime configuration path and cooked package manifest references; D1 does not invent a second schema.
- `runtime_distribution_location` with three mutually exclusive source modes: absolute executable path (native host discovers the release root in D1.2), explicit absolute directory root (CI/relocated distribution), or a borrowed `io::virtual_file_system` plus virtual root (Android/package resources). The contract does not call `current_path()` or read an SDK/source tree.
- `runtime_launch_host` with the writable user-storage root, capabilities and optional device profile. Those machine-specific values stay out of the serialized D0 manifest.
- A stable ordered list of `assets::content_bundle_id` startup bundles. IDs use existing C3 identities (`Engine`, `Boot`, `Startup/Common`); the D1.3/C4 stages decide mounting and residency.
- `validate_runtime_launch()` and `prepare_runtime_launch()`, which return structured `project_error` rather than starting an empty game or performing I/O.

Native validation reuses D0 for safe staged paths and supported target/role policy, then checks requested role, required client startup scene, actual host platform, the exclusive distribution locator, host-owned writable storage, and duplicate/empty bundle IDs. A dedicated server can intentionally omit startup scenes, use renderer `none`, and run without a window. Android can supply an already mounted virtual distribution with no native executable or surface at startup.

## Subsequent implementation

- **D1.2:** use the host locator to find/read `arc-distribution.json`, verify required bytes/integrity, engine compatibility and runtime config, then call this validation.
- **D1.3:** mount referenced cooked manifests through `cooked_asset_catalog`, then integrate the C4 bundle mount/activation API (#992). Mount does not imply resident assets.
- **D1.4:** register project runtime/server modules, realize cooked startup scenes in explicit framework worlds and report absent scene/content instead of accepting the framework's implicit default world.
- **D1.5–D1.7:** wrap the validated launch in a running session with shared lifecycle/shutdown, and adapt existing Windows and headless host entries. Do not move native OS/surface code into `engine/project`.
- **D1.8:** relocate packaged client/server fixtures and exercise repeated start/stop, failure cleanup, missing content, cancellation and native host parity. GPU rendering golden tests belong to their own track.

The tests in `engine/project/tests/runtime_launch_tests.cpp` exercise role/target validation, scene/bundle ordering, relative-path rejection, explicit distribution roots, and a VFS-backed Android contract without requiring Android NDK on the native test runner.
