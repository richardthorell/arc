# Runtime virtual filesystem and live content

ARC runtime content is addressed through a provider-based, read-only virtual filesystem. Asset GUIDs remain the
authoritative identity stored by scenes and gameplay systems; a VFS URI is only the current storage location for a
cooked artifact.

## Addressing and mounts

Virtual paths use normalized URI syntax. Scheme and authority are ASCII case-insensitive and normalized to lower
case, while path components are case-sensitive. Backslashes and parent traversal are rejected. Current conventions
are:

- `package://base/...` for files supplied by the installed build;
- `artifact://game/<asset-guid>/<schema-id>/<encoded-name>` for independently streamable cooked artifacts.

Providers are mounted at explicit priorities. A higher-priority provider wins. A `not_found` result continues into
the next provider, while a tombstone stops lookup. Provider failures and corrupt content also stop lookup so damaged
or rejected live content cannot silently reveal stale packaged bytes.

`resolved_virtual_file` is the unit passed to streaming systems. It captures the provider, mount, content generation,
and size once. Range reads do not reparse the URI or lock the mount table. A provider reports a stale handle if its
captured generation has been superseded. Renderer resource generations provide the corresponding publication guard,
so an old completion cannot populate a replacement texture or virtual-geometry resource.

Native paths remain valid at authoring boundaries: source scans, metadata sidecars, cooker output, and deliberately
external user files such as profile photos. Runtime content should resolve a logical artifact once and retain its VFS
handle.

## Included providers

- `filesystem_file_provider` exposes one contained root with asynchronous range reads, enumeration, and portable
  debounced polling. It never returns a native path through the VFS API.
- `package_artifact_provider` indexes `.arccookmanifest` records and maps artifact identities to validated package
  ranges. `.arcpak` filenames and offsets stay inside the provider.
- `cas_overlay_provider` exposes a persistent live/OTA manifest backed by immutable SHA-256 blobs. It mounts above
  packaged artifacts and supports tombstones.
- `memory_file_provider` supplies deterministic tests and small ephemeral fixtures. It is not the production live
  update store.

The cooked asset catalog translates `(asset GUID, schema, name)` into VFS handles and translates committed VFS
changes back into typed artifact events. A catalog reset is emitted when its bounded event history overflows or a
mount changes.

## Verified update ingestion

`runtime_update_receiver` is transport-neutral. A transport supplies a target-cooked manifest, then calls `begin`,
`begin_blob`, one or more sequential `stage_blob` calls, `finish_blob`, `finish_verify`, and `commit`. `ingest` is a
whole-buffer convenience for tests and local tools.

Blobs are staged below the platform-supplied writable cache root and verified by declared size and SHA-256. The
receiver rejects unsupported formats, the wrong target profile or base build, stale sequence numbers, missing blobs,
truncation, and hash mismatches. Only after every blob verifies does it atomically replace `overlay-manifest.json` and
activate one new provider generation. A failure leaves the previous manifest and runtime generation active and emits
no artifact changes. Staging debris from an interrupted session is discarded when a receiver starts; committed blobs
remain reusable by hash across updates and restarts.

The receiver performs filesystem work synchronously from the caller's perspective and is intended to be driven from
an IO/transport worker, never the render thread. It contains no sockets, authentication, discovery, or device-control
policy.

## Future transports and platforms

The connection layer ends at `runtime_update_receiver`:

- a desktop editor bridge can stream the manifest and blob chunks over an authenticated socket;
- Android tooling can use ADB port forwarding for the same protocol;
- an OTA client can download batches and drive the same receiver;
- an APK asset provider can implement `virtual_file_provider` and mount at `package://base` without changing catalog
  or renderer code.

Play/pause, scene synchronization, bandwidth scheduling, device discovery, and remote authentication are separate
follow-up systems. They should consume typed artifact events after commit rather than bypassing the receiver or
writing generic VFS files.

## Polling and telemetry

Provider changes are collected only by `virtual_file_system::poll_changes`; callbacks run synchronously at that
caller-controlled point. The bounded journal supports `events_since` and reports overflow/rescan. VFS telemetry
tracks mounts, resolutions/fallthroughs, read operations/bytes, and event overflow. Update telemetry tracks accepted
and rejected batches, staged/reused blobs and bytes, active sequence, and active artifact count. Texture and
virtual-geometry streaming telemetry reports stale completions at the renderer publication boundary.
