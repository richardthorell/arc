# Live Play L1.3b2c1 — Private target-cooked scene staging

Follows [#1211](https://github.com/richardthorell/arc/pull/1211),
[#1218](https://github.com/richardthorell/arc/pull/1218) and issue
[#1198](https://github.com/richardthorell/arc/issues/1198).

## Editor-main process preflight

The native `scene.autosave` response now includes `sourceScenePath`, captured
in the same native command as its scene GUID and history revision. This is the
**existing saved scene path**, not the private snapshot output path. An unnamed
new scene with no source path cannot yet be used for external native Play; the
editor should prompt users to save the new scene once (unsaved *changes* to a
saved scene remain supported).

`prepareLivePlayCookedStaging` binds that path to the project's registered
content roots and `.arcmeta` sidecar. The registry **asset GUID** is distinct
from the document's `scene.id`: the asset GUID selects the cook root while the
saved document's scene ID must match the native captured Authoring World.

The editor-main service verifies the snapshot's SHA-256 against L1.3a's sealed
bytes, runs `arc-cook cook` with the trusted Development-only ephemeral
override from #1218, validates each manifest's target/profile/scene asset
identity, then calls the existing native `verify`, `package` and `verify`
commands. All cook/package output stays under the private Live Play session
directory. It never writes project scene source files or replaces the normal
project's `current-workflow.json` / C9 current release.

The service has explicit cancellation, project/session-is-current guards,
bounded tool lifetimes, single completion-record checks, artifact path
containment, duplicate family protection, and failed-staging cleanup.

## Scope boundary

This is the **private local cooked package preparation service**. It is not
automatically invoked on Play until the C9 #1137 editor-hosted server can
publish a verified target-cooked generation; doing so now would add latency
without loading anything into the native game. It does not serve raw scene
bytes, publish a C9 release descriptor, override a native content endpoint,
or report successful scene activation.

The following L1.3b2c2 slice should use this module after L1.3a capture,
publish immutable chunks and manifest via C9's authenticated editor server,
offer the verified generation through the L1.3b1 control plane (#1215),
and await genuine native scene activation (#1217). Keep #1198 open until a
dirty cube transform appears in external Windows Play.
