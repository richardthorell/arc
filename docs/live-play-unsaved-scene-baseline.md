# Live Play L1.3a — Unsaved Authoring World capture

Tracking: [#1198](https://github.com/richardthorell/arc/issues/1198),
[Live Play epic #1194](https://github.com/richardthorell/arc/issues/1194),
and [L1.2c #1209](https://github.com/richardthorell/arc/pull/1209).

## Contract

Windows standalone Development Play now captures an **immutable, session-owned
authoring scene baseline** after the project-published Development package has
been validated and before launching its native process. It is a staging-only
L1.3a milestone: the running game does **not yet consume** this baseline.

The editor main process issues the existing native `scene.autosave` command to a
private temporary location. That command copies the current Authoring World
and reuses `save_scene_document`, including stable entity GUIDs, authored
components, references, dependency records, and native document integrity.
It does not mark the document saved, change its active path, or change its
undo/history revision. The capture can therefore include unsaved transforms,
lights and other supported serializable scene components.

The editor validates format/version/scene GUID, records native scene/history
revisions, computes a SHA-256 of the serialized bytes and stores a small
identity manifest with project/build/profile/session/runtime-instance,
scene GUID, entity count and digest. The `.arcscene` and manifest remain
under a private `arc-live-play-baseline-*` temporary directory outside the
project. The scene file is made read-only after capture; no renderer IPC or
command-line argument exposes its path. Capture is bounded to 16 MiB.

Only one scene baseline may exist per active Windows Play session. Stop,
failed native process, project switch/close, shutdown and launch failures
discard it. Cancellation during an in-flight native capture rejects stale
output once the host command finishes. A different session cannot clear the
active snapshot.

## Deliberate boundaries

- No `Save` or implicit write to the authored `.arcscene` file.
- No C9 cooked catalog, content server, asset upload, or runtime-side scene
  materialization yet. The staged `.arcscene` is **authored input**, not a
  verified target-cooked C9 generation.
- No live entity update commands, runtime hierarchy, Inspector or Android
  transport. Default Editor Viewport Play remains unchanged.
- Invalid terrain/other unsupported native scene components fail explicitly
  through the normal scene serializer; no partial baseline is published.
- An existing Development package is still required by #1209.

## Follow-up

L1.3b will convert the session-authored snapshot to target-compatible
cooked/validated scene content through C9 (#1134–#1137 and #995), associate
the resulting generation with the attached runtime and confirm that unsaved
scene values appear at startup. Subsequent modifications use L1.4 (#1199);
explicitly saved asset edits use C9.5/C9.6 (#1138/#1139).
