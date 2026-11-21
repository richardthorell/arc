# Live Play L1.3b2b — Development ephemeral scene cooker input

Tracking: [#1198](https://github.com/richardthorell/arc/issues/1198), [#1194](https://github.com/richardthorell/arc/issues/1194), [#1133](https://github.com/richardthorell/arc/issues/1133).

## What this PR adds

L1.3a (#1211) captures the unsaved Authoring World privately, but a runtime
cannot consume raw `.arcscene` files. This PR introduces a **Development-only**
ephemeral source override inside the existing native asset cooker. A captured
scene's bytes replace **only** the source of its existing registered scene asset GUID
during that cook transaction. The normal processors, dependency graph, target
profile, DDC/cache keys, `.arccookmanifest` and `.arcpak` packaging remain
authoritative; there is no second scene database or binary serializer.

- `cook_request.ephemeral_scene` accepts a registered scene **asset GUID**
  from the `.arcmeta` registry and bounded (16 MiB) immutable authoring bytes.
  The scene document's distinct `scene.id` must match the saved scene document.
- The source override's SHA-256 is fed into the existing cook action key.
  Cached results for saved versions cannot be reused for modified unsaved
  bytes. A second identical override may reuse its verified DDC action.
- Dependency closure comes from the registered graph. The normal native
  reflected scene processor rejects missing **required** references;
  an ephemeral source also rejects unseen *optional* references rather
  than publishing incomplete dependencies.
- Shipping cooks reject any override; unknown/unrelated asset identities,
  missing roots, empty/oversized bytes, and non-scene asset types fail closed.
- The local `arc-cook cook` CLI can be invoked by trusted editor main-process
  orchestration using `--ephemeral-scene <private absolute path>`
  and `--ephemeral-asset-guid <registered scene asset GUID>`. It accepts only
  Development profiles and an output directory **outside** the project.
  It does not rename, autosave, rewrite, or commit the original scene.
- No runtime argv contains source paths; this CLI invocation is editor-side
  staging only. The registered **asset GUID** selects the cook root and the
  captured `scene.id` must match that asset's saved scene document.
  Tests cover provenance, cache isolation and Shipping refusal.

## Boundaries and next integration

This PR implements the **actual normal-cooker input** for an unsaved scene.
It does not publish a C9 remote release descriptor, content-server objects,
or activate a world. The C9 track #1135/#1137/#995 owns verification,
immutable artifact publication and transport; L1.3b2c will call this
Development-only cooker using the session-owned snapshot, validate a complete
cooked generation, and hand it to the content server. The runtime must
only ACK `active` after verified C9 activation (#1217). Editor Viewport Play
and Shipping builds are unchanged.
