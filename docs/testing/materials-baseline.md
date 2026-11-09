# Materials Baseline (#943)

Open the repository-root `MaterialsBaseline.arcproject` from the ARC editor, then open its default scene, **Materials Baseline**. The scene is repository/engine-owned, not derived from the Blank 3D template.

### Stable baseline v1

- Fixed scene/entity/material GUIDs, authored camera pose, manual exposure, directional light and solid-color environment.
- Fixed sky time with clouds, fog, stars, moon, animation and environment playback disabled.
- Known-UV engine primitives, grouped into core material families, Standard Lit response, and composition/texture/emission.
- Material Instances share the production Standard Lit parent; a Checker-specialized instance and the reusable Floor instance exercise the shared function workflow.
- Native scene asset bindings remain authoritative; no renderer-only stand-ins or texture substitutions.
- The **Unlit / Textured** specimen uses the checked-in Aerial Grass Rock 1K color texture, with asset metadata; missing image files must never silently satisfy the baseline.

### Validation

```sh
python tools/tests/materials_baseline.py --root .
cmake --preset default
cmake --build --preset default --parallel
ctest --preset default -R arc-materials-baseline-contract
```

The first test checks scene inputs, stable material references, GUIDs, metadata, and dependency manifests, with diagnostic names for failed capability entries.

### Native cook/package regression (#1086)

```sh
cmake --build --preset default --target arc-cook arc-materials-baseline-package-verify
ctest --preset default -R '^arc-materials-baseline-cook

### GPU captures and remaining scope

This first milestone establishes the canonical scene and structural contract. **It now includes native cooked package-load validation, but does not yet implement deterministic GPU captures, image tolerances or versioned golden images.** Those require a device-backed test runner, explicit capture of active render/material path and device profile, and reviewed reference PNG updates. Never auto-update goldens from a normal test run; missing shader/texture, wrong permutation, unplanned fallback or unavailable required path must fail clearly.

Expand this baseline rather than creating unrelated fixture scenes: normal/AO maps, anisotropy, masking/two-sided, mip/HDR and material function variants as executable. Follow #359 for authoritative compiler/runtime cleanup, #858 for the Blank 3D starter migration, and #478 for visibility-buffer coverage.

### Clear-coat switching acceptance (#1050)

The native test `compiled coat variants survive typed clearing reset and scene history`
uses real compiled programs, the entity-local editing service, renderer publication,
and scene undo/redo. It checks inherited active routing, explicit clearing to neutral,
reset to inherited routing, entity isolation, and weight edits without program changes.
It requires Slang and skips explicitly when unavailable. It does **not** load the
fixture's texture on a GPU or claim image correctness.

`registered coat texture overrides preserve linear semantics and isolate color cache
entries` covers real-file GUID assignment using a generated TGA and the native asset
registry. It verifies normal-versus-color upload semantics, separate cache handles
for the same file, Forward/Deferred switching on assign/clear/reset, and rehydration
of persisted override records. It uses synthetic program metadata (no shader compiler
required) and inspects renderer upload events, not GPU images or a full scene-file
save/reopen. Entity-local loaders must use the selected program's texture-binding
semantic, including function-specialized bindings; normal maps must not reuse a
generic-color cached upload.

`coat texture assignments clear and inheritance survive scene save and fresh reopen`
adds the scene-file persistence gate. It creates registered `.arcmat`, `.arcmatinst`,
and generated TGA sources, edits one of two shared-material entities, and saves
separate assigned/cleared/reset scenes. Each file is reopened with a fresh asset
registry, scene, renderer, and texture cache. Checks cover persistent material and
entity GUIDs, exact local override records, inherited versus explicit-empty routing,
independent entity state, valid active texture bindings, and linear normal uploads.
Runtime handle numbers and program generations are deliberately not compared across
reloads. Slang is required; an unavailable compiler is an explicit skip, not a pass.
The fixture deliberately uses `Content`, not the legacy `assets` directory: scene
reload must pass the asset registry's configured root to material realization so
inherited instance textures do not disappear or silently change routing.
This exercises real compilation and renderer publication but still does not render
GPU images. Device-backed switching/RNM captures and performance acceptance remain
outstanding.

Run the remaining device-backed check in this baseline with the authenticated editor
gateway. Discover current operation schemas; `materials.inspectEntity` requires its
v2 operation contract and current catalog revision. Obtain edit approval before
changing the scene, use GUIDs, and carry returned scene/world revisions through edits.

1. Use two visible entities sharing Standard Lit (including an authored Material
   Instance). Inspect parameter metadata to identify the exposed coat-normal texture
   and coat weight IDs; do not infer IDs from names or embed IDs from another graph.
2. Frame both objects with fixed camera, light, exposure, and device profile. Capture
   color, depth, objectId, and normals. Require ObjectID coverage for both target GUIDs
   before claiming visual verification.
3. Start with no independent coat-normal binding: require Deferred readback. Assign
   a registered normal-semantic texture to only one entity: require Clustered Forward
   readback and a changed program generation, leaving the second entity unchanged.
4. Explicitly clear the texture: require Deferred and the cached neutral generation.
   Reset the override: require the inherited route (Forward if the instance inherits
   a binding, Deferred otherwise). Repeat the cycle, then verify undo and redo.
5. Toggle coat weight between zero and nonzero while bound: routing and compiled
   program generation must remain unchanged. Test base/detail RNM normals with coat
   both disabled and enabled; neutral coat must inherit the evaluated base normal.
6. Request a missing/incompatible texture and verify failure preserves the binding,
   overrides, generation, and image. Save/reopen a separate test scene explicitly to
   check persistence; never implicitly save the user's current scene or dirty graphs.

Settle frames and capture after each successful mutation; associate comparisons with
the intended entity via ObjectID, not whole-image differences alone. Record effective
render options, profile/device, revisions, applied descriptors, and capture IDs.
Descriptor availability and successful compilation are not GPU success. Missing
captures, off-screen targets, fallback pipelines, or a skipped test leave this gate
unverified. Cancel the test transaction to restore user content when verification is
complete. Performance measurements remain a separate acceptance gate.

A listed texture GUID is not proof that its source is installed. Read-only texture
overrides resolve through the asset registry's configured content roots; missing or
undecodable sources must report the GUID, source hint, and cause while retaining the
previous material. A failed material command ends its gateway transaction and may
roll back all earlier test edits in that transaction. Read the new scene revision
and start a fresh transaction instead of retrying with old entity IDs or revisions.
 --output-on-failure
```

The native test stages an **isolated copy** of the baseline project and only its relevant source folders in a temporary CMake build directory (not inside the repository's source assets). It exercises the production `arc-cook` pipeline for a clean first cook, warm cache hit, package, and package verification. The C++ verifier mounts the resulting package with ARC's **native package_artifact_reader**, reads and checks the hash of **every named artifact** (including separate shader passes), and deserializes the cooked Material and Material Instance payloads. It requires all 13 authored assignments (including two bindings to the shared Floor instance), parent Material dependencies, both Checker and Floor Function Slot specializations and their packaged Material Function, authored numeric instance values, and a genuine packaged 2D texture. Changing a baseline instance value must invalidate the cook cache.

This is a **cooker/runtime package binding test**, not an actual GPU draw call. In addition to the positive package-load checks, the test deliberately rejects missing source textures, unknown Function Slots, stale scene Material GUIDs and invalid Material graphs. Value-only instances reuse their parent program from the same cook transaction, including a DDC cache hit; the source-import artifact is never treated as a cooked parent. GPU image/reference validation and device-specific shader fallback diagnostics are tracked in #1087.

### GPU captures and remaining scope

This first milestone establishes the canonical scene and structural contract. **Native cooked package loads are covered; deterministic GPU captures, image tolerances and versioned golden images are not yet implemented.** Those require a device-backed test runner, explicit capture of active render/material path and device profile, and reviewed reference PNG updates. Never auto-update goldens from a normal test run; missing shader/texture, wrong permutation, unplanned fallback or unavailable required path must fail clearly.

Expand this baseline rather than creating unrelated fixture scenes: normal/AO maps, anisotropy, masking/two-sided, mip/HDR and material function variants as executable. Follow #359 for authoritative compiler/runtime cleanup, #858 for the Blank 3D starter migration, and #478 for visibility-buffer coverage.
