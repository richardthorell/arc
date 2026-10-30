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
 --output-on-failure
```

The native test stages an **isolated copy** of the baseline project and only its relevant source folders in a temporary CMake build directory (not inside the repository's source assets). It exercises the production `arc-cook` pipeline for a clean first cook, warm cache hit, package, and package verification. The C++ verifier mounts the resulting package with ARC's **native package_artifact_reader**, reads and checks the hash of **every named artifact** (including separate shader passes), and deserializes the cooked Material and Material Instance payloads. It requires all 13 authored assignments (including two bindings to the shared Floor instance), parent Material dependencies, both Checker and Floor Function Slot specializations and their packaged Material Function, authored numeric instance values, and a genuine packaged 2D texture. Changing a baseline instance value must invalidate the cook cache.

This is a **cooker/runtime package binding test**, not an actual GPU draw call. GPU image/reference validation is tracked in #1087. The direct material renderer integration path, shader/texture fallback diagnostics and additional negative shader/permutation scenarios should be expanded before closing #1086 if CI exposes gaps.

### GPU captures and remaining scope

This first milestone establishes the canonical scene and structural contract. **It does not yet implement cooked runtime load validation, deterministic GPU captures, image tolerances or versioned golden images.** Those require a device-backed test runner, explicit capture of active render/material path and device profile, and reviewed reference PNG updates. Never auto-update goldens from a normal test run; missing shader/texture, wrong permutation, unplanned fallback or unavailable required path must fail clearly.

Expand this baseline rather than creating unrelated fixture scenes: normal/AO maps, anisotropy, masking/two-sided, mip/HDR and material function variants as executable. Follow #359 for authoritative compiler/runtime cleanup, #858 for the Blank 3D starter migration, and #478 for visibility-buffer coverage.
