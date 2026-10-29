# Materials Baseline (#943)

Open the repository-root `MaterialsBaseline.arcproject` from the ARC editor, then open its default scene, **Materials Baseline**. The scene is repository/engine-owned, not derived from the Blank 3D template.

### Stable baseline v1

- Fixed scene/entity/material GUIDs, authored camera pose, manual exposure, directional light and solid-color environment.
- Fixed sky time with clouds, fog, stars, moon, animation and environment playback disabled.
- Known-UV engine primitives, grouped into core material families, Standard Lit response, and composition/texture/emission.
- Material Instances share the production Standard Lit parent; a Checker-specialized instance and the reusable Floor instance exercise the shared function workflow.
- Native scene asset bindings remain authoritative; no renderer-only stand-ins or texture substitutions.

### Validation

```sh
python tools/tests/materials_baseline.py --root .
cmake --preset default
cmake --build --preset default --parallel
ctest --preset default -R arc-materials-baseline-contract
```

The test checks scene inputs, stable material references, GUIDs, metadata, and dependency manifests, with diagnostic names for failed capability entries.

### GPU captures and remaining scope

This first milestone establishes the canonical scene and structural contract. **It does not yet implement cooked runtime load validation, deterministic GPU captures, image tolerances or versioned golden images.** Those require a device-backed test runner, explicit capture of active render/material path and device profile, and reviewed reference PNG updates. Never auto-update goldens from a normal test run; missing shader/texture, wrong permutation, unplanned fallback or unavailable required path must fail clearly.

Expand this baseline rather than creating unrelated fixture scenes: normal/AO maps, anisotropy, masking/two-sided, mip/HDR and material function variants as executable. Follow #359 for authoritative compiler/runtime cleanup, #858 for the Blank 3D starter migration, and #478 for visibility-buffer coverage.
