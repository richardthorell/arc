# Material compiler compatibility audit

Issue: #359

## Authoritative path

ARC's current material authoring/cooking path is intentionally narrow:

1. `render::tools::parse_material_authoring_json()` validates authoring schema v4 and rejects legacy top-level material fields.
2. Graph materials compile through `compile_material_graph_json()` and `make_graph_material_evaluator()`.
3. Handwritten Material Shaders compile through `make_custom_material_evaluator()`.
4. Both paths converge on `generate_material_pass_slang()` and the normal shader compiler/cache.
5. The cooker publishes `material_package_v3`; runtime consumes compiled pass bindings rather than evaluating authoring JSON.

The asset cooker therefore does not need a second material evaluator or schema adapter. New material features should extend this path instead of introducing editor-, cooker-, or runtime-specific evaluation semantics.

## Compatibility inventory

### Explicit compatibility that should remain

- Material authoring documents must carry the current schema version. Unsupported versions fail with an actionable `unsupported_version` error.
- Legacy top-level fields (`shader`, `surface`, `textures`, `advanced`) are rejected explicitly rather than interpreted heuristically.
- Unknown editor metadata is preserved in the canonical authoring JSON so editor-only metadata can round-trip without affecting compilation.
- Handwritten Material Shaders remain a supported first-class implementation path; they converge on the same pass codegen/compiler/package contracts as graph materials.
- Terrain packages may omit compiled surface passes because terrain uses its dedicated renderer path.

### Compatibility behavior still to remove or make explicit

`material_asset.cpp` still silently maps unknown current-schema semantic strings to defaults:

- unknown `domain` values become `surface`;
- unknown `shadingModel` values become `standard`;
- unknown `blendMode` values become `opaque`;
- `custom_lit` is accepted as an alias for the canonical `customLit` spelling.

These are compatibility fallbacks inside the otherwise strict v4 parser. They should be replaced by explicit validation/migration so a typo or stale semantic cannot silently compile with different rendering behavior.

## Regression coverage already present

`engine/render-tools/tests/material_asset_tests.cpp` covers current graph/handwritten authoring, rejection of legacy schema versions and fields, exact graph-vs-shader implementation selection, deterministic package serialization, and the terrain package exception.

Follow-up cleanup should add focused tests for strict semantic-enum validation before removing the fallbacks above.

## Guardrails

- Do not add a second evaluator in the editor or cooker.
- Do not infer old schemas from field shape; migrations must be versioned and explicit.
- Parameter-only authoring changes must not force shader recompilation unless they alter static/permutation semantics.
- Runtime material loading must continue to consume cooked package/pass contracts, not authoring graph JSON.
- Compatibility code must have a documented source version, target version, and removal condition.
