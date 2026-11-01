# Material compiler authority

ARC has one authoritative material compilation path: the native Material IR/compiler under `engine/render-tools/`, consumed by the cooker/runtime path and surfaced to the editor through native compile responses.

The renderer-side `editor/src/renderer/src/material/materialCompiler.ts` module is an adapter only. It may normalize native diagnostics and derive authoring-only presentation metadata such as exposed parameter controls, but it must not type-check, evaluate, lower, or compile material graphs independently.

## Compatibility boundary

Legacy material assets are handled by the explicit migration layer in `materialAssetMigration.ts`. Migration may normalize old authored schema into the current graph representation, but migrated graphs must then use the same native compiler path as newly authored graphs. Compatibility behavior must not grow into a second evaluator or shader-generation path.

## Guardrail

`materialCompilerAuthority.test.ts` protects this boundary by rejecting renderer-side graph/node compile or evaluate entry points in the editor compiler adapter. When compatibility support is required, prefer an explicit, deterministic asset migration followed by native compilation.

## Render-path presentation

Material Editor render-path labels use the optional `materialRenderPath` field in a successful native `shader.compile` response (`deferred` or `clusteredForward`). The host realizes the complete in-memory `previewSource`, including authored settings and registered functions, and reports the resolved descriptor only when a runtime program exists. Live material preview surfaces reuse that compiled descriptor instead of independently compiling the source again.

The label describes the compiled document candidate, not GPU publication or visual validation. It does not describe an entity's Material Instance overrides. Unknown/missing paths and failed compiles display `Unavailable`; outstanding graph compiles display `Pending compilation`. Custom shaders without this native compile response remain unavailable rather than being guessed as Forward. No graph-connection heuristic may fill in missing routing metadata.

Semantic graph and setting edits invalidate in-flight compile replies immediately, including during the debounce window and while live update is disabled. Settings are compiled automatically only when live update is enabled; changes remain unsaved until explicit Save.

Issue #1050 still owns live rendered/instance agreement and the Materials Baseline performance acceptance. A compiled-program label alone does not satisfy those gates.

## Remaining audit work

Issue #359 remains responsible for auditing the native/editor/cooker call sites for obsolete schema adapters or duplicate evaluation behavior, removing any that remain, verifying deterministic old-asset migration, and proving parameter-only edits avoid shader rebuilds where compilation is unnecessary.
