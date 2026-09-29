# Material compiler authority

ARC has one authoritative material compilation path: the native Material IR/compiler under `engine/render-tools/`, consumed by the cooker/runtime path and surfaced to the editor through native compile responses.

The renderer-side `editor/src/renderer/src/material/materialCompiler.ts` module is an adapter only. It may normalize native diagnostics and derive authoring-only presentation metadata such as exposed parameter controls, but it must not type-check, evaluate, lower, or compile material graphs independently.

## Compatibility boundary

Legacy material assets are handled by the explicit migration layer in `materialAssetMigration.ts`. Migration may normalize old authored schema into the current graph representation, but migrated graphs must then use the same native compiler path as newly authored graphs. Compatibility behavior must not grow into a second evaluator or shader-generation path.

## Guardrail

`materialCompilerAuthority.test.ts` protects this boundary by rejecting renderer-side graph/node compile or evaluate entry points in the editor compiler adapter. When compatibility support is required, prefer an explicit, deterministic asset migration followed by native compilation.

## Remaining audit work

Issue #359 remains responsible for auditing the native/editor/cooker call sites for obsolete schema adapters or duplicate evaluation behavior, removing any that remain, verifying deterministic old-asset migration, and proving parameter-only edits avoid shader rebuilds where compilation is unnecessary.
