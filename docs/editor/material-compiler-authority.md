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

### Applied entity readback

`materials.inspectEntity` contract v2 returns `entity.appliedRuntime` from the renderer descriptor attached to the entity, after inherited and entity-local instance overrides. It does not recompile the saved definition to guess the applied path. `renderPath`, `independentClearCoatNormal`, and the decimal-string `programGeneration` describe that attached descriptor; missing resources report `unavailable` with null values. A descriptor without a compiled program has a null program generation.

The existing `runtimeHandle.generation` identifies resource lifetime. `programGeneration` identifies the compiled program, not a monotonic edit counter: switching back to a cached variant may restore an earlier generation, and parameter-only edits may retain it. Saved-source reflection and `definitionRevision` remain distinct from applied runtime state; a rejected replacement can leave the last-good descriptor attached.

`runtimeStatus` is `notInspected` for asset-only descriptions, `descriptorAvailable` for an attached renderer descriptor, or `unavailable` for a missing descriptor. Renderer publication queues backend work, so descriptor availability is **not** GPU completion. `gpuVerified` and `visualVerified` remain false. No dirty document is saved or implicitly applied by this query.

Issue #1050 still owns live rendered switching verification and the Materials Baseline performance acceptance. Neither a compiled-program label nor descriptor readback alone satisfies those gates.

## Remaining audit work

Issue #359 remains responsible for auditing the native/editor/cooker call sites for obsolete schema adapters or duplicate evaluation behavior, removing any that remain, verifying deterministic old-asset migration, and proving parameter-only edits avoid shader rebuilds where compilation is unnecessary.
