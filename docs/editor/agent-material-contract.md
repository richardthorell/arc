# Agent material contract (#986)

## Current checkpoint

`assets.inspect` and `materials.describe` are registered through the shared #792 operation catalog. Their Zod definitions in `editor/src/common/materialOperationContract.ts` supply transport schemas and TypeScript types. Native inspection uses the saved source and Material IR reflection, not open document buffers or a JavaScript graph evaluator.

`assets.inspect` resolves GUID identity through the logical Asset Database and reports its catalog/record revisions and provider vocabulary. Material dependencies resolve through the same database. The Asset Search S1 index (`asset_search_index` / `asset_search_indexer`) consumes this database too. This work does not create a second inventory, search index, ranking implementation, or change cursor. Existing `assets.list` pagination remains unchanged; later indexed discovery should adapt that existing surface to the shared search backend.

Material definition revisions hash saved source bytes and resolved dependency GUIDs/bytes. A successful description means the definition passed native reflection validation, **not** that its GPU pipelines compiled, were applied to an entity, or were visually observed. `runtimeStatus: notInspected` preserves that distinction. Unavailable parameter values are explicitly marked instead of inferred. Handwritten shader reflection is currently reported unsupported.

Dependency collection is bounded to 128 registered sources, 8 MiB per source and 32 MiB total. Function alternatives are tested by the native compiler against the material's declared interfaces/allowed alternatives, with at most 256 trial compilations and 16 returned choices per slot. Truncation is explicit. Legacy path-only dependencies must resolve unambiguously; an unresolved GUID never falls back to its path hint.

## Entity-local editing

`materials.inspectEntity` describes a GUID-backed entity binding, inherited and local parameter values, selected functions, and the renderer resource handle. This is a saved-definition inspection with local overrides, not a claim that an arbitrary dirty document has been published. `resourceAlive` means the renderer owns the handle; it is not GPU pipeline or visual verification.

`materials.assign`, `materials.setOverride`, `materials.resetOverride`, and `materials.selectFunction` are registered schema-driven operations. They require an approved scene edit session, expected scene revision, world epoch, and the inspected definition revision. The native host checks authority and definition revision again before realizing a replacement through scene history. Values use string parameter IDs and GUID-backed texture references. Explicit malformed, out-of-range, missing-parameter and incompatible-function requests fail rather than silently becoming no-ops. Stale persisted overrides produce recovery diagnostics.

The Inspector migration from #1060 has been rolled back pending UI parity: the native description does not yet carry all authored grouping, color-editor, texture-default and function-control information. The Inspector temporarily retains its prior authoring-metadata presentation and native compatibility commands, including same-material multi-selection, while using shared `UiPanelCardRow` controls. The typed agent operations above remain available and unchanged. Do not migrate the Inspector again until the restored UI regression suite and live nested-function/texture tests pass with the replacement contract.

The legacy batch `entity.setBaseColor` operation is a compatibility adapter to `materials.setOverride`. It requires one native-reflected exposed parameter directly connected to the base-color output; it never searches display names. Ambiguous or indirect color composition returns discovered candidates and asks the caller to use the typed operation.

### Typed clear and reset operations

`materials.clearTexture` accepts the existing edit authority and a stable `parameterId`, with no replacement value. It writes an explicit empty texture override. Inspection reports `overridden: true`, `resolvedValue: null`, and `valueStatus: resolved`; this differs from an unavailable default. `materials.resetOverride` removes that local entry and restores inheritance. Failed explicit texture realization retains the previous binding instead of publishing an empty replacement.

`materials.resetFunction` accepts edit authority and a stable `slotId`, with no asset GUID or replacement value. The host validates the slot against the current inspected definition before removing its local function selection and local input overrides. This restores the assigned material/instance's inherited selection, not necessarily the parent graph's authored default. Both operations use the existing approved scene transaction and history path and are discovered from the shared operation catalog.

This clear/reset increment does not migrate the Inspector. Texture default/requirement reflection, atomic multi-selection, and the final versioned parity contract remain follow-up work before the UI switch and legacy retirement.

### Paginated function discovery

`materials.listFunctions` accepts exactly one material `guid` or `entityGuid`, a stable `slotId`, and a page `limit` of 1–16. Start at offset zero; continue with `nextOffset` and `discoveryRevision` until the offset is null. Offsets address deterministic GUID-ordered candidates, not returned matches. The revision covers saved source/dependency bytes, local overrides, and slot identity. Stale continuations fail without mutation; restart discovery rather than combine generations.

Pages compile candidates from the saved definition's loaded function library/dependency closure. This is not a second global asset inventory. The existing dependency limits still apply. `materials.selectFunction` validates the requested GUID through native realization and compiler compatibility, rather than requiring it to appear in the first 16 preview choices. Neither discovery nor selection saves dirty documents or changes the Inspector transport.

## Remaining acceptance work before closing #986

### Inspector parity migration checkpoint

Native graph compilation now returns authoring-only parameter presentation separately from shader layout: editor kind, group identity/label/order, and parameter order. Function inlining namespaces group identities and retains their presentation. Material inspection projects this data by stable parameter ID, exposes explicit local parameter values, and distinguishes inherited function selections from entity overrides. These additive fields are optional during the contract rollout; the future Inspector migration must require the completed, versioned contract rather than guess missing metadata.

The restored Inspector is unchanged at this checkpoint. Typed texture clearing, function reset, and paginated discovery are available. Texture defaults/requirements, atomic multi-selection, revision-safe Inspector integration, and legacy command retirement remain required before removing its compatibility path. Existing saved scene data must remain readable.

- Complete texture requirement/default-source reflection and confirm inherited values across every function/instance realization path.
- Audit saved-definition versus cached/dirty runtime generations on assignment, dependency rename/move recovery, and parameter-update specialization caching.
- Validate native compilation, undo/reopen, function/texture last-good behavior and the seeded multipart rendered scenario on an updated editor build. Native reflection and parameter isolation/rejection regression tests are included, but are not a substitute for rendered acceptance.
- Verify restored Inspector presentation for all parameter kinds and groups without reintroducing a frontend graph evaluator.

Shared graph editing and mixed-domain transactions remain #792 work. Neither issue is closed by this checkpoint. Full native compilation/platform validation is deferred to CI; focused editor checks run locally.
