# Agent material contract (#986)

## Current checkpoint

`assets.inspect` and `materials.describe` are registered through the shared #792 operation catalog. Their Zod definitions in `editor/src/common/materialOperationContract.ts` supply transport schemas and TypeScript types. Native inspection uses the saved source and Material IR reflection, not open document buffers or a JavaScript graph evaluator.

`assets.inspect` resolves GUID identity through the logical Asset Database and reports its catalog/record revisions and provider vocabulary. Material dependencies resolve through the same database. The Asset Search S1 index (`asset_search_index` / `asset_search_indexer`) consumes this database too. This work does not create a second inventory, search index, ranking implementation, or change cursor. Existing `assets.list` pagination remains unchanged; later indexed discovery should adapt that existing surface to the shared search backend.

Material definition revisions hash saved source bytes and resolved dependency GUIDs/bytes. A successful description means the definition passed native reflection validation, **not** that its GPU pipelines compiled, were applied to an entity, or were visually observed. `runtimeStatus: notInspected` preserves that distinction. Unavailable parameter values are explicitly marked instead of inferred. Handwritten shader reflection is currently reported unsupported.

Dependency collection is bounded to 128 registered sources, 8 MiB per source and 32 MiB total. Function alternatives are tested by the native compiler against the material's declared interfaces/allowed alternatives, with at most 256 trial compilations and 16 returned choices per slot. Truncation is explicit. Legacy path-only dependencies must resolve unambiguously; an unresolved GUID never falls back to its path hint.

## Remaining before closing #986

- Complete definition/value reflection (including texture requirements and source defaults), entity inspection, and matching stable IDs across native realization paths.
- Expose typed assignment, override/reset and function-selection operations with approved scene transactions, world/scene authority, and definition-revision checks. The native parameter-service checkpoint is not yet an exposed agent mutation API.
- Move the Inspector to those same services and remove encoded-name readback and procedural-token material edits, retaining compatibility adapters.
- Finish agent verification instructions, undo/reopen/security and last-good tests, and live rendered acceptance.

Shared graph editing and mixed-domain transactions remain #792 work. Neither issue is closed by this checkpoint. Full native compilation/platform validation is deferred to CI; focused editor checks run locally.
