# Terrain M3.5: non-destructive paint layers

Terrain painting now uses the same asset-owned authoring path as sculpting. A brush stroke records exact per-sample weight deltas, converts them to stable region-local addresses, and folds them into sparse Paint Layer payloads. Shared boundary samples use a canonical address, so a save/load or region repartition does not duplicate paint at seams.

Paint edits dirty the attributes domain only. The rebuild scheduler expands affected regions by one source sample for shared-boundary consistency, evaluates the Paint Layer on immutable worker snapshots, and publishes new regional attribute textures without rebuilding or replacing terrain geometry. Geometry and attribute residency remain independently addressable for later streaming work.

During an active stroke, the editor updates only the asset-owned proxy's affected attribute textures. It retains geometry handles, selection/picking identity, and asset ownership. Cancelling restores the scene weights and republishes the restored attributes; committing atomically writes the sparse layer payload and starts the authoritative regional rebuild. Terrain authoring work is pumped independently of successful viewport presentation, so backend attachment, resize, or recovery cannot freeze a stroke or its rebuild.

Create Terrain now establishes the unified workflow immediately when a project asset registry is available. It writes the generated surface as a 16-bit heightmap source, creates a `.terrain` asset with default Sculpt and Paint layers, assigns its stable asset identity to the scene component, and transitions from the immediate compatibility surface to asset-owned regions in the background. Projects without an asset root retain the inline heightfield path for compatibility and tests.

Each rebuild session decodes one immutable heightmap snapshot and shares it across all regional jobs. Authoring region size is derived from 64 source quads and remains independent from runtime render clusters or pages. The current RGBA weight cache is still the compatibility representation for four material layers; arbitrary named channels and their streaming policy remain Milestone 6 work.

Regression coverage verifies stable paint accumulation, JSON round trips, attribute-domain dirtiness and halo propagation, attribute-only publication, unchanged geometry handles, asset-backed creation and persistence, live preview ownership, and cancellation rollback.
