# Terrain M3.4: asynchronous region rebuilds

Asset-backed terrain now has an editor-owned rebuild session. Sculpt commits submit the dirty authoring regions and their seam dependencies to ARC's shared job system. Stack edits invalidate geometry across the asset. Opening a bound terrain schedules its initial asset-owned generation.

Workers own immutable asset snapshots and produce region-local evaluated surfaces, virtual/conventional geometry, and attribute textures. They never capture the editor world or renderer. Publication polls without waiting, rejects obsolete authoring/dirty revisions, stages all resources, and replaces the batch at the frame boundary. Failed staging leaves the visible generation intact and retries. Dirty geometry/topology domains are acknowledged only after publication; unrelated domains remain dirty.

The compatibility brush surface remains visible during a stroke and while its new generation builds. Its previous asset-owned region resources remain available so unaffected regions keep their handles when the build publishes. Publication updates the picking/sculpt sample cache from the evaluated regions and returns rendering to the asset-owned result. Scene resets and entity removal discard sessions without waiting for workers. Asset preview viewports and play mode do not publish into the authoring scene.

The initial editor provider supports flat and resolved heightmap sources on an axis-aligned sample grid. It crops evaluation to each authoring region with a one-sample halo, applies sparse payloads across shared borders, and supplies matching boundary normals. Mesh/procedural providers and topology-changing modifiers remain later roadmap work. Heightmap decoding runs on workers but currently decodes the source per scheduled region; a shared source-tile cache remains an optimization. GPU uploads occur at publication; this milestone does not add upload budgeting or a new cooked-disk cache.

Regression coverage includes dirty-only scheduling, immutable snapshots, stale results, worker failure, abandoned queues, staged-publication rollback, unchanged region handles, and the sculpt-preview transition.
