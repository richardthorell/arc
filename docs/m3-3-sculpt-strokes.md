# M3.3 Terrain Sculpt Stroke Ownership

M3.3 moves non-paint terrain brush strokes from destructive editor-only state into asset-owned sparse Sculpt Layer payloads.

## Authoring flow

```text
viewport Sculpt / Smooth / Flatten stroke
    -> compatibility heightfield preview
    -> exact per-sample height deltas
    -> stable authoring-region coordinates
    -> selected Sculpt Layer sparse payload
    -> one TerrainAsset revision on stroke commit
    -> mark touched geometry regions dirty
    -> save + reimport TerrainAsset
```

Persistent sculpt samples use a normalized 16-bit region-local X/Z address rather than runtime heightfield indices. This keeps authored edits independent of the current preview or render resolution.

The selected modifier is identified by its stable Terrain modifier ID and is host-authoritative. Sculpt strokes require an enabled Sculpt Layer on asset-backed terrain. Legacy inline terrain keeps the compatibility editing path.

## Milestone boundary

M3.3 establishes non-destructive sculpt ownership and evaluation. During an active stroke the editor still updates the legacy heightfield as an immediate preview. M3.4 will consume the dirty authoring regions for asynchronous incremental evaluation/build and publish rebuilt region generations. Paint continues to use the compatibility material-weight path until M3.5.
