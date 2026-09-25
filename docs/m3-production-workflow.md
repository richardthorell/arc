# Terrain M3: production editor workflow

This final M3 slice makes the asset-owned sculpt and paint pipeline the visible, authoritative Terrain Editor
workflow.

## Modifier stack

The Terrain Stack panel operates on stable modifier IDs and supports:

- sculpt and paint layer creation
- layer selection, naming, enable/disable, and deletion
- duplication of descriptors and their sparse region payloads with a fresh stable ID
- direct drag reorder, with button-based ordering retained for keyboard-accessible operation

Inspecting or selecting a layer is transient editor state. It does not dirty the scene, create an undo entry, or
require project write access. Mutating operations atomically write the `.terrain` source and schedule a rebuild.

## Rebuild feedback

The host exposes `idle`, `queued`, `building`, `publishing`, and `failed` phases with the current authoring revision,
dirty-region count, geometry-region count, attribute-region count, and persistent error text. The panel polls only
while work is active.

In Terrain viewport mode, dirty authoring-region borders follow the compatibility surface used for interactive
brush feedback. Geometry dirtiness is orange, attribute dirtiness is cyan, combined dirtiness is magenta, and failed
work is red. The overlay disappears after the authoritative generation publishes.

## Authoritative undo and redo

Interactive height and weight arrays remain a low-latency preview cache, but they are not the saved source of an
asset-backed edit. A terrain history entry now carries the matching before/after `TerrainAsset` state. Undo and redo:

1. restore the preview-grid delta or scene snapshot;
2. atomically restore the authoritative `.terrain` document;
3. invalidate the asset-manager entry; and
4. schedule regional evaluation and publication while retaining the previous render generation.

This applies to brush strokes and modifier-stack mutations, so reloading the project after undo produces the same
terrain shown in the editor. Terrain asset history contributes to the existing bounded editor-history budget.

## Compatibility boundary

Old scenes without a `TerrainAsset` can still use the inline heightfield path so existing content loads and remains
editable. Newly created terrain and all asset-backed sculpt/paint work use modifier payloads as their authority. M4
can replace the evaluated grid provider with mesh-native/adaptive topology without changing this editor contract.
