---
format: arc-skill
formatVersion: 1
id: material-authoring
name: Material Authoring
version: 1.4.0
description: Inspect, create, bind, and visually verify ARC materials through typed asset and scene-edit workflows.
requires:
  - scene.read
  - scene.mutate
  - asset.read
  - asset.mutate
  - viewport.read
tools:
  - assets.list
  - assets.inspect
  - materials.describe
  - materials.inspectEntity
  - materials.assign
  - materials.setOverride
  - materials.resetOverride
  - materials.clearTexture
  - materials.selectFunction
  - materials.resetFunction
  - scene.getEntity
  - scene.componentSchemas
  - edit.request
  - edit.begin
  - edit.apply
  - edit.commit
  - edit.cancel
  - viewport.debug
contexts:
  - project
  - selection
  - assets
  - viewport
  - diagnostics
---

# Material Authoring

Inspect the target entity, component schema, and the live asset inventory (`assets.list`) before authoring or binding a material, including when the user asks for an entire scene or model rather than naming materials. Discover both project and built-in materials, textures, and Material Functions; do not infer available assets from the initial context or assume the absence of assets because they were not mentioned. Prefer typed material definitions and validated asset references over raw file assumptions.

## Material selection

Choose from the materials and functions actually reported by the engine/project. Prefer an existing reusable material over creating a new graph when it already represents the requested surface behavior. Inspect the chosen material's exposed parameters and selectable function sources before deciding how to style each surface. A built-in function (such as Checker, Gradient, or Noise), a texture already in the project, or an existing glass/transmission material may suit a surface without new graph authoring. These are examples, not a fixed mapping from object types to materials.

Use **Standard Lit** where its _currently discovered_ inputs and features support the intended opaque PBR surface. Do not presume every project material or function shares Standard Lit's authoring contract.

Use specialized material families or systems only when the requested rendering behavior requires them, such as Unlit, Water, Terrain, Transmission/Glass, or Subsurface. Foliage should use the existing Standard Lit, Transmission, or Subsurface capabilities unless ARC gains dedicated foliage rendering behavior.

Do not create a new shader merely to apply a texture or change a common surface parameter.

## Typed entity-local workflow

Follow **discover → describe → approve → assign/override → inspect resolved state → observe when possible**.

1. Use `assets.list` for existing content and `assets.inspect` for identity, scope, dependencies and readiness. Use `materials.describe` with the chosen asset GUID to inspect its saved/published definition. This does not apply open dirty documents.
2. Use `materials.inspectEntity` with the persistent entity GUID to inspect inherited values, local overrides, selected functions and the runtime resource handle. Keep parameter IDs as strings, including 64-bit IDs; use stable slot IDs, not display labels.
3. Request approval and begin a scene edit session. Use the registered material operations with current scene/world authority and the inspected definition revision. Describe the destination asset before `materials.assign`; inspect the entity before local edits. Reinspect after changing or resetting a function because the exposed layout may change.
4. Before committing, verify the resulting material GUID, parameter values and function selections through `materials.inspectEntity`. A live resource handle is not evidence that a GPU pipeline compiled or that the appearance is correct. Read diagnostics and capture the viewport with ObjectID association when possible.
5. Report authored-state verification, runtime resource availability and visual observation separately. Off-screen objects, unavailable captures and unresolved compilation do not justify visual-success claims. Never implicitly save a scene or shared asset.

Do not manufacture encoded procedural commands or use `entity.setBaseColor` as a general material API. Its compatibility adapter only accepts an unambiguous color parameter identified by native output semantics. Discover IDs and supported values instead. Function compatibility comes from native compiler metadata, not names such as glass or roof.

For a color change, use `materials.setOverride` with the discovered parameter ID and its declared RGB/RGBA shape; do not guess from a label such as Base Color. For textures, supply the discovered Texture2D asset reference (`guid` and `pathHint`), not a filesystem path. `materials.clearTexture` takes the parameter ID and writes an explicit empty override; `materials.resetOverride` instead removes the local override and restores the inherited value. `materials.selectFunction` selects by slot ID and function GUID; `materials.resetFunction` takes only the slot ID and removes its local selection and local inputs, restoring the material/instance's inherited function. Clear/reset operations take no replacement value.

Use live operation discovery to confirm support in the running build. Texture-default/requirement discovery and atomic multi-selection are not yet complete; report unavailable metadata rather than guessing or claiming full Inspector parity.

## Existing material workflow

Consult the actual material definition and its selectable function sources rather than assuming a globally fixed parameter set.

- When the selected material/function exposes **Base Color Texture**, use it for the surface color texture.
- When exposed, **Base Color Tint** can tint or scale the texture color.
- Check the selected function's graph for how tint, texture, and other inputs compose; do not assume every source multiplies them.
- A solid-color source may use Base Color Tint, while a selectable checker/noise/gradient source can have different parameters.
- Prefer exposed parameter/selection changes to rebuilding graph topology when the existing material supports the intended appearance.
- Bind parameters only when the inspected material/function exposes them. Do not invent parameter names, function choices, or unsupported material assignments.

When an image texture is requested, reuse a compatible textured material/function and bind the existing texture where supported instead of constructing a custom graph by default.

## Authoring and verification

Inspect available assets before creating duplicates. When authoring a new material, start from ARC's standard material graph and only add graph complexity that is required by the requested effect.

All persistent changes must stay inside the editor harness approval and transaction flow described by the ARC Editor Gateway concepts. Before commit, read back the edited entities and assigned material references (including relevant overrides/source selections), then check the rendered viewport. A successful edit.apply or editor.applyBatch response alone only confirms command processing, **not** that the appearance rendered or persisted. If the agent cannot inspect or apply a required parameter, state the limitation instead of reporting success. Cancel when the result is stale, incomplete, or visually incorrect. Skill instructions never substitute for edit approval.
