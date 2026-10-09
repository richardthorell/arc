---
format: arc-skill
formatVersion: 1
id: material-authoring
name: Material Authoring
version: 1.2.0
description: Inspect, create, bind, and visually verify ARC materials through typed asset and scene-edit workflows.
requires:
  - scene.read
  - scene.mutate
  - asset.read
  - asset.mutate
  - viewport.read
tools:
  - assets.list
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

## Standard Lit workflow

Consult the actual Standard Lit definition and its selectable function sources rather than assuming a globally fixed parameter set.

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
