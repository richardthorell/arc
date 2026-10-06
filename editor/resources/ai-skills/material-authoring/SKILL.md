---
format: arc-skill
formatVersion: 1
id: material-authoring
name: Material Authoring
version: 1.1.0
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

Inspect the target entity, component schema, and existing assets before authoring or binding a material. Prefer typed material definitions and validated project asset references over raw file assumptions.

## Material selection

Prefer an existing reusable material over creating a new graph when it already represents the requested surface behavior.

Use **Standard Lit** for ordinary opaque metallic/roughness PBR surfaces. It is ARC's general-purpose default material and should be the first choice for common textured or solid-color objects.

Use specialized material families only when the requested rendering behavior requires them, such as Unlit, Foliage, Water, Terrain, Transmission/Glass, or Subsurface.

Do not create a new shader merely to apply a texture or change a common surface parameter.

## Standard Lit workflow

Standard Lit is texture-ready by default.

- Use **Base Color Texture** for the surface color texture.
- Use **Base Color Tint** to tint or scale the texture color.
- The effective base color is the texture multiplied by the tint.
- If no Base Color Texture is assigned, the material behaves as a solid-color material using Base Color Tint.
- Assigning or replacing a Base Color Texture is a parameter change; do not rebuild graph topology just to swap the texture.
- Prefer exposed material parameters for common appearance changes such as color, roughness, metallic response, and supported texture inputs.

When the user's request is simply to make an object use an image texture, prefer assigning that texture to Standard Lit over constructing a custom Texture Sample graph.

## Authoring and verification

Inspect available assets before creating duplicates. When authoring a new material, start from ARC's standard material graph and only add graph complexity that is required by the requested effect.

All persistent changes must stay inside the editor harness approval and transaction flow described by the ARC Editor Gateway concepts. Verify the affected entity and rendered result before commit; cancel when the result is stale, incomplete, or visually incorrect. Skill instructions never substitute for edit approval.
