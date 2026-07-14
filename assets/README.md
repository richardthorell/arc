# ARC Assets

Small default assets used by the editor and renderer bring-up live here.

- `models/UAL2_Standard.glb` is the default startup mesh.
- `environments/` contains small CC0 HDRIs for outdoor lighting and future IBL
  tests.
- `textures/terrain/` contains small CC0 terrain material maps for renderer and
  terrain-editor bring-up.
- `materials/` contains editable `.arcmat` JSON material assets. These can be
  dragged from the Content Browser onto mesh renderer material slots or directly
  onto objects in the viewport.
- Texture slots accept `.png`, `.jpg`, `.jpeg`, `.tga`, `.hdr`, and `.dds`.
  Common images are decoded to upload-ready RGBA pixels in graphics builds.
  DDS headers, mip payloads, and BC compression metadata remain intact for
  native GPU upload, with a visible fallback on unsupported devices.
- Scene-like assets (`.glb`, `.gltf`, `.fbx`) can be opened from the editor File
  menu or Content Browser. GLB is imported through the current static mesh path.
  FBX is recognized and reports a clear unsupported diagnostic unless the
  optional `ufbx` dependency is enabled and wired for static scene conversion.
- `shaders/default_phong.vert`, `shaders/default_phong.frag`, and
  `shaders/shadow_depth.vert` are the readable sources for the current
  Phong-and-shadow bring-up shaders.
- `ASSET_LICENSES.md` records third-party asset sources and licenses.

These are development assets for editor smoke testing, not the final engine asset
database. The renderer currently uses checked-in/generated shader data for
bootstrap reliability while the shader compiler and hot-reload path matures.
Some material presets use future-facing shader names such as `arc/terrain_phong`
or `arc/water_preview`; these are safe metadata labels for now and fall back to
the current renderer material path until dedicated shader variants exist.
