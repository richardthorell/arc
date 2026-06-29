# ARC Assets

Small default assets used by the editor and renderer bring-up live here.

- `models/UAL2_Standard.glb` is the default startup mesh.
- `shaders/default_unlit.vert` and `shaders/default_unlit.frag` are the readable
  sources for the current default Phong-style mesh shader.

These are development assets for editor smoke testing, not the final engine asset
database. The renderer currently uses checked-in/generated shader data for
bootstrap reliability while the shader compiler and hot-reload path matures.
