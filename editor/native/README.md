# ARC Editor Native Host

The native host owns authoritative scene state, history, persistence,
asset/material integration, viewport interaction, and renderer submission for
the Electron editor. It communicates with Electron through the typed
line-delimited host protocol.

Build it from the repository root with:

```bash
cmake --preset editor-vulkan
cmake --build --preset editor-vulkan --target arc_host_process --parallel
```

The host has no UI toolkit dependency. SDL, Dear ImGui, and ImGuizmo belonged
to the retired editor shell and are intentionally not part of this target.
