# ARC Editor

This folder contains the first ARC editor shell. It is a cross-platform C++
application that links the engine in-process, uses SDL for window/input hosting,
and uses Dear ImGui docking for editor panels.

The editor currently exercises the engine renderer and scene stack directly. It
creates a default scene, loads the startup GLB from `assets/`, drives the Vulkan
viewport when `ARC_BUILD_RENDER_VULKAN=ON`, and keeps the temporary SDL-renderer
path available for fallback runs.

Editor code should include engine APIs through the shared module layout:

```cpp
#include <arc/framework/framework.h>
#include <arc/input/input.h>
#include <arc/render/render.h>
#include <arc/scene/scene.h>
```

Build it from the repo root with:

```bash
cmake -B build -S . -DCMAKE_BUILD_TYPE=Release -DARC_BUILD_EDITOR=ON
cmake --build build --config Release --target arc_editor --parallel
```

Or build it if needed and run it with:

```bash
python run_editor.py
```

Use `python run_editor.py --no-vulkan-render` to force the fallback editor path.

Third-party dependencies are configured from the shared root `third_party/`
folder so editor and engine dependencies can use the same dependency policy.
