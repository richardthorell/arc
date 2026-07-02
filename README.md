# ARC

[![MSBuild](https://github.com/richardthorell/arc/actions/workflows/msbuild.yml/badge.svg)](https://github.com/richardthorell/arc/actions/workflows/msbuild.yml)

ARC is an early-stage 3D engine monorepo. The current codebase is focused on
the low-level foundation an engine needs before higher-level rendering,
scene, asset, and editor systems land.

Public engine headers now follow one include convention:

```cpp
#include <arc/<module>/<header>.h>
```

## Current Shape

- `engine/` contains the core engine library.
- `engine/simd/` provides portable SIMD primitives and game-oriented SIMD helpers.
- `engine/math/` provides vector, matrix, and quaternion types built on top of SIMD where available.
- `engine/geometric/` provides simple 2D/3D primitives such as points, lines, boxes, circles, and spheres.
- `engine/diagnostics/` provides logging and diagnostic sinks.
- `engine/jobs/` provides the shared job/thread worker service.
- `engine/memory/` provides tracked memory-resource instrumentation.
- `engine/framework/` provides the platform-neutral application lifecycle.
- `engine/input/` provides runtime keyboard, mouse, and future gamepad bindings.
- `engine/render/` provides the backend-neutral rendering foundation, Vulkan
  backend, render graph, renderer resource handles, and scene draw packet
  preparation.
- `engine/scene/` provides ECS scene primitives, render components, lights, and
  extraction into renderer scene packets.
- `engine/platform/windows/` provides the optional raw Win32 entry host.
- `editor/` contains the first cross-platform editor shell.
- `third_party/` centralizes external dependency setup for editor and future engine modules.
- `samples/` is reserved for examples and experiments.
- `tests/` is reserved for future high-level integration scenarios.

## Building

```bash
cmake -B build -S . -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release --parallel
ctest --test-dir build --output-on-failure
```

To build the editor shell:

```bash
cmake -B build -S . -DCMAKE_BUILD_TYPE=Release -DARC_BUILD_EDITOR=ON
cmake --build build --config Release --target arc_editor --parallel
```

To build it if needed and run it:

```bash
python run_editor.py
```

The editor runner enables the Vulkan renderer by default. Use
`python run_editor.py --no-vulkan-render` to build the temporary SDL-renderer
editor path instead.

## Public Includes

The current engine foundation can be consumed through module headers:

```cpp
#include <arc/simd/simd.h>
#include <arc/math/math.h>
#include <arc/geometric/geometric.h>
#include <arc/diagnostics/diagnostics.h>
#include <arc/jobs/jobs.h>
#include <arc/memory/memory.h>
#include <arc/framework/framework.h>
#include <arc/input/input.h>
#include <arc/render/render.h>
#include <arc/scene/scene.h>
```

More specific math and geometric headers are also available when a translation
unit only needs part of the API.

The renderer is still in bring-up mode. The current milestone has the
backend-neutral scene draw pipeline shape in place: render items, CPU frustum
culling, sorting, instancing batches, indirect draw command scaffolding, and
standard graph pass names for depth prepass, G-buffer, transparent, picking, and
selection outline. Vulkan currently consumes that packet through the existing
mesh rendering path while the real G-buffer/picking/outline backend work lands.
