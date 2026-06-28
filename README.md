# ARC

ARC is an early-stage 3D engine monorepo. The current codebase is focused on
the low-level foundation an engine needs before higher-level rendering,
scene, asset, and editor systems land.

## Current Shape

- `engine/` contains the core engine library.
- `engine/simd/` provides portable SIMD primitives and game-oriented SIMD helpers.
- `engine/math/` provides vector, matrix, and quaternion types built on top of SIMD where available.
- `engine/geometric/` provides simple 2D/3D primitives such as points, lines, boxes, circles, and spheres.
- `engine/diagnostics/` provides logging and diagnostic sinks.
- `engine/jobs/` provides the shared job/thread worker service.
- `engine/memory/` provides tracked memory-resource instrumentation.
- `engine/framework/` provides the platform-neutral application lifecycle.
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

## Public Includes

The current engine foundation can be consumed through module headers:

```cpp
#include <arc/simd.h>
#include <math/math.h>
#include <geometric/geometric.h>
#include <arc/diagnostics.h>
#include <arc/jobs.h>
#include <arc/memory.h>
#include <arc/framework.h>
```

More specific math and geometric headers are also available when a translation
unit only needs part of the API.
