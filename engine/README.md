# ARC Engine

The `engine` package contains the core ARC engine library. It is organized as a set of engine modules with explicit dependencies.

Current modules:

- `simd/` provides the SIMD foundation through the `arc-simd` target and `arc::simd` alias.
- `math/` provides the math foundation through the `arc-math` target and `arc::math` alias. It depends on `arc-simd`.
- `geometric/` provides 2D/3D primitive types through the `arc-geometric` target and `arc::geometric` alias. It depends on `arc-math`.
- `diagnostics/` provides logging through the `arc-diagnostics` target and `arc::diagnostics` alias.
- `jobs/` provides shared worker-pool services through the `arc-jobs` target and `arc::jobs` alias. It depends on `arc-diagnostics`.
- `memory/` provides tracked memory-resource instrumentation through the `arc-memory` target and `arc::memory` alias. It depends on `arc-diagnostics`.
- `framework/` provides the platform-neutral application lifecycle and runtime module manager through the `arc-framework` target and `arc::framework` alias. It depends on `arc-geometric`, `arc-jobs`, and `arc-memory`.
- `input/` provides runtime input bindings through the `arc-input` target and `arc::input` alias. It depends on `arc-framework`.
- `render/` provides the backend-neutral renderer foundation through the `arc-render` target and `arc::render` alias. It depends on `arc-framework`. It currently includes render events, render graph resources, renderer handles, material/resource scaffolding, scene draw packets, CPU culling/sorting/batching, and Vulkan backend bring-up.
- `scene/` provides ECS scene primitives and render extraction through the `arc-scene` target and `arc::scene` alias. It depends on `arc-render`. It currently includes transform/camera/mesh/light/editor components and extraction into renderer scene packets.
- `platform/windows/` provides the optional raw Win32 entry host through the `arc-platform-windows` target and `arc::platform-windows` alias. It depends on `arc`.
- `arc` remains the aggregate target for consumers that want the whole engine foundation.

## Include Policy

All public engine headers use the same shape:

```cpp
#include <arc/<module>/<header>.h>
```

The old root-style module includes such as `<arc/render.h>` and the older
math/geometric includes such as `<math/vector.h>` are no longer the preferred
layout.

The SIMD public include path remains:

```cpp
#include <arc/simd/simd.h>
```

The math public include paths are:

```cpp
#include <arc/math/vector.h>
#include <arc/math/matrix.h>
#include <arc/math/quaternion.h>
#include <arc/math/math.h>
```

The geometric public include paths are:

```cpp
#include <arc/geometric/point.h>
#include <arc/geometric/line.h>
#include <arc/geometric/box.h>
#include <arc/geometric/circle.h>
#include <arc/geometric/geometric.h>
```

The framework public include path is:

```cpp
#include <arc/framework/framework.h>
```

The diagnostics, jobs, and memory public include paths are:

```cpp
#include <arc/diagnostics/diagnostics.h>
#include <arc/jobs/jobs.h>
#include <arc/memory/memory.h>
```

The render public include path is:

```cpp
#include <arc/render/render.h>
```

The input and scene public include paths are:

```cpp
#include <arc/input/input.h>
#include <arc/scene/scene.h>
```

Lower-level render headers can be included directly when needed:

```cpp
#include <arc/render/render_graph.h>
#include <arc/render/render_world.h>
#include <arc/render/resources.h>
```

The SIMD module provides:

- Type-safe SIMD abstractions: `simd<T, N>` and `simd_mask<N>`
- Free-function based APIs
- Arithmetic, logical, comparison, mask, math, and game-oriented vector operations
- Backend support for x64, NEON, and RISC-V
- CMake integration through the `arc-simd` interface target

## Building And Testing

From the repository root:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release --parallel
ctest --test-dir build --output-on-failure
```

The SIMD tests can also be configured directly:

```bash
cmake -S engine/simd/tests -B build-simd-tests
cmake --build build-simd-tests --parallel
ctest --test-dir build-simd-tests --output-on-failure
```
