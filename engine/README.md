# ARC Engine

The `engine` package contains the core ARC engine library. It is organized as a set of engine modules with explicit dependencies.

Current modules:

- `simd/` provides the SIMD foundation through the `arc-simd` target and `arc::simd` alias.
- `math/` provides the math foundation through the `arc-math` target and `arc::math` alias. It depends on `arc-simd`.
- `geometric/` provides 2D/3D primitive types through the `arc-geometric` target and `arc::geometric` alias. It depends on `arc-math`.
- `diagnostics/` provides logging through the `arc-diagnostics` target and `arc::diagnostics` alias.
- `jobs/` provides dependency-aware task scheduling, priorities, cancellation, work stealing, main/render/IO affinity, profiling, futures, and optional C++20 coroutine integration through the `arc-jobs` target and `arc::jobs` alias. It depends on `arc-memory` and `arc-diagnostics`.
- `memory/` provides domain/tag tracking, adaptive budgets, pressure recovery, leak snapshots, PMR adapters, arenas, world/component storage, packet pools, and streaming heaps through the `arc-memory` target and `arc::memory` alias. It depends on `arc-diagnostics`.
- `ecs/` provides stable entity identities, paged component storage, prepared queries, structural command buffers, system scheduling, reflection, hierarchy, templates, prefabs, replication visitors, and world-region contracts through the `arc-ecs` target and `arc::ecs` alias. It depends on `arc-jobs` and `arc-memory`.
- `io/` provides backend-neutral asynchronous file reads, ranged reads, writes, atomic writes, metadata queries, and cancellation through the `arc-io` target and `arc::io` alias. Its portable implementation runs chunked blocking operations on scheduler-owned IO workers.
- `framework/` provides the platform-neutral application lifecycle and runtime module manager through the `arc-framework` target and `arc::framework` alias. It depends on `arc-geometric`, `arc-jobs`, and `arc-memory`.
- `input/` provides runtime input bindings through the `arc-input` target and `arc::input` alias. It depends on `arc-framework`.
- `render/` provides the backend-neutral renderer foundation through the `arc-render` target and `arc::render` alias. It depends on `arc-framework`. It currently includes render events, render graph resources, renderer handles, material/resource scaffolding, scene draw packets, CPU culling/sorting/batching, and Vulkan backend bring-up.
- `scene/` provides game/render components and render extraction through the `arc-scene` target and `arc::scene` alias. It depends on `arc-ecs` and `arc-render`; existing scene registry/entity APIs remain compatibility aliases over the generic ECS.
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

The diagnostics, jobs, memory, and IO public include paths are:

```cpp
#include <arc/diagnostics/diagnostics.h>
#include <arc/jobs/jobs.h>
#include <arc/memory/memory.h>
#include <arc/io/io.h>
```

Job coroutine integration is enabled by default and has no external runtime
dependency. Configure with `-DARC_ENABLE_JOB_COROUTINES=OFF`, or use the
`default-no-coroutines` configure/build/test presets, to validate the
non-coroutine contract.

The render public include path is:

```cpp
#include <arc/render/render.h>
```

The ECS, input, and scene public include paths are:

```cpp
#include <arc/ecs/ecs.h>
#include <arc/input/input.h>
#include <arc/scene/scene.h>
```

Lower-level render headers can be included directly when needed:

```cpp
#include <arc/render/render_graph.h>
#include <arc/render/render_world.h>
#include <arc/render/resources.h>
```

The raster renderer resolves immutable adapter capabilities into a separate
quality/path configuration. `low` currently selects the direct forward
fallback with 32 point/spot lights and 1024-pixel directional shadows;
`medium` selects the deferred path. Render graphs use typed formats and strong
resource handles, validate access hazards, and report transitions, lifetimes,
and transient aliases to the editor.

The active Vulkan implementation targets Vulkan 1.2 and queries optional
features individually. Dynamic rendering is still required by the current
Vulkan raster-pass implementation; a legacy `VkRenderPass` executor remains a
follow-up before Vulkan 1.2 devices without `VK_KHR_dynamic_rendering` are
supported.

The SIMD module provides:

- Type-safe SIMD abstractions: `simd<T, N>` and `simd_mask<N>`
- Free-function based APIs
- Arithmetic, logical, comparison, mask, math, and game-oriented vector operations
- Backend support for x64, NEON, and RISC-V
- CMake integration through the `arc-simd` interface target

## Building And Testing

From the repository root:

```bash
cmake --preset default
cmake --build --preset default --parallel
ctest --preset default
```

The SIMD tests can also be configured directly:

```bash
cmake -S engine/simd/tests -B out/build/simd-tests
cmake --build out/build/simd-tests --parallel
ctest --test-dir out/build/simd-tests --output-on-failure
```
