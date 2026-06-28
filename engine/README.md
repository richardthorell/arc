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
- `platform/windows/` provides the optional raw Win32 entry host through the `arc-platform-windows` target and `arc::platform-windows` alias. It depends on `arc`.
- `arc` remains the aggregate compatibility target for consumers that want the whole engine foundation.

The SIMD public include path remains:

```cpp
#include <arc/simd.h>
```

The math public include paths are:

```cpp
#include <math/vector.h>
#include <math/matrix.h>
#include <math/quaternion.h>
#include <math/math.h>
```

The geometric public include paths are:

```cpp
#include <geometric/point.h>
#include <geometric/line.h>
#include <geometric/box.h>
#include <geometric/circle.h>
#include <geometric/geometric.h>
```

The framework public include path is:

```cpp
#include <arc/framework.h>
```

The diagnostics, jobs, and memory public include paths are:

```cpp
#include <arc/diagnostics.h>
#include <arc/jobs.h>
#include <arc/memory.h>
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
