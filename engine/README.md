# ARC Engine

The `engine` package contains the core ARC engine library. It is organized as a set of engine modules with explicit dependencies.

Current modules:

- `simd/` provides the SIMD foundation through the `arc-simd` target and `arc::simd` alias.
- `math/` provides the math foundation through the `arc-math` target and `arc::math` alias. It depends on `arc-simd`.
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
