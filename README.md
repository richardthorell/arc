# ARC

ARC is growing into a 3D engine monorepo.

The repository is organized around a few top-level areas:

- `engine/` contains the core engine library, split into modules such as `simd/`, `math/`, and `geometric/`.
- `editor/` is reserved for future editor tooling.
- `samples/` is reserved for example applications and experiments.
- `tests/` is reserved for high-level engine/editor/sample integration tests.

## Building

```bash
cmake -B build -S . -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release --parallel
ctest --test-dir build --output-on-failure
```

The current SIMD include path remains compatible:

```cpp
#include <arc/simd.h>
```
