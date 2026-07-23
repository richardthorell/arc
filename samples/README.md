# ARC Samples

This folder contains sample applications and experiments.

The root CMake option `ARC_BUILD_SAMPLES` is present and defaults to `OFF`.

- `arc_boot_test_app` is a minimal Windows app that boots the ARC framework,
  opens a native window, and shuts down when the window is closed.
- `arc_headless` runs the same fixed-step framework without a window, renderer,
  Vulkan, SDL, or Electron. It supports deterministic finite runs for CI and
  clock-paced server runs.

Samples should include engine headers through the module layout, for example:

```cpp
#include <arc/framework/application.h>
```

Build samples from the repository root with:

```bash
cmake -B out/build/samples -S . -DARC_BUILD_SAMPLES=ON
cmake --build out/build/samples --config Release --target arc_boot_test_app
```

Run five deterministic server ticks without sleeping:

```bash
cmake --build out/build/samples --config Release --target arc_headless
out/build/samples/samples/Release/arc_headless --ticks 5 --seed 12345 --no-sleep
```
