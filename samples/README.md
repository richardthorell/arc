# ARC Samples

This folder contains sample applications and experiments.

The root CMake option `ARC_BUILD_SAMPLES` is present and defaults to `OFF`.

- `arc_boot_test_app` is a minimal Windows app that boots the ARC framework,
  opens a native window, and shuts down when the window is closed.

Build samples from the repository root with:

```bash
cmake -B build -S . -DARC_BUILD_SAMPLES=ON
cmake --build build --config Release --target arc_boot_test_app
```
