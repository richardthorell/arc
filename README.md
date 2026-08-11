# arc

| Target | Status |
| --- | --- |
| Clang | [![Build: Clang](https://github.com/richardthorell/arc/actions/workflows/build-clang.yml/badge.svg?branch=main)](https://github.com/richardthorell/arc/actions/workflows/build-clang.yml) |
| GCC | [![Build: GCC](https://github.com/richardthorell/arc/actions/workflows/build-gcc.yml/badge.svg?branch=main)](https://github.com/richardthorell/arc/actions/workflows/build-gcc.yml) |
| MSVC | [![Build: MSVC](https://github.com/richardthorell/arc/actions/workflows/build-msvc.yml/badge.svg?branch=main)](https://github.com/richardthorell/arc/actions/workflows/build-msvc.yml) |
| Documentation | [![Doxygen Docs Data](https://github.com/richardthorell/arc/actions/workflows/doxygen-xml.yml/badge.svg?branch=main)](https://github.com/richardthorell/arc/actions/workflows/doxygen-xml.yml) |

**arc** is a modern C++ 3D game engine focused on performance, clean systems architecture, and editor-driven workflows.

The engine is built around modular runtime systems, scene and rendering architecture, editor-first workflows, and source-driven tooling. The goal is to provide a compact but capable engine core that supports real-time rendering, scene editing, runtime experimentation, and future game/editor production workflows.

## Overview

arc is organized as a modular engine and editor stack:

- **Engine core** — application framework, diagnostics, jobs, memory tracking, input, and platform-neutral runtime services.
- **ECS and scene system** — stable entities, paged component storage, cached queries, deterministic structural commands, parallel system scheduling, reflection, hierarchy, prefabs, transforms, lights, cameras, and render extraction.
- **Renderer** — backend-neutral rendering interfaces, render graph concepts, resource handles, scene draw packets, and Vulkan-oriented rendering architecture.
- **Asset pipeline** — foundation for loading, managing, and preparing engine resources such as meshes, materials, textures, and shaders.
- **Editor** — Electron/React authoring environment backed by a native C++ scene and rendering host.
- **Tooling** — generated API documentation, automated CI builds, and source-driven documentation data for the website.

## Rendering

The rendering layer is designed around explicit rendering architecture rather than a monolithic renderer.

Current and planned renderer concepts include:

- Backend-neutral render interfaces
- Vulkan renderer backend
- Render graph structure
- Renderer-owned resource handles
- Scene extraction into render packets
- CPU frustum culling
- Sorting and batching
- Instancing-ready draw data
- Indirect draw command scaffolding
- Standard pass structure for depth, geometry, transparency, picking, and selection outline workflows

The renderer is intended to support both runtime rendering and editor viewport rendering.

## Editor

arc includes an editor shell intended to become the main workflow surface for building and inspecting scenes.

The editor combines a reusable Electron/React workbench with a native C++ host:

- Electron owns the docked workbench, hierarchy, inspectors, asset tools, and document UX.
- The native host owns authoritative scene state, history, persistence, viewport input, and rendering.
- The viewport is an engine-rendered native surface embedded in the Electron workbench.
- Host protocol contracts keep editor UI concerns separate from engine and renderer internals.

The editor is part of the engine workflow rather than a separate application layer bolted on afterward.

## Documentation

API documentation is generated from the source tree using Doxygen.

The documentation pipeline generates Doxygen XML, converts it into static JSON, and publishes it through GitHub Pages for use by the arc website documentation viewer.

Documentation data is published at:

```text
https://richardthorell.github.io/arc/api/index.json
```

The generated documentation is split into static JSON files so the website can load an index first, then lazy-load details for classes, structs, namespaces, files, and members.

## Building

Configure and build:

```bash
cmake --preset default
cmake --build --preset default --parallel
```

Run tests:

```bash
ctest --preset default
```

Build the editor:

```bash
cmake --preset editor-vulkan
cmake --build --preset editor-vulkan --target arc_host_process --parallel
```

Build and run the editor using the helper script:

```bash
python run_editor.py
```

The editor runner builds the native Vulkan host and launches the Electron
workbench. To bypass the project browser while developing the editor, create or
reuse a persistent Blank 3D project under `out/`:

```bash
python run_editor.py --quick-start
```

The Electron executable also accepts `--quick-start` directly. Passing a
specific `.arcproject` path continues to open that project directly.

To build without the Vulkan viewport backend:

```bash
python run_editor.py --no-vulkan-render
```

CMake presets write generated files under `out/build/...` so the repository root stays clear of configuration-specific build folders.

ARC has three supported product configurations:

- `Debug` enables assertions and developer diagnostics without optimization.
- `RelWithDebInfo` enables optimization, symbols, and development tooling.
- `Shipping` enables LTO and runtime-only features while disabling tests, the
  editor, source monitoring, and source import/cooking.

The matching `debug`, `relwithdebinfo`, and `shipping` configure/build presets
work for both single- and multi-configuration generators. Clang sanitizer
presets are also available:

```bash
cmake --preset sanitize-address-undefined
cmake --build --preset sanitize-address-undefined --parallel
ctest --preset sanitize-address-undefined

cmake --preset sanitize-thread
cmake --build --preset sanitize-thread --parallel
ctest --preset sanitize-thread
```

Electron dependencies are immutable. Use `npm ci` rather than `npm install`:

```bash
cd editor
npm ci
npm run typecheck
npm run lint
npm run format:check
npm test
```

Validate the dependency policy and every discovered built-in shader with:

```bash
python3 tools/check_dependencies.py
cmake --build --preset render-vulkan --target arc-shaders-check
```

CPU-side regression benchmarks are dependency-free and calibrated against a
stable integer workload:

```bash
cmake -S . -B out/build/benchmarks -DARC_BUILD_BENCHMARKS=ON -DCMAKE_BUILD_TYPE=RelWithDebInfo
cmake --build out/build/benchmarks --target arc-benchmarks
cmake --build out/build/benchmarks --target arc-benchmarks-check
```

Generate an optional Clang/LLVM coverage report (requires Ninja, Clang, and LLVM coverage tools):

```bash
cmake --preset coverage-clang
cmake --build --preset coverage-clang --parallel
python3 tools/generate_coverage.py
```

Reports are written under `out/coverage`; coverage CI publishes them as an artifact without enforcing a percentage threshold.

## Installed SDK and external projects

ARC projects are standalone repositories described by a version-2
`<Project>.arcproject` file. Generate one of the installed templates with the
native project tool:

```bash
arc-project create --name MyGame --destination MyGame \
  --template blank-3d --templates <ARC>/share/arc/templates --engine 0.1.0
arc-project validate --project MyGame/MyGame.arcproject --require-paths
arc-project configure --project MyGame/MyGame.arcproject --sdk <ARC>
arc-project build --project MyGame/MyGame.arcproject --config RelWithDebInfo
```

Available templates are Blank 3D, Blank Headless, Rendering Sample, and Empty
C++.

Project Runtime, Editor, and Server modules use ARC's generated reflection and
generation-based native reload workflow. See
[C++ project modules](docs/project-modules.md) for the stable ABI, annotations,
schema compatibility rules, and editor Build workflow.
Generated projects keep source-controlled files in `Source/`, `Content/`,
`Config/`, and `Plugins/`; transient editor state, caches, recovery generations,
and products live in `Saved/`, `Intermediate/`, and `Build/`.

External CMake projects consume the relocatable SDK without referencing the ARC
checkout:

```cmake
find_package(ARC 0.1.0 EXACT CONFIG REQUIRED COMPONENTS Runtime Vulkan)
target_link_libraries(MyGame PRIVATE ARC::Runtime ARC::RenderVulkan)
```

Engine installations are discovered through the per-user installation registry,
whose entries point to an installed `arc-installation.json` manifest:

```bash
arc-project engine register --manifest <ARC>/arc-installation.json
arc-project engine list
arc-project engine verify
arc-project toolchains
```

The same CLI owns descriptor upgrades, toolchain probing, CMake configure/build,
and Visual Studio, VS Code, and CLion generation so the start screen and command
line use identical validation and template behavior.

## CI

arc is built continuously across pinned runner and compiler families:

- Clang 18 on Ubuntu 24.04, including ASan/UBSan and TSan jobs
- GCC 14 on Ubuntu 24.04 with Vulkan compile coverage
- MSVC on Windows Server 2022 with the native editor, Vulkan, Electron package,
  clean launch smokes, and Shipping artifacts

CI also compiles all shaders with Vulkan SDK 1.4.350.0, runs static analysis and
format gates, and compares deterministic subsystem benchmarks against calibrated
normalized baselines.

Each compiler has its own workflow so build status can be tracked independently.
