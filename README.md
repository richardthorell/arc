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
- **Scene system** — scene representation, entities, components, transforms, lights, cameras, and scene extraction for rendering.
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
workbench. To build without the Vulkan viewport backend:

```bash
python run_editor.py --no-vulkan-render
```

CMake presets write generated files under `out/build/...` so the repository root stays clear of configuration-specific build folders.

Generate an optional Clang/LLVM coverage report (requires Ninja, Clang, and LLVM coverage tools):

```bash
cmake --preset coverage-clang
cmake --build --preset coverage-clang --parallel
python3 tools/generate_coverage.py
```

Reports are written under `out/coverage`; coverage CI publishes them as an artifact without enforcing a percentage threshold.

## CI

arc is built continuously across multiple compiler toolchains:

- Clang on Ubuntu
- GCC on Ubuntu
- MSVC on Windows

Each compiler has its own workflow so build status can be tracked independently.
