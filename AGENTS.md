# AGENTS.md

Guidance for AI coding agents working in this repository.

## Project Overview

ARC is an early-stage C++ 3D engine monorepo with a long-term goal of becoming a fully functional AAA game engine. The project is not there today, but every milestone should deliberately move it closer to being the best engine ever created: fast, scalable, visually exceptional, pleasant to build with, and strong enough for ambitious real-time games.

The current project focus is the graphical foundation and getting the editor up and running. Prioritize work that advances the renderer, Vulkan path, scene-to-renderer data flow, editor viewport, and editor usability while keeping the engine architecture clean enough to grow into a complete production engine.

The codebase is organized around a reusable engine library, a cross-platform editor shell, third-party dependency setup, samples, and tests.

## Repository Layout

- `engine/` contains the core engine library.
- `engine/simd/` contains portable SIMD primitives and game-oriented SIMD helpers.
- `engine/math/` contains vectors, matrices, quaternions, and math utilities built on top of SIMD where appropriate.
- `engine/geometric/` contains 2D/3D primitives such as points, lines, boxes, circles, and spheres.
- `engine/diagnostics/` contains logging and diagnostic sinks.
- `engine/jobs/` contains shared job/thread worker services.
- `engine/memory/` contains tracked memory-resource instrumentation.
- `engine/framework/` contains platform-neutral application lifecycle code.
- `engine/input/` contains runtime keyboard, mouse, and future gamepad bindings.
- `engine/render/` contains backend-neutral rendering code, the Vulkan backend, render graph code, renderer handles, and scene draw packet preparation.
- `engine/scene/` contains ECS scene primitives, render components, lights, and extraction into renderer scene packets.
- `engine/platform/windows/` contains the optional raw Win32 entry host.
- `editor/` contains the first cross-platform editor shell.
- `third_party/` centralizes external dependency setup.
- `samples/` is reserved for examples and experiments.
- `tests/` is reserved for future high-level integration scenarios.

## Build and Test Commands

Use CMake from the repository root.

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

To build and run the editor through the helper script:

```bash
python run_editor.py
```

Use this to run the temporary SDL-renderer editor path instead of the Vulkan renderer:

```bash
python run_editor.py --no-vulkan-render
```

## Public Include Convention

Public engine headers must follow this include convention:

```cpp
#include <arc/<module>/<header>.h>
```

Examples:

```cpp
#include <arc/simd/simd.h>
#include <arc/math/math.h>
#include <arc/geometric/geometric.h>
#include <arc/diagnostics/diagnostics.h>
#include <arc/jobs/jobs.h>
#include <arc/memory/memory.h>
#include <arc/framework/framework.h>
#include <arc/input/input.h>
#include <arc/render/render.h>
#include <arc/scene/scene.h>
```

Do not introduce new public include styles unless the project explicitly changes this convention.

## C++ Guidelines

- Prefer modern C++ with clear ownership, explicit lifetimes, and minimal hidden allocation.
- Keep engine APIs small, predictable, and suitable for game/runtime use.
- Avoid unnecessary virtual dispatch, heap allocation, exceptions in hot paths, and global mutable state.
- Prefer strongly typed handles/IDs for renderer, scene, and asset-facing APIs.
- Use `const` correctness and pass by value/reference intentionally.
- Keep headers lightweight. Avoid dragging expensive platform, Vulkan, SDL, or editor dependencies into public engine headers.
- Put platform/backend-specific details behind narrow interfaces or implementation files.
- Preserve the existing namespace, file naming, and module boundaries when adding code.

## Rendering and Engine Direction

ARC is moving toward a backend-neutral renderer with Vulkan as the active backend. The renderer currently includes the shape of a scene draw pipeline: render items, CPU frustum culling, sorting, instancing batches, indirect draw command scaffolding, and standard graph pass names for depth prepass, G-buffer, transparent, picking, and selection outline.

When changing rendering code:

- Keep backend-neutral concepts separate from Vulkan-specific implementation details.
- Avoid coupling scene extraction directly to Vulkan objects.
- Preserve room for multiple render paths, including editor viewport rendering and future high-quality/realistic rendering features.
- Prefer explicit resource lifetime and synchronization boundaries.
- Keep render graph pass names and packet/data-flow concepts consistent unless intentionally redesigning them.

## Editor Direction

The editor uses an ImGui/SDL-style shell with the engine rendering into a viewport panel. When changing editor code:

- Keep editor UI code separate from engine runtime systems where practical.
- Do not make engine modules depend on editor-only concepts.
- Treat the viewport as an engine-rendered surface embedded in tooling UI.
- Prefer simple, inspectable UI state over complex framework abstractions.

## Dependency Guidelines

- Keep third-party integrations centralized through `third_party/` and CMake options.
- Do not add new dependencies without a clear reason and a small integration surface.
- Avoid dependencies that force broad architectural choices on the engine.
- Prefer libraries that are portable, permissively licensed, actively maintained, and usable from C++.

## Testing Expectations

- Run the standard CMake build and `ctest` after meaningful code changes when possible.
- Add or update tests for math, SIMD, geometry, memory, jobs, and other deterministic systems.
- Rendering/editor changes may need smoke testing through `arc_editor` or `run_editor.py` until more automated coverage exists.
- Keep tests deterministic and avoid requiring a specific GPU unless the test is explicitly a graphics smoke test.

## Change Management

- Keep changes scoped and module-focused.
- Avoid large drive-by rewrites unless the task explicitly asks for refactoring.
- Update `README.md` or nearby documentation when changing build commands, public includes, module boundaries, or major behavior.
- Prefer small PRs with clear summaries and test notes.
- Call out any untested areas, platform assumptions, or GPU/backend assumptions in the PR description.

## Style Notes for Generated Code

- Match the surrounding code style first.
- Prefer clear names over abbreviations unless the existing module uses a known abbreviation.
- Avoid clever template/meta-programming unless it materially simplifies the engine API.
- Keep comments focused on intent, invariants, synchronization, ownership, and non-obvious platform/backend behavior.
- Do not add license headers or file banners unless the repository already requires them for that area.

## Safety Rules for Agents

- Do not commit secrets, API keys, credentials, generated build folders, local IDE state, or machine-specific files.
- Do not vendor large binary assets or SDKs unless explicitly requested.
- Do not remove existing modules, public headers, or build options without checking how they are referenced.
- Do not silently change public API names or include paths.
- Do not assume Windows-only behavior unless working inside the Windows platform module.

## Useful Workflow

1. Inspect the relevant module and its CMake setup before editing.
2. Make the smallest change that satisfies the task.
3. Build the affected target when possible.
4. Run tests when available.
5. Summarize what changed, what was tested, and any follow-up work needed.
