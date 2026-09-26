# Shared GPU Editor Viewport Presentation

Status: design contract for #350

## Goal

Replace the editor's native child-window viewport presentation with an engine-rendered offscreen surface that Electron can compose with normal React UI, without making the editor or backend-neutral renderer depend on Vulkan objects.

The migration must be incremental: the existing native viewport remains a supported fallback until the shared-GPU path is proven on each platform/backend combination.

## Design principles

- The engine owns rendering, render-target lifetime, frame production, and GPU synchronization.
- The editor owns layout, input routing, overlays, visibility, and presentation scheduling.
- Backend-specific export/import handles stay behind a narrow presentation bridge.
- A frame is immutable after publication. Reuse requires explicit release from the consumer.
- Resize, DPI changes, visibility, surface loss, and device loss are state transitions rather than ad-hoc resource recreation.
- No renderer-facing engine API exposes Electron, Chromium, Vulkan, Win32, Direct3D, or Metal types.

## Architecture

```text
Scene / editor camera
        |
        v
Backend-neutral renderer
        |
        v
Editor offscreen target + frame pool
        |
        v
Backend presentation exporter
        |
   shared GPU image
   + sync primitive
        |
        v
Native editor presentation bridge
        |
        v
Electron/Chromium compositor
        |
   React overlays / chrome
```

### Backend-neutral contract

The renderer should expose an editor presentation abstraction with opaque, strongly typed IDs rather than native handles. Conceptually:

```cpp
struct editor_surface_id;
struct editor_frame_id;

struct editor_surface_desc {
    uint32_t width;
    uint32_t height;
    float dpi_scale;
    texture_format preferred_format;
};

struct editor_frame {
    editor_frame_id id;
    uint64_t generation;
    uint32_t width;
    uint32_t height;
};
```

The exact API should follow existing renderer handle conventions when implemented. The backend-specific bridge resolves an `editor_frame_id` into a platform/backend export descriptor; that descriptor must not escape into scene extraction, render graph scheduling, or general renderer APIs.

### Frame pool

Each viewport owns a small bounded pool, initially three frames. Frames move through explicit states:

```text
Available -> Rendering -> Ready -> Presented -> Available
                 |                    |
                 +---- discarded -----+
```

The producer never overwrites `Ready` or `Presented` frames. If Electron is late, the renderer may drop an older unpublished frame rather than blocking the render thread. Pool size is a policy knob, not part of frame identity.

Each frame carries a monotonically increasing surface generation. Consumers must reject frames from an old generation after resize, DPI change, surface recreation, or device recovery.

## Synchronization boundary

Publication is the only ownership handoff.

1. Renderer acquires an Available frame.
2. Render graph writes the viewport image and transitions it to the backend's exportable/readable state.
3. Backend signals a GPU synchronization primitive and publishes the frame plus opaque export metadata.
4. The compositor waits on/imports that synchronization through the native bridge.
5. After Chromium no longer needs the image, the bridge releases the frame to the pool.

No CPU `waitIdle` belongs in the normal path. Backends should use timeline/fence/semaphore equivalents internally. The neutral contract expresses readiness and release, not the synchronization primitive type.

## Export/import boundary

A backend adapter owns native sharing details. Examples include Vulkan external-memory/semaphore handles on Windows, D3D12 shared resources/fences, and Metal shared surfaces/events where supported. These are implementation choices, not editor API concepts.

The bridge reports capabilities before creating a shared surface:

- GPU sharing supported
- compatible pixel formats
- synchronization mode
- cross-process support, if required by the Electron integration
- HDR/color-space support

If the required sharing path is unavailable, ARC uses the native viewport fallback. A future CPU-copy fallback may exist for diagnostics, but it should not become the normal architecture.

## Resize, DPI, and visibility lifecycle

A viewport surface has these logical states:

```text
Inactive -> Creating -> Active -> Recreating -> Active
                          |             |
                          v             v
                       Hidden        Failed
                          |             |
                          +------> Fallback
```

Resize and DPI changes create a new surface generation. Existing frames from the previous generation may drain, but cannot be presented after the new generation becomes active. Zero-sized/hidden viewports stop frame acquisition instead of allocating 0x0 resources.

Rapid resize should coalesce requests and recreate from the latest requested dimensions. React layout coordinates remain CSS-space; the native bridge converts to physical pixel dimensions using the effective DPI scale.

## Device loss and recovery

Device loss invalidates every exported resource and synchronization object from that backend generation.

Recovery order:

1. Stop frame acquisition/publication.
2. Notify the editor bridge that the current surface generation is invalid.
3. Release/import-side references that can be released safely.
4. Recover/recreate the backend device through the renderer's normal device-recovery path.
5. Recreate editor surface pools.
6. Publish a new generation only after a complete frame succeeds.

Electron must never retain or attempt to present a frame from the lost device generation. Failure to recreate the shared path switches that viewport to the existing native fallback and surfaces a diagnostic instead of requiring an editor restart.

## Input and overlay composition

Electron remains authoritative for viewport bounds and user interaction. Pointer/keyboard input is translated from the composed viewport's CSS coordinates into normalized or physical viewport coordinates before being sent through the existing editor input/camera path.

React owns editor-only overlays such as selection labels, gizmo chrome, floating navigation, diagnostics, menus, drag/drop targets, and loading/error states. Engine-rendered scene pixels stay in the shared GPU surface. This allows normal Electron z-order and removes native child-window clipping/composition constraints.

## Color and presentation

The shared surface contract records format and color-space metadata. The renderer produces a presentation-ready editor image according to the selected editor output transform; Chromium must not silently reinterpret linear data as sRGB or apply an additional tone map. HDR support is capability-gated and can initially fall back to the SDR editor output path.

## Incremental migration

### Phase 1 — renderer surface contract

- Introduce opaque editor surface/frame identities.
- Add bounded offscreen frame-pool ownership in backend-neutral rendering code.
- Keep the existing native child-window presentation unchanged.

### Phase 2 — Vulkan/Windows sharing prototype

- Implement exportable offscreen images and synchronization in the Vulkan backend.
- Add native bridge import/presentation behind a feature flag.
- Validate resize, DPI, hide/show, frame dropping, and shutdown.

### Phase 3 — Electron composition

- Compose shared frames inside the normal Electron viewport region.
- Move editor overlays above the composed surface.
- Route input using Electron-owned bounds.
- Retain one-click fallback to native presentation.

### Phase 4 — hardening and backend expansion

- Integrate renderer device-loss recovery.
- Add diagnostics for frame age, pool pressure, dropped frames, and sharing failures.
- Validate D3D12/Metal adapters through the same neutral contract as those backends arrive.
- Remove the native child-window path only after all supported desktop targets have a reliable replacement.

## Diagnostics

At minimum expose per viewport:

- active presentation path (`shared-gpu` or `native-fallback`)
- surface generation and physical dimensions
- pool size and frame states
- last produced/presented frame IDs
- dropped/stale frame count
- producer and presentation latency
- last sharing/import/recovery error

Diagnostics belong to the editor tooling layer; backend-specific details can be included as optional structured fields without becoming renderer-wide concepts.

## Validation plan

Automated deterministic tests should cover frame-pool state transitions, stale-generation rejection, resize coalescing, hidden/zero-size behavior, and recovery state transitions without requiring a GPU. Backend smoke tests should cover export/import synchronization and resource lifetime on supported hardware.

Manual editor validation should include rapid resize, DPI/display movement, tab hide/show, docking/undocking, sustained camera motion under compositor stalls, editor shutdown with frames in flight, and forced surface/device-recovery hooks.

## Relationship to rendering implementation

#399 implements the rendering half of this architecture and should use this document as its contract. It must keep native sharing objects inside backend/presentation implementation code and preserve the native fallback during migration.

## Non-goals

- Redesigning scene extraction or render graph semantics.
- Making Electron responsible for engine render-target allocation.
- Standardizing native GPU handles across APIs.
- Requiring every backend/platform to support zero-copy sharing before it can render ARC.
- Removing the current viewport path in the first implementation milestone.
