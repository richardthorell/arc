# Android native host (M1 / A1–A2)

This is the **minimal Android platform host**, intentionally independent of the
ARC runtime bootstrap (#1094), renderer (#402), and distribution finalizer (#403).
The host starts a native `android.app.NativeActivity` and processes lifecycle and
native-window events through the NDK `native_app_glue` loop. It does **not** render a frame or load content.

## Prerequisites

- Android SDK with API 35 platforms, Android Build Tools, and CMake 3.22.1
- Android NDK installed through Android SDK Manager
- JDK 17 and Gradle 8.9 compatible with Android Gradle Plugin 8.7.3
- An Android device/emulator supporting **arm64-v8a** and Android 8.0+ (API 26+)

The checkout does not vendor Android SDK/NDK, Gradle binaries, or a Gradle
wrapper JAR. Point Gradle at the SDK using `ANDROID_HOME` or an untracked
`local.properties` file with `sdk.dir=...`.

## Build and launch

Run from `engine/platform/android`:

```sh
gradle :app:assembleDebug
adb install -r app/build/outputs/apk/debug/app-debug.apk
adb shell am start -n org.arcengine.host/android.app.NativeActivity
adb logcat -s ARC:I '*:S'
```

For a native-only configure diagnostic (using an installed NDK toolchain):

```sh
cmake -S app/src/main/cpp -B build/android-native \
  -DCMAKE_TOOLCHAIN_FILE="$ANDROID_HOME/ndk/<version>/build/cmake/android.toolchain.cmake" \
  -DANDROID_ABI=arm64-v8a -DANDROID_PLATFORM=android-26
```

The host uses ARC's existing `arc::diagnostics` logger with an Android Logcat
sink. Severity and category are preserved; the sink maps trace/debug/info/warn/
error/fatal onto Android priorities. This isolated fixture compiles the existing
engine diagnostics source; the integrated A3 host will link `arc-diagnostics`.

The native library is named `libarc_android_host.so`, and its
`android_main` is exported through the NDK NativeActivity glue. CMake
explicitly rejects non-Android toolchains rather than pretending a desktop
build is an Android APK.

## A2 lifecycle behavior

The NativeActivity callback maps START/STOP, RESUME/PAUSE, focus, native-window
attach/detach, window resize, content rectangle and configuration updates,
low-memory notifications and destroy into a host-owned `lifecycle_state`.
The window is **borrowed** from the Android glue. The host does not retain or
release it after `APP_CMD_TERM_WINDOW`. A new surface increases
`window_generation` while the process and common runtime are preserved.
Surface resize and orientation changes do not increment the generation.
The host observes app-private writable storage and the APK asset manager
without incorrectly treating packaged assets as ordinary files.

The transition model deliberately contains no Android SDK or rendering headers.
It has standalone host-buildable unit tests:

```sh
cmake -S tests -B build/lifecycle-tests
cmake --build build/lifecycle-tests
ctest --test-dir build/lifecycle-tests --output-on-failure
```

This A2 checkpoint establishes the lifecycle state contract only; hooking
the renderer and shared runtime to those transitions happens in A3/#402.

## Planned follow-ups

- **A2 remaining (#401):** Input handling and richer storage/asset path services.
- **A3 (#401/#1094):** Invoke the shared ARC runtime launch context from the
  Android host without introducing Android/JNI dependencies into shared code.
- **#402:** Android Vulkan surface/swapchain and actual scene rendering.
- **#403:** Project template/CLI integration, artifact generation, install,
  launch, and logcat diagnostics.

This isolated Gradle application is a build-system smoke fixture, not the
final Android project generator or a replacement for the shared distribution
manifest/cooker.

## A3 — APK VFS and packaged runtime bootstrap

The APK asset source reads distribution files using Android's `AAssetManager`.
`android_apk_provider` now adapts that reader to the **existing**
`arc::io::virtual_file_provider` contract (metadata, ranged asynchronous reads,
cancellation, and structured IO failures). The NativeActivity owns an
`android_runtime_bridge` for its process lifetime, which:

1. Creates ARC's job scheduler and virtual filesystem.
2. Mounts the APK provider at `package://runtime`.
3. Supplies Android platform capabilities, absolute app-private storage,
   arm64/Vulkan target expectations, and a mounted-VFS root to
   `arc::project::discover_runtime_boot`.
4. Holds the resulting validated boot context across pause/resume and native
   surface loss, then unmounts and drains the scheduler on process shutdown.

The Gradle CMake project now links actual ARC modules through the engine's
module CMake definitions: `arc-project`, `arc-io`, `arc-jobs` and their
required dependencies. Third-party content/JSON dependencies are fetched
using the existing ARC CMake setup. ARC version expectations are currently
pinned to 0.1.0 for the fixture, with Android `Debug` and
`Shipping` (Gradle release) package configuration identities.

**Packaged asset layout:** put `arc-distribution.json`, runtime config,
cooked manifests and cooked chunks in `app/src/main/assets/` using their
relative D0 file paths. The asset provider does not expose system-native
libraries stored in `lib/arm64-v8a/`; D1's startup integrity mode verifies
the runtime config and cooked manifests from the APK. The platform finalizer
(#403) will automate this layout and package/module metadata generation.

An APK with no staged D0 distribution now produces a clear structured boot
error in Logcat rather than falsely claiming that a game has launched.
With a valid staged manifest/content, D1 checks the manifest schema, target,
engine/SDK versions, and boot-file hashes. **This does not yet activate a
project game module, load a startup scene, or render with Vulkan.** Those
integrations belong to #401 follow-ups / #402 / #403.

For development, run `gradle :app:assembleDebug` and inspect
`adb logcat -s ARC:I '*:S'`. Run portable path/lifecycle checks using the
`tests/` CMake harness. Android SDK/NDK build and on-device boot validation
remain necessary.

## A4 — Android cooked catalog activation

After `discover_runtime_boot` verifies the staged D0 manifest, the Android
runtime bridge now calls `bootstrap_runtime_catalog` with the already-mounted
APK VFS. This validates target-specific cooked manifests, required startup
bundles and startup scene GUIDs, and mounts cooked package metadata without
making all asset bytes resident. A failed mount returns a structured ARC
diagnostic; the host does not claim success. The catalog session is destroyed
before the validated boot context, APK mount and job scheduler.

The cook target platform now includes Android. The cooker reads and writes
`platform: "android"` rather than silently converting it to Windows; the
catalog requires an actual Android cooked target before accepting the APK.
A new cook-manifest serialization regression test covers Android/arm64/ASTC.
The generic C6 quality/variant-selection policy remains in #994.

**Remaining:** activate the project game module/world and connect native
surface + Vulkan rendering (#402). Automated Android content generation,
APK packaging and device testing remain #403. This PR does not claim that
an Android app renders a scene.

## Vulkan V1 — Android WSI surface adapter (#402)

The native host now includes the Android Vulkan WSI surface callback needed by
`arc::render::vulkan::vulkan_backend_config::create_surface`. Pass the
`create_android_vulkan_surface` function, the current borrowed
`vulkan_native_window` as `surface_user_data`, and request
`VK_KHR_ANDROID_SURFACE_EXTENSION_NAME` along with the common
`VK_KHR_SURFACE_EXTENSION_NAME` instance extension. The callback resolves
`vkCreateAndroidSurfaceKHR` through ARC's existing instance procedure
resolver. The Vulkan backend owns the returned surface and its destruction.

`active_vulkan_window` returns a binding only when the Activity is started,
resumed, focused and has a nonzero native surface extent. Each attach receives
a new window generation; resizing keeps the same generation. A new generation
means the renderer must tear down its previous presentation resources before
creating a new surface. Window pointers must never outlive
`APP_CMD_TERM_WINDOW`.

This is the **V1 WSI adapter**, not the full renderer. V2 connects the ARC
renderer to the Android host and handles swapchain/device recreation, then
V3 adds a representative scene, mobile capability fallbacks and device tests.
Run the portable lifecycle contract tests through the Android `tests/` CMake
harness; the WSI callback itself requires an Android NDK build and real device.

## Vulkan V2 — ARC renderer on the Android native window

The Android host now links the existing `arc-render-vulkan` backend and
creates it with `VK_KHR_surface` and `VK_KHR_android_surface` using the V1
callback. This is the **shared ARC renderer**, not a separate Android renderer.
The host-owned presenter uses a borrowed ANativeWindow and the existing
window-generation counter. `APP_CMD_TERM_WINDOW` destroys the backend,
swapchain and surface before the Android glue releases the native window;
`APP_CMD_INIT_WINDOW` creates a fresh backend for the new generation. Pause,
stop and focus loss suppress presentation without discarding the unrelated
runtime catalog. Nonrecoverable device/surface errors are logged and presentation
is disabled until a new window generation, avoiding busy retry loops.

While presenting, the native event loop polls without blocking and calls ARC's
`present_surface_frame` with the current native pixel extent. On resize,
ARC's existing Vulkan swapchain code recreates the swapchain on demand.
There is no project scene submitted in this milestone: on-device WSI validation
and representative scene rendering still need V3 and #403 packaging tests.

## Vulkan V3.1 — Android GPU presentation smoke workload

On each valid native surface generation the Android Vulkan presenter now
compiles ARC's existing `make_clear_present_graph("viewport")` and submits
an immutable `render_frame_packet` to `arc-render-vulkan`. Presentation then
uses the existing backend's Vulkan swapchain path. Surface dimension changes
recompile the graph with the new physical output extent; `APP_CMD_TERM_WINDOW`
still destroys all Vulkan resources before the borrowed window is invalidated.
The Android Logcat diagnostics report the selected adapter and Vulkan API
version, and fail the generation if graphics/compute/presentation support
is missing or if graph compilation/submission fails. This is a **renderer
presentation smoke workload**, not a loaded ARC project game scene.

### Manual device acceptance

Use a Vulkan-capable arm64 Android device. Build the Android Gradle fixture
from `engine/platform/android`, install and launch it:

```sh
gradle :app:assembleDebug
adb install -r app/build/outputs/apk/debug/app-debug.apk
adb shell am start -n org.arcengine.host/android.app.NativeActivity
adb logcat -s ARC:I '*:S'
```

Verify `Vulkan adapter:`, `ARC clear/present smoke graph submitted` and
successful frame presentation with no Vulkan validation errors. Rotate the
device repeatedly, switch apps and return, lock/unlock the screen, and finally
close the Activity. Check that each new native-window generation initializes
a fresh renderer, old surfaces are destroyed before window release, and no
device-loss loop or stale-window access occurs. Repeat on both Adreno and
Mali hardware when available and record GPU model, Vulkan API, Android/NDK
versions and driver errors.

**Remaining for full #402 acceptance:** submit an actual project scene
(mesh, camera, lights), demonstrate content loading and scene activation
from the cooked catalog, and collect measured Adreno/Mali validation.
No device test was run as part of this change.

## Vulkan V3.2 — Startup scene materialization and scene extraction

The process-owned Android runtime bridge selects the active provider's scene
artifact from the D1/A4 cooked catalog and reads it through the VFS, then
materializes the archive into a runtime ECS world using ARC's existing
`materialize_cooked_scene` contract. It does not use APK file paths or a
hardcoded scene fixture. Missing/corrupt cooked scene bytes fail with an
ARC diagnostic instead of silently reporting successful scene activation.

The surface-owned renderer now uses `arc::render::renderer`, with a
conservative forward-plus/low policy, and calls `arc::scene::render_scene`
for each active camera frame. ARC's standard `make_scene_draw_graph` and
`renderer::render_frame` submit that render-world event stream through the
same Vulkan backend used on desktop. Without a loaded scene or active camera,
the existing clear/present graph remains a diagnostic fallback. The ECS world
survives pause, surface destruction and renderer re-creation.

This establishes the **scene decode/extraction/render submission boundary**.
Mesh/material GPU resource residency from cooked artifacts is not activated
by the materializer, so nontrivial models may not render correctly until
that resource hydration/streaming integration is completed. Android NDK
compilation and actual Adreno/Mali scene presentation remain unverified.
