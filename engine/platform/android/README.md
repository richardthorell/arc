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
