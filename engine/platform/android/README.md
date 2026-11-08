# Android native host (M1 / A1)

This is the **minimal Android platform host**, intentionally independent of the
ARC runtime bootstrap (#1094), renderer (#402), and distribution finalizer (#403).
The host currently starts a native `android.app.NativeActivity` with the Android
NDK `native_app_glue` event loop. It does **not** render a frame or load content.

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

## Planned follow-ups

- **A2 (#401):** Map NativeActivity lifecycle, window/surface recreation,
  orientation, input, and app-private writable/asset paths into a host adapter.
- **A3 (#401/#1094):** Invoke the shared ARC runtime launch context from the
  Android host without introducing Android/JNI dependencies into shared code.
- **#402:** Android Vulkan surface/swapchain and actual scene rendering.
- **#403:** Project template/CLI integration, artifact generation, install,
  launch, and logcat diagnostics.

This isolated Gradle application is a build-system smoke fixture, not the
final Android project generator or a replacement for the shared distribution
manifest/cooker.
