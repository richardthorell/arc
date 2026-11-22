#pragma once

/** @file
 * @brief Android platform TLS binding for arc-http.
 *
 * NativeActivity owns the JavaVM and app class loader. Register once on
 * Android app startup, unregister after all HTTP client jobs have drained.
 * The native HTTP API remains the same on Windows, Linux and Android.
 */

#if defined(__ANDROID__)

#include <jni.h>

namespace arc::http
{

/** Resolve the app's bundled Java bridge through the Activity's class loader.
 * The native host must have android.permission.INTERNET in its manifest.
 * Returns false if the Java bridge is unavailable; fetches then fail closed.
 */
[[nodiscard]] bool register_android_http_transport(JavaVM* vm, jobject activity) noexcept;

/** Stop new transfers from using the bridge. Caller first drains HTTP jobs. */
void unregister_android_http_transport() noexcept;

} // namespace arc::http

#endif
