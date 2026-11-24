#include <arc/http/android_http.h>
#include <arc/http/content_http.h>

#if defined(__ANDROID__)

#include <algorithm>
#include <array>
#include <charconv>
#include <cstdint>
#include <limits>
#include <mutex>
#include <optional>
#include <string>
#include <string_view>
#include <utility>

namespace arc::http
{
namespace
{
constexpr std::size_t chunk_size = 16u * 1024u;
constexpr std::uint64_t max_release_bytes = 8ull * 1024ull * 1024ull;

std::mutex registration_mutex;
JavaVM* registered_vm = nullptr;
jclass registered_class = nullptr; // Owned global ref; protected by mutex.
jmethodID registered_open = nullptr;

content_http_fetch_result failure(content_http_error_code code, std::string message)
{
    return {.response = std::nullopt, .validation = {.code = code, .message = std::move(message)}};
}

class attached_thread
{
public:
    explicit attached_thread(JavaVM* vm) : vm_(vm)
    {
        if (!vm_) return;
        const auto status = vm_->GetEnv(reinterpret_cast<void**>(&env_), JNI_VERSION_1_6);
        if (status == JNI_EDETACHED)
        {
            if (vm_->AttachCurrentThread(&env_, nullptr) == JNI_OK) attached_ = true;
        }
        else if (status != JNI_OK)
            env_ = nullptr;
    }
    attached_thread(const attached_thread&) = delete;
    attached_thread& operator=(const attached_thread&) = delete;
    ~attached_thread()
    {
        if (attached_) vm_->DetachCurrentThread();
    }
    [[nodiscard]] JNIEnv* get() const noexcept
    {
        return env_;
    }

private:
    JavaVM* vm_{};
    JNIEnv* env_{};
    bool attached_{};
};

class local_ref
{
public:
    local_ref(JNIEnv* env, jobject object) : env_(env), object_(object) {}
    local_ref(const local_ref&) = delete;
    local_ref& operator=(const local_ref&) = delete;
    ~local_ref()
    {
        if (object_) env_->DeleteLocalRef(object_);
    }
    [[nodiscard]] jobject get() const noexcept
    {
        return object_;
    }

private:
    JNIEnv* env_{};
    jobject object_{};
};

bool clear_exception(JNIEnv* env) noexcept
{
    if (!env->ExceptionCheck()) return false;
    env->ExceptionClear();
    return true;
}

content_http_error_code java_exception(JNIEnv* env) noexcept
{
    if (!env->ExceptionCheck()) return content_http_error_code::network_failure;
    local_ref error(env, env->ExceptionOccurred());
    env->ExceptionClear();
    local_ref timeout_class(env, env->FindClass("java/net/SocketTimeoutException"));
    if (clear_exception(env) || !timeout_class.get()) return content_http_error_code::network_failure;
    return env->IsInstanceOf(error.get(), static_cast<jclass>(timeout_class.get()))
               ? content_http_error_code::timeout
               : content_http_error_code::network_failure;
}

std::string from_java(JNIEnv* env, jstring value)
{
    if (!value) return {};
    const char* text = env->GetStringUTFChars(value, nullptr);
    if (!text) return {};
    std::string result(text);
    env->ReleaseStringUTFChars(value, text);
    return result;
}

std::string field_string(JNIEnv* env, jobject response, jclass cls, const char* name)
{
    const auto id = env->GetFieldID(cls, name, "Ljava/lang/String;");
    if (!id || env->ExceptionCheck()) return {};
    local_ref value(env, env->GetObjectField(response, id));
    if (env->ExceptionCheck()) return {};
    return from_java(env, static_cast<jstring>(value.get()));
}

class response_guard
{
public:
    response_guard(JNIEnv* env, jobject response, jmethodID close) : env_(env), response_(response), close_(close) {}
    response_guard(const response_guard&) = delete;
    response_guard& operator=(const response_guard&) = delete;
    ~response_guard()
    {
        if (response_ && close_)
        {
            // A pending exception must not prevent connection cleanup.
            if (env_->ExceptionCheck()) env_->ExceptionClear();
            env_->CallVoidMethod(response_, close_);
            if (env_->ExceptionCheck()) env_->ExceptionClear();
        }
        if (response_) env_->DeleteLocalRef(response_);
    }

private:
    JNIEnv* env_{};
    jobject response_{};
    jmethodID close_{};
};

content_http_fetch_result fetch(std::string url, std::string range, std::string etag, bool release,
                                std::uint64_t byte_budget, const content_http_limits& limits,
                                const jobs::cancellation_token& cancellation)
{
    if (cancellation.stop_requested()) return failure(content_http_error_code::cancelled, "Android HTTP cancelled");
    JavaVM* vm{};
    jclass transport{};
    jmethodID open{};
    {
        std::lock_guard lock(registration_mutex);
        vm = registered_vm;
        open = registered_open;
        if (!vm || !registered_class || !open)
            return failure(content_http_error_code::transport_unavailable, "Android HTTP bridge not registered");
        // The class local ref survives unregister while this transfer runs.
        attached_thread thread(vm);
        if (!thread.get())
            return failure(content_http_error_code::transport_unavailable, "Cannot attach Android HTTP thread");
        transport = static_cast<jclass>(thread.get()->NewGlobalRef(registered_class));
        if (!transport || clear_exception(thread.get()))
            return failure(content_http_error_code::transport_unavailable, "Android HTTP class unavailable");
    }

    // Keep the class valid throughout JNI calls and release it before detaching.
    attached_thread thread(vm);
    if (!thread.get())
        return failure(content_http_error_code::transport_unavailable, "Cannot attach Android HTTP thread");
    JNIEnv* env = thread.get();
    local_ref bridge(env, transport);
    if (byte_budget >= static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max()) ||
        byte_budget == std::numeric_limits<std::uint64_t>::max())
        return failure(content_http_error_code::oversized_response, "HTTP object exceeds process address space");

    local_ref java_url(env, env->NewStringUTF(url.c_str()));
    local_ref java_range(env, env->NewStringUTF(range.c_str()));
    local_ref java_etag(env, env->NewStringUTF(etag.c_str()));
    local_ref java_token(env, env->NewStringUTF(limits.development_bearer_token.c_str()));
    if (clear_exception(env) || !java_url.get() || !java_range.get() || !java_etag.get() || !java_token.get())
        return failure(content_http_error_code::network_failure, "Android HTTP request allocation failed");

    const auto timeout =
        static_cast<jint>(std::min<std::int64_t>(limits.timeout.count(), std::numeric_limits<jint>::max()));
    jobject response =
        env->CallStaticObjectMethod(static_cast<jclass>(bridge.get()), open, java_url.get(), java_range.get(),
                                    java_etag.get(), java_token.get(), release ? JNI_TRUE : JNI_FALSE, timeout);
    if (env->ExceptionCheck())
    {
        const auto code = java_exception(env);
        return failure(code, code == content_http_error_code::timeout
                                 ? "Android HTTP transfer timed out"
                                 : "Android HTTP connection or certificate validation failed");
    }
    if (!response) return failure(content_http_error_code::network_failure, "Android HTTP returned no response");

    local_ref cls(env, env->GetObjectClass(response));
    if (clear_exception(env) || !cls.get())
    {
        response_guard cleanup(env, response, nullptr);
        return failure(content_http_error_code::network_failure, "Android HTTP response type invalid");
    }
    const auto response_cls = static_cast<jclass>(cls.get());
    const auto close = env->GetMethodID(response_cls, "close", "()V");
    if (clear_exception(env) || !close)
    {
        response_guard cleanup(env, response, nullptr);
        return failure(content_http_error_code::network_failure, "Android HTTP response cannot close");
    }
    response_guard cleanup(env, response, close);
    const auto status_field = env->GetFieldID(response_cls, "status", "I");
    const auto read = env->GetMethodID(response_cls, "read", "([B)I");
    if (clear_exception(env) || !status_field || !read)
        return failure(content_http_error_code::network_failure, "Invalid Android HTTP response methods");

    content_http_response out;
    const auto status = env->GetIntField(response, status_field);
    if (clear_exception(env) || status < 0)
        return failure(content_http_error_code::network_failure, "Android HTTP status unavailable");
    out.status = static_cast<unsigned int>(status);
    out.content_length = field_string(env, response, response_cls, "contentLength");
    out.content_range = field_string(env, response, response_cls, "contentRange");
    out.etag = field_string(env, response, response_cls, "etag");
    out.redirect_location = field_string(env, response, response_cls, "redirectLocation");
    if (clear_exception(env))
        return failure(content_http_error_code::network_failure, "Android HTTP response metadata unavailable");

    if ((out.status >= 300 && out.status < 400) || !out.redirect_location.empty())
        return failure(content_http_error_code::rejected_redirect, "Android HTTP redirects rejected");
    if (out.status == 401 || out.status == 403)
        return failure(content_http_error_code::unauthorized, "Android HTTP access denied");
    if (out.status == 404) return failure(content_http_error_code::not_found, "Android HTTP object missing");
    if (out.status == 429 || out.status == 503)
        return failure(content_http_error_code::rate_limited, "Android HTTP server temporarily unavailable");
    if (out.status == 412)
        return failure(content_http_error_code::etag_mismatch, "Android HTTP ETag precondition failed");
    if (out.status != (release || range.empty() ? 200u : 206u))
        return failure(content_http_error_code::invalid_status, "Android HTTP unexpected status");

    if (!out.content_length.empty())
    {
        std::uint64_t declared{};
        const auto [end, ec] =
            std::from_chars(out.content_length.data(), out.content_length.data() + out.content_length.size(), declared);
        if (ec != std::errc{} || end != out.content_length.data() + out.content_length.size())
            return failure(content_http_error_code::inconsistent_length, "Malformed Android Content-Length");
        if (declared > byte_budget)
            return failure(content_http_error_code::oversized_response, "Android HTTP declared body exceeds limit");
        out.body.reserve(static_cast<std::size_t>(declared));
    }

    local_ref chunk(env, env->NewByteArray(static_cast<jsize>(chunk_size)));
    if (clear_exception(env) || !chunk.get())
        return failure(content_http_error_code::network_failure, "Android HTTP read buffer allocation failed");

    std::array<std::byte, chunk_size> bytes{};
    for (;;)
    {
        if (cancellation.stop_requested()) return failure(content_http_error_code::cancelled, "Android HTTP cancelled");
        const auto received = env->CallIntMethod(response, read, chunk.get());
        if (env->ExceptionCheck())
        {
            const auto code = java_exception(env);
            return failure(code, code == content_http_error_code::timeout ? "Android HTTP body read timed out"
                                                                          : "Android HTTP body read failed");
        }
        if (received < 0) break;
        if (received == 0) continue;
        if (static_cast<std::size_t>(received) > bytes.size())
            return failure(content_http_error_code::network_failure, "Android HTTP returned oversized read");
        if (static_cast<std::uint64_t>(received) > byte_budget - out.body.size())
            return failure(content_http_error_code::oversized_response, "Android HTTP body exceeds byte budget");
        env->GetByteArrayRegion(static_cast<jbyteArray>(chunk.get()), 0, received,
                                reinterpret_cast<jbyte*>(bytes.data()));
        if (clear_exception(env))
            return failure(content_http_error_code::network_failure, "Android HTTP response copy failed");
        out.body.insert(out.body.end(), bytes.begin(), bytes.begin() + received);
    }
    if (cancellation.stop_requested()) return failure(content_http_error_code::cancelled, "Android HTTP cancelled");
    return {.response = std::move(out), .validation = {}};
}

} // namespace

bool register_android_http_transport(JavaVM* vm, jobject activity) noexcept
{
    if (!vm || !activity) return false;
    attached_thread thread(vm);
    JNIEnv* env = thread.get();
    if (!env) return false;
    local_ref activity_class(env, env->GetObjectClass(activity));
    if (!activity_class.get() || clear_exception(env)) return false;
    const auto get_loader =
        env->GetMethodID(static_cast<jclass>(activity_class.get()), "getClassLoader", "()Ljava/lang/ClassLoader;");
    if (!get_loader || clear_exception(env)) return false;
    local_ref loader(env, env->CallObjectMethod(activity, get_loader));
    if (!loader.get() || clear_exception(env)) return false;
    local_ref loader_class(env, env->GetObjectClass(loader.get()));
    if (!loader_class.get() || clear_exception(env)) return false;
    const auto load =
        env->GetMethodID(static_cast<jclass>(loader_class.get()), "loadClass", "(Ljava/lang/String;)Ljava/lang/Class;");
    if (!load || clear_exception(env)) return false;
    local_ref name(env, env->NewStringUTF("org.arcengine.host.http.ArcHttpTransport"));
    if (!name.get() || clear_exception(env)) return false;
    local_ref java_class(env, env->CallObjectMethod(loader.get(), load, name.get()));
    if (!java_class.get() || clear_exception(env)) return false;
    const auto clazz = static_cast<jclass>(java_class.get());
    const auto open = env->GetStaticMethodID(clazz, "open",
                                             "(Ljava/lang/String;Ljava/lang/String;Ljava/lang/String;Ljava/lang/String;ZI)"
                                             "Lorg/arcengine/host/http/ArcHttpTransport$Response;");
    if (!open || clear_exception(env)) return false;
    auto* global = static_cast<jclass>(env->NewGlobalRef(clazz));
    if (!global || clear_exception(env)) return false;
    std::lock_guard lock(registration_mutex);
    if (registered_class) env->DeleteGlobalRef(registered_class);
    registered_class = global;
    registered_vm = vm;
    registered_open = open;
    return true;
}

void unregister_android_http_transport() noexcept
{
    std::lock_guard lock(registration_mutex);
    if (registered_vm && registered_class)
    {
        attached_thread thread(registered_vm);
        if (thread.get()) thread.get()->DeleteGlobalRef(registered_class);
    }
    registered_class = nullptr;
    registered_vm = nullptr;
    registered_open = nullptr;
}

content_http_fetch_result android_fetch_content_http(const content_http_request& request,
                                                     const content_http_limits& limits,
                                                     const jobs::cancellation_token& cancellation)
{
    auto valid = validate_content_http_request(request, limits);
    if (!valid.succeeded()) return {.response = std::nullopt, .validation = valid};
    std::string range;
    if (request.range_start)
        range = "bytes=" + std::to_string(*request.range_start) + "-" +
                std::to_string(*request.range_start + *request.range_length - 1u);
    const auto maximum = request.range_start ? *request.range_length : request.object_size;
    auto fetched = fetch(request.origin + "/" + request.object_path, range, request.if_match_etag, false, maximum,
                         limits, cancellation);
    if (!fetched.succeeded()) return fetched;
    valid = validate_content_http_response(request, *fetched.response, limits);
    if (!valid.succeeded()) return {.response = std::nullopt, .validation = valid};
    return fetched;
}

content_http_fetch_result android_fetch_content_http_release(const content_http_release_request& request,
                                                             const content_http_limits& limits,
                                                             const jobs::cancellation_token& cancellation)
{
    auto valid = validate_content_http_release_request(request, limits);
    if (!valid.succeeded()) return {.response = std::nullopt, .validation = valid};
    auto fetched =
        fetch(request.origin + "/releases/" + request.project_guid + "/" + request.target_profile + "/current.json", {},
              {}, true, std::min(max_release_bytes, limits.maximum_object_bytes), limits, cancellation);
    if (!fetched.succeeded()) return fetched;
    valid = validate_content_http_release_response(request, *fetched.response, limits);
    if (!valid.succeeded()) return {.response = std::nullopt, .validation = valid};
    return fetched;
}
} // namespace arc::http

#endif
