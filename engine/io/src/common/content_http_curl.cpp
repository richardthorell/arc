#include <arc/io/content_http.h>

#if defined(ARC_CONTENT_HTTP_CURL) && defined(__linux__) && !defined(__ANDROID__)

#include <curl/curl.h>

#include <algorithm>
#include <charconv>
#include <chrono>
#include <cctype>
#include <climits>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <mutex>
#include <string>
#include <string_view>
#include <utility>

namespace arc::io
{
namespace
{

constexpr std::uint64_t maximum_release_bytes = 8ull * 1024ull * 1024ull;
constexpr std::size_t maximum_header_bytes = 64u * 1024u;

content_http_fetch_result failure(content_http_error_code code, std::string message)
{
    return {.response = std::nullopt, .validation = {.code = code, .message = std::move(message)}};
}

struct transfer_state
{
    content_http_response response;
    jobs::cancellation_token cancellation;
    std::uint64_t maximum_bytes{};
    std::size_t header_bytes{};
    bool body_too_large{};
    bool headers_too_large{};
    bool malformed_headers{};
};

std::string_view trim(std::string_view value) noexcept
{
    while (!value.empty() && (value.front() == ' ' || value.front() == '\t'))
        value.remove_prefix(1);
    while (!value.empty() &&
           (value.back() == '\r' || value.back() == '\n' || value.back() == ' ' || value.back() == '\t'))
        value.remove_suffix(1);
    return value;
}

std::string lower(std::string_view value)
{
    std::string result(value);
    std::transform(result.begin(), result.end(), result.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return result;
}

std::size_t receive_header(char* data, std::size_t size, std::size_t count, void* context)
{
    auto& state = *static_cast<transfer_state*>(context);
    if (size != 0 && count > std::numeric_limits<std::size_t>::max() / size)
    {
        state.headers_too_large = true;
        return 0;
    }
    const auto length = size * count;
    if (length > maximum_header_bytes - state.header_bytes)
    {
        state.headers_too_large = true;
        return 0;
    }
    state.header_bytes += length;
    std::string_view line(data, length);
    if (line.starts_with("HTTP/"))
    {
        // Discard headers from interim 100 responses, never mix two blocks.
        state.response.content_length.clear();
        state.response.content_range.clear();
        state.response.etag.clear();
        state.response.redirect_location.clear();
        return length;
    }
    const auto colon = line.find(':');
    if (colon == std::string_view::npos) return length;
    const auto name = lower(line.substr(0, colon));
    const auto value = trim(line.substr(colon + 1));
    auto set_unique = [&](std::string& output)
    {
        if (!output.empty())
        {
            state.malformed_headers = true;
            return;
        }
        output.assign(value);
    };
    if (name == "content-length")
        set_unique(state.response.content_length);
    else if (name == "content-range")
        set_unique(state.response.content_range);
    else if (name == "etag")
        set_unique(state.response.etag);
    else if (name == "location")
        set_unique(state.response.redirect_location);
    else if (name == "content-encoding" && value != "identity")
        state.malformed_headers = true;
    return state.malformed_headers ? 0 : length;
}

std::size_t receive_body(char* data, std::size_t size, std::size_t count, void* context)
{
    auto& state = *static_cast<transfer_state*>(context);
    if (state.cancellation.stop_requested()) return 0;
    if (size != 0 && count > std::numeric_limits<std::size_t>::max() / size)
    {
        state.body_too_large = true;
        return 0;
    }
    const auto length = size * count;
    if (length > state.maximum_bytes - state.response.body.size())
    {
        state.body_too_large = true;
        return 0;
    }
    const auto* begin = reinterpret_cast<const std::byte*>(data);
    state.response.body.insert(state.response.body.end(), begin, begin + length);
    return length;
}

int transfer_progress(void* context, curl_off_t, curl_off_t, curl_off_t, curl_off_t)
{
    const auto& state = *static_cast<transfer_state*>(context);
    return state.cancellation.stop_requested() ? 1 : 0;
}

bool append_header(curl_slist*& list, const std::string& value)
{
    auto* updated = curl_slist_append(list, value.c_str());
    if (!updated) return false;
    list = updated;
    return true;
}

bool use_direct_loopback(std::string_view origin)
{
    return origin == "http://localhost" || origin.starts_with("http://localhost:") || origin == "http://127.0.0.1" ||
           origin.starts_with("http://127.0.0.1:") || origin == "http://[::1]" || origin.starts_with("http://[::1]:");
}

content_http_fetch_result perform_get(std::string url, std::string_view origin, const content_http_limits& limits,
                                      std::uint64_t byte_limit, const jobs::cancellation_token& cancellation,
                                      std::string range, std::string etag, bool release)
{
    if (cancellation.stop_requested()) return failure(content_http_error_code::cancelled, "Content fetch cancelled");
    static std::once_flag initialization;
    static CURLcode initialization_result = CURLE_FAILED_INIT;
    std::call_once(initialization, [] { initialization_result = curl_global_init(CURL_GLOBAL_DEFAULT); });
    if (initialization_result != CURLE_OK)
        return failure(content_http_error_code::transport_unavailable, "libcurl initialization failed");

    std::unique_ptr<CURL, decltype(&curl_easy_cleanup)> handle(curl_easy_init(), curl_easy_cleanup);
    if (!handle) return failure(content_http_error_code::transport_unavailable, "libcurl session unavailable");
    transfer_state state;
    state.cancellation = cancellation;
    state.maximum_bytes = byte_limit;
    const auto timeout =
        static_cast<long>(std::min<std::int64_t>(limits.timeout.count(), static_cast<std::int64_t>(LONG_MAX)));

    if (curl_easy_setopt(handle.get(), CURLOPT_URL, url.c_str()) != CURLE_OK ||
        curl_easy_setopt(handle.get(), CURLOPT_FOLLOWLOCATION, 0L) != CURLE_OK ||
        curl_easy_setopt(handle.get(), CURLOPT_SSL_VERIFYPEER, 1L) != CURLE_OK ||
        curl_easy_setopt(handle.get(), CURLOPT_SSL_VERIFYHOST, 2L) != CURLE_OK ||
        curl_easy_setopt(handle.get(), CURLOPT_NOSIGNAL, 1L) != CURLE_OK ||
        curl_easy_setopt(handle.get(), CURLOPT_TIMEOUT_MS, timeout) != CURLE_OK ||
        curl_easy_setopt(handle.get(), CURLOPT_CONNECTTIMEOUT_MS, timeout) != CURLE_OK ||
        curl_easy_setopt(handle.get(), CURLOPT_NOPROGRESS, 0L) != CURLE_OK ||
        curl_easy_setopt(handle.get(), CURLOPT_XFERINFOFUNCTION, transfer_progress) != CURLE_OK ||
        curl_easy_setopt(handle.get(), CURLOPT_XFERINFODATA, &state) != CURLE_OK ||
        curl_easy_setopt(handle.get(), CURLOPT_WRITEFUNCTION, receive_body) != CURLE_OK ||
        curl_easy_setopt(handle.get(), CURLOPT_WRITEDATA, &state) != CURLE_OK ||
        curl_easy_setopt(handle.get(), CURLOPT_HEADERFUNCTION, receive_header) != CURLE_OK ||
        curl_easy_setopt(handle.get(), CURLOPT_HEADERDATA, &state) != CURLE_OK ||
        curl_easy_setopt(handle.get(), CURLOPT_USERAGENT, "ARC remote content/1") != CURLE_OK ||
        curl_easy_setopt(handle.get(), CURLOPT_HTTP_CONTENT_DECODING, 0L) != CURLE_OK)
        return failure(content_http_error_code::transport_unavailable, "libcurl safe request setup failed");

#if LIBCURL_VERSION_NUM >= 0x075500
    if (curl_easy_setopt(handle.get(), CURLOPT_PROTOCOLS_STR, "http,https") != CURLE_OK)
        return failure(content_http_error_code::transport_unavailable, "libcurl protocol restriction failed");
#else
    if (curl_easy_setopt(handle.get(), CURLOPT_PROTOCOLS, CURLPROTO_HTTP | CURLPROTO_HTTPS) != CURLE_OK)
        return failure(content_http_error_code::transport_unavailable, "libcurl protocol restriction failed");
#endif

    if (use_direct_loopback(origin) && curl_easy_setopt(handle.get(), CURLOPT_NOPROXY, "*") != CURLE_OK)
        return failure(content_http_error_code::transport_unavailable, "libcurl loopback setup failed");

    curl_slist* headers = nullptr;
    if (!append_header(headers, "Accept-Encoding: identity") ||
        (release && !append_header(headers, "Cache-Control: no-cache, no-store")) ||
        (release && !append_header(headers, "Pragma: no-cache")) ||
        (!etag.empty() && !append_header(headers, "If-Match: " + etag)) ||
        (!range.empty() && !append_header(headers, "Range: bytes=" + range)))
    {
        curl_slist_free_all(headers);
        return failure(content_http_error_code::transport_unavailable, "libcurl request header allocation failed");
    }
    std::unique_ptr<curl_slist, decltype(&curl_slist_free_all)> owned_headers(headers, curl_slist_free_all);
    if (curl_easy_setopt(handle.get(), CURLOPT_HTTPHEADER, headers) != CURLE_OK)
        return failure(content_http_error_code::transport_unavailable, "libcurl HTTP header setup failed");

    const auto code = curl_easy_perform(handle.get());
    long status{};
    static_cast<void>(curl_easy_getinfo(handle.get(), CURLINFO_RESPONSE_CODE, &status));
    state.response.status = status > 0 ? static_cast<unsigned int>(status) : 0u;
    if (state.cancellation.stop_requested())
        return failure(content_http_error_code::cancelled, "Content fetch cancelled");
    if (state.body_too_large)
        return failure(content_http_error_code::oversized_response, "HTTP body exceeds declared content budget");
    if (state.headers_too_large || state.malformed_headers)
        return failure(content_http_error_code::inconsistent_length, "Malformed or oversized HTTP response headers");
    if ((status >= 300 && status < 400) || !state.response.redirect_location.empty())
        return failure(content_http_error_code::rejected_redirect, "Content redirect rejected");
    if (status == 401 || status == 403)
        return failure(content_http_error_code::unauthorized, "Remote content access denied");
    if (status == 404) return failure(content_http_error_code::not_found, "Remote content not found");
    if (status == 429 || status == 503)
        return failure(content_http_error_code::rate_limited, "Remote content temporarily unavailable");
    if (code == CURLE_OPERATION_TIMEDOUT)
        return failure(content_http_error_code::timeout, "Remote content transfer timed out");
    if (code != CURLE_OK)
        return failure(content_http_error_code::network_failure,
                       std::string("libcurl transfer failed: ") + curl_easy_strerror(code));

    char* effective_url = nullptr;
    if (curl_easy_getinfo(handle.get(), CURLINFO_EFFECTIVE_URL, &effective_url) != CURLE_OK || !effective_url ||
        url != effective_url)
        return failure(content_http_error_code::rejected_redirect, "Remote content URL was changed");

    return {.response = std::move(state.response), .validation = {}};
}

} // namespace

content_http_fetch_result curl_fetch_content_http(const content_http_request& request,
                                                  const content_http_limits& limits,
                                                  const jobs::cancellation_token& cancellation)
{
    auto preflight = validate_content_http_request(request, limits);
    if (!preflight.succeeded()) return {.response = std::nullopt, .validation = preflight};
    const auto expected = request.range_start ? *request.range_length : request.object_size;
    std::string range;
    if (request.range_start)
        range = std::to_string(*request.range_start) + "-" +
                std::to_string(*request.range_start + *request.range_length - 1u);
    auto fetched = perform_get(request.origin + "/" + request.object_path, request.origin, limits, expected,
                               cancellation, std::move(range), request.if_match_etag, false);
    if (!fetched.succeeded()) return fetched;
    const auto valid = validate_content_http_response(request, *fetched.response, limits);
    if (!valid.succeeded()) return {.response = std::nullopt, .validation = valid};
    return fetched;
}

content_http_fetch_result curl_fetch_content_http_release(const content_http_release_request& request,
                                                          const content_http_limits& limits,
                                                          const jobs::cancellation_token& cancellation)
{
    auto preflight = validate_content_http_release_request(request, limits);
    if (!preflight.succeeded()) return {.response = std::nullopt, .validation = preflight};
    const auto budget = std::min<std::uint64_t>(maximum_release_bytes, limits.maximum_object_bytes);
    const auto url =
        request.origin + "/releases/" + request.project_guid + "/" + request.target_profile + "/current.json";
    auto fetched = perform_get(url, request.origin, limits, budget, cancellation, {}, {}, true);
    if (!fetched.succeeded()) return fetched;
    const auto valid = validate_content_http_release_response(request, *fetched.response, limits);
    if (!valid.succeeded()) return {.response = std::nullopt, .validation = valid};
    return fetched;
}

} // namespace arc::io

#endif
