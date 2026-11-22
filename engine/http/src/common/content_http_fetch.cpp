#include <arc/http/content_http.h>

#include <algorithm>
#include <charconv>
#include <iterator>
#include <chrono>
#include <limits>
#include <string>
#include <utility>

#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <Windows.h>
#include <winhttp.h>
#endif

namespace arc::http
{
#if defined(ARC_CONTENT_HTTP_CURL) && defined(__linux__) && !defined(__ANDROID__)
content_http_fetch_result curl_fetch_content_http(const content_http_request& request,
                                                  const content_http_limits& limits,
                                                  const jobs::cancellation_token& cancellation);
content_http_fetch_result curl_fetch_content_http_release(const content_http_release_request& request,
                                                          const content_http_limits& limits,
                                                          const jobs::cancellation_token& cancellation);
#endif
namespace
{

content_http_fetch_result failure(content_http_error_code code, std::string message)
{
    return {.response = std::nullopt, .validation = {.code = code, .message = std::move(message)}};
}

#if defined(_WIN32)
class http_handle
{
public:
    explicit http_handle(HINTERNET handle = nullptr) noexcept : handle_(handle) {}
    ~http_handle()
    {
        if (handle_) WinHttpCloseHandle(handle_);
    }
    http_handle(const http_handle&) = delete;
    http_handle& operator=(const http_handle&) = delete;
    [[nodiscard]] HINTERNET get() const noexcept
    {
        return handle_;
    }
    [[nodiscard]] explicit operator bool() const noexcept
    {
        return handle_ != nullptr;
    }

private:
    HINTERNET handle_;
};

std::wstring wide(std::string_view utf8)
{
    if (utf8.empty() || utf8.size() > static_cast<std::size_t>(std::numeric_limits<int>::max())) return {};
    const int size =
        MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS, utf8.data(), static_cast<int>(utf8.size()), nullptr, 0);
    if (!size) return {};
    std::wstring result(static_cast<std::size_t>(size), L'\0');
    if (!MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS, utf8.data(), static_cast<int>(utf8.size()), result.data(),
                             size))
        return {};
    return result;
}

std::string narrow(const std::wstring& text)
{
    if (text.empty()) return {};
    const auto length = WideCharToMultiByte(CP_UTF8, WC_ERR_INVALID_CHARS, text.data(), static_cast<int>(text.size()),
                                            nullptr, 0, nullptr, nullptr);
    if (length <= 0) return {};
    std::string result(static_cast<std::size_t>(length), '\0');
    if (!WideCharToMultiByte(CP_UTF8, WC_ERR_INVALID_CHARS, text.data(), static_cast<int>(text.size()), result.data(),
                             length, nullptr, nullptr))
        return {};
    return result;
}

struct origin_parts
{
    std::wstring host;
    INTERNET_PORT port{};
    bool tls{};
};

origin_parts split_origin(std::string_view origin)
{
    const bool tls = origin.starts_with("https://");
    const auto authority = origin.substr(tls ? 8 : 7);
    const auto first = authority.find(':');
    std::string_view host = authority;
    std::string_view port;
    if (authority.starts_with("["))
    {
        const auto close = authority.find(']');
        host = authority.substr(1, close - 1);
        if (close + 1 < authority.size()) port = authority.substr(close + 2);
    }
    else if (first != std::string_view::npos)
    {
        host = authority.substr(0, first);
        port = authority.substr(first + 1);
    }
    unsigned long parsed_port = tls ? INTERNET_DEFAULT_HTTPS_PORT : INTERNET_DEFAULT_HTTP_PORT;
    if (!port.empty()) parsed_port = std::stoul(std::string(port));
    return {.host = wide(host), .port = static_cast<INTERNET_PORT>(parsed_port), .tls = tls};
}

std::string response_header(HINTERNET request, DWORD header)
{
    DWORD bytes{};
    if (WinHttpQueryHeaders(request, header, WINHTTP_HEADER_NAME_BY_INDEX, nullptr, &bytes, WINHTTP_NO_HEADER_INDEX) ||
        GetLastError() != ERROR_INSUFFICIENT_BUFFER || bytes > 65536u)
        return {};
    std::wstring text(bytes / sizeof(wchar_t), L'\0');
    if (!WinHttpQueryHeaders(request, header, WINHTTP_HEADER_NAME_BY_INDEX, text.data(), &bytes,
                             WINHTTP_NO_HEADER_INDEX))
        return {};
    text.resize(bytes / sizeof(wchar_t));
    while (!text.empty() && text.back() == L'\0')
        text.pop_back();
    return narrow(text);
}

bool cancelled(const jobs::cancellation_token& token) noexcept
{
    return token.stop_requested();
}

content_http_fetch_result winhttp_fetch(const content_http_request& request, const content_http_limits& limits,
                                        const jobs::cancellation_token& token)
{
    if (cancelled(token)) return failure(content_http_error_code::cancelled, "HTTP fetch cancelled");
    const auto origin = split_origin(request.origin);
    if (origin.host.empty()) return failure(content_http_error_code::invalid_origin, "Invalid UTF-8 HTTP host");

    const auto proxy_mode = request.origin.starts_with("http://127.0.0.1") ||
                                    request.origin.starts_with("http://localhost") ||
                                    request.origin.starts_with("http://[::1]")
                                ? WINHTTP_ACCESS_TYPE_NO_PROXY
                                : WINHTTP_ACCESS_TYPE_DEFAULT_PROXY;
    http_handle session(
        WinHttpOpen(L"ARC remote content/1", proxy_mode, WINHTTP_NO_PROXY_NAME, WINHTTP_NO_PROXY_BYPASS, 0));
    if (!session) return failure(content_http_error_code::network_failure, "WinHTTP session unavailable");
    const auto timeout = static_cast<int>(std::min<std::int64_t>(limits.timeout.count(), 600000));
    if (!WinHttpSetTimeouts(session.get(), timeout, timeout, timeout, timeout))
        return failure(content_http_error_code::network_failure, "WinHTTP timeouts could not be configured");

    http_handle connection(WinHttpConnect(session.get(), origin.host.c_str(), origin.port, 0));
    if (!connection)
        return failure(content_http_error_code::network_failure, "WinHTTP connection could not be created");

    const auto resource = wide("/" + request.object_path);
    http_handle handle(WinHttpOpenRequest(connection.get(), L"GET", resource.c_str(), nullptr, WINHTTP_NO_REFERER,
                                          WINHTTP_DEFAULT_ACCEPT_TYPES, origin.tls ? WINHTTP_FLAG_SECURE : 0));
    if (!handle) return failure(content_http_error_code::network_failure, "WinHTTP request could not be created");
    DWORD disable = WINHTTP_DISABLE_REDIRECTS;
    if (!WinHttpSetOption(handle.get(), WINHTTP_OPTION_DISABLE_FEATURE, &disable, sizeof(disable)))
        return failure(content_http_error_code::network_failure, "WinHTTP redirects could not be disabled");

    std::wstring headers;
    if (request.range_start)
    {
        headers += L"Range: bytes=" + std::to_wstring(*request.range_start) + L"-" +
                   std::to_wstring(*request.range_start + *request.range_length - 1) + L"\r\n";
    }
    if (!request.if_match_etag.empty()) headers += L"If-Match: " + wide(request.if_match_etag) + L"\r\n";

    if (cancelled(token)) return failure(content_http_error_code::cancelled, "HTTP fetch cancelled");
    if (!WinHttpSendRequest(handle.get(), headers.empty() ? WINHTTP_NO_ADDITIONAL_HEADERS : headers.c_str(),
                            headers.empty() ? 0 : static_cast<DWORD>(headers.size()), WINHTTP_NO_REQUEST_DATA, 0, 0,
                            0) ||
        !WinHttpReceiveResponse(handle.get(), nullptr))
    {
        const auto error = GetLastError();
        return failure(error == ERROR_WINHTTP_TIMEOUT ? content_http_error_code::timeout
                                                      : content_http_error_code::network_failure,
                       "WinHTTP send/receive failed (" + std::to_string(error) + ")");
    }
    if (cancelled(token)) return failure(content_http_error_code::cancelled, "HTTP fetch cancelled");
    DWORD status{}, status_bytes = sizeof(status);
    if (!WinHttpQueryHeaders(handle.get(), WINHTTP_QUERY_STATUS_CODE | WINHTTP_QUERY_FLAG_NUMBER,
                             WINHTTP_HEADER_NAME_BY_INDEX, &status, &status_bytes, WINHTTP_NO_HEADER_INDEX))
        return failure(content_http_error_code::network_failure, "HTTP status code unavailable");

    content_http_response result;
    result.status = status;
    result.content_length = response_header(handle.get(), WINHTTP_QUERY_CONTENT_LENGTH);
    result.content_range = response_header(handle.get(), WINHTTP_QUERY_CONTENT_RANGE);
    result.etag = response_header(handle.get(), WINHTTP_QUERY_ETAG);
    result.redirect_location = response_header(handle.get(), WINHTTP_QUERY_LOCATION);

    if (status == 401u || status == 403u) return failure(content_http_error_code::unauthorized, "HTTP access denied");
    if (status == 404u) return failure(content_http_error_code::not_found, "Remote cooked object not found");
    if (status == 429u || status == 503u)
        return failure(content_http_error_code::rate_limited, "Remote cooked object temporarily unavailable");
    if ((status >= 300u && status < 400u) || !result.redirect_location.empty())
        return failure(content_http_error_code::rejected_redirect, "Remote content redirect rejected");
    if (status != (request.range_start ? 206u : 200u))
        return failure(content_http_error_code::invalid_status, "HTTP status does not match requested content");

    const auto expected = request.range_start ? *request.range_length : request.object_size;
    if (expected > static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max()))
        return failure(content_http_error_code::oversized_response, "Object exceeds process address space");
    result.body.reserve(static_cast<std::size_t>(expected));
    while (result.body.size() < expected)
    {
        if (cancelled(token)) return failure(content_http_error_code::cancelled, "HTTP fetch cancelled");
        std::byte buffer[16384];
        const auto remaining = expected - result.body.size();
        const DWORD capacity = static_cast<DWORD>(std::min<std::uint64_t>(sizeof(buffer), remaining));
        DWORD read{};
        if (!WinHttpReadData(handle.get(), buffer, capacity, &read))
        {
            const auto error = GetLastError();
            return failure(error == ERROR_WINHTTP_TIMEOUT ? content_http_error_code::timeout
                                                          : content_http_error_code::network_failure,
                           "WinHTTP body read failed (" + std::to_string(error) + ")");
        }
        if (read == 0) break;
        result.body.insert(result.body.end(), buffer, buffer + read);
    }
    // Prove no excess byte is returned even when Content-Length lies or chunked
    // encoding streams beyond the declared immutable object length.
    if (result.body.size() == expected)
    {
        std::byte excess{};
        DWORD read{};
        if (!WinHttpReadData(handle.get(), &excess, 1, &read))
            return failure(content_http_error_code::network_failure, "WinHTTP end-of-body check failed");
        if (read != 0) return failure(content_http_error_code::oversized_response, "HTTP body exceeds declared size");
    }
    if (cancelled(token)) return failure(content_http_error_code::cancelled, "HTTP fetch cancelled");
    auto verified = validate_content_http_response(request, result, limits);
    if (!verified.succeeded()) return {.response = std::nullopt, .validation = std::move(verified)};
    return {.response = std::move(result), .validation = {}};
}

content_http_fetch_result winhttp_fetch_release(const content_http_release_request& request,
                                                const content_http_limits& limits,
                                                const jobs::cancellation_token& token)
{
    if (cancelled(token)) return failure(content_http_error_code::cancelled, "Release fetch cancelled");
    const auto origin = split_origin(request.origin);
    if (origin.host.empty()) return failure(content_http_error_code::invalid_origin, "Invalid UTF-8 release host");
    const bool loopback = request.origin.starts_with("http://127.0.0.1") ||
                          request.origin.starts_with("http://localhost") || request.origin.starts_with("http://[::1]");
    http_handle session(WinHttpOpen(L"ARC remote release/1",
                                    loopback ? WINHTTP_ACCESS_TYPE_NO_PROXY : WINHTTP_ACCESS_TYPE_DEFAULT_PROXY,
                                    WINHTTP_NO_PROXY_NAME, WINHTTP_NO_PROXY_BYPASS, 0));
    if (!session) return failure(content_http_error_code::network_failure, "WinHTTP session unavailable");
    const auto timeout = static_cast<int>(std::min<std::int64_t>(limits.timeout.count(), 600000));
    if (!WinHttpSetTimeouts(session.get(), timeout, timeout, timeout, timeout))
        return failure(content_http_error_code::network_failure, "WinHTTP timeout setup failed");

    http_handle connection(WinHttpConnect(session.get(), origin.host.c_str(), origin.port, 0));
    if (!connection) return failure(content_http_error_code::network_failure, "WinHTTP connection failed");

    const auto resource = wide("/releases/" + request.project_guid + "/" + request.target_profile + "/current.json");
    if (resource.empty()) return failure(content_http_error_code::invalid_request, "Invalid release resource");
    http_handle handle(WinHttpOpenRequest(connection.get(), L"GET", resource.c_str(), nullptr, WINHTTP_NO_REFERER,
                                          WINHTTP_DEFAULT_ACCEPT_TYPES, origin.tls ? WINHTTP_FLAG_SECURE : 0));
    if (!handle) return failure(content_http_error_code::network_failure, "WinHTTP release request failed");
    DWORD disable = WINHTTP_DISABLE_REDIRECTS;
    if (!WinHttpSetOption(handle.get(), WINHTTP_OPTION_DISABLE_FEATURE, &disable, sizeof(disable)))
        return failure(content_http_error_code::network_failure, "Could not disable release redirects");
    constexpr wchar_t headers[] = L"Cache-Control: no-cache, no-store\r\nPragma: no-cache\r\n";
    if (cancelled(token)) return failure(content_http_error_code::cancelled, "Release fetch cancelled");
    if (!WinHttpSendRequest(handle.get(), headers, static_cast<DWORD>(std::size(headers) - 1), WINHTTP_NO_REQUEST_DATA,
                            0, 0, 0) ||
        !WinHttpReceiveResponse(handle.get(), nullptr))
    {
        const auto error = GetLastError();
        return failure(error == ERROR_WINHTTP_TIMEOUT ? content_http_error_code::timeout
                                                      : content_http_error_code::network_failure,
                       "WinHTTP release send/receive failed (" + std::to_string(error) + ")");
    }
    if (cancelled(token)) return failure(content_http_error_code::cancelled, "Release fetch cancelled");

    DWORD status{}, status_bytes = sizeof(status);
    if (!WinHttpQueryHeaders(handle.get(), WINHTTP_QUERY_STATUS_CODE | WINHTTP_QUERY_FLAG_NUMBER,
                             WINHTTP_HEADER_NAME_BY_INDEX, &status, &status_bytes, WINHTTP_NO_HEADER_INDEX))
        return failure(content_http_error_code::network_failure, "Release response status unavailable");

    content_http_response response;
    response.status = status;
    response.content_length = response_header(handle.get(), WINHTTP_QUERY_CONTENT_LENGTH);
    response.content_range = response_header(handle.get(), WINHTTP_QUERY_CONTENT_RANGE);
    response.redirect_location = response_header(handle.get(), WINHTTP_QUERY_LOCATION);

    // Reject statuses and redirects before receiving response bodies.
    if (status == 401 || status == 403) return failure(content_http_error_code::unauthorized, "Release access denied");
    if (status == 404) return failure(content_http_error_code::not_found, "Release descriptor missing");
    if (status == 429 || status == 503)
        return failure(content_http_error_code::rate_limited, "Release server temporarily unavailable");
    if ((status >= 300 && status < 400) || !response.redirect_location.empty())
        return failure(content_http_error_code::rejected_redirect, "Release redirect rejected");
    if (status != 200) return failure(content_http_error_code::invalid_status, "Unexpected release HTTP status");

    constexpr std::uint64_t descriptor_limit = 8ull * 1024ull * 1024ull;
    const auto budget = std::min(descriptor_limit, limits.maximum_object_bytes);
    if (!response.content_length.empty())
    {
        std::uint64_t declared{};
        const auto first = response.content_length.data();
        const auto [end, ec] = std::from_chars(first, first + response.content_length.size(), declared);
        if (ec != std::errc{} || end != first + response.content_length.size())
            return failure(content_http_error_code::inconsistent_length, "Invalid release Content-Length");
        if (declared > budget)
            return failure(content_http_error_code::oversized_response, "Release Content-Length exceeds byte budget");
        response.body.reserve(static_cast<std::size_t>(declared));
    }

    for (;;)
    {
        if (cancelled(token)) return failure(content_http_error_code::cancelled, "Release fetch cancelled");
        if (response.body.size() > budget)
            return failure(content_http_error_code::oversized_response, "Release descriptor exceeds byte budget");
        std::byte buffer[16384];
        const DWORD capacity =
            static_cast<DWORD>(std::min<std::uint64_t>(sizeof(buffer), budget + 1u - response.body.size()));
        DWORD read{};
        if (!WinHttpReadData(handle.get(), buffer, capacity, &read))
        {
            const auto error = GetLastError();
            return failure(error == ERROR_WINHTTP_TIMEOUT ? content_http_error_code::timeout
                                                          : content_http_error_code::network_failure,
                           "Release body read failed (" + std::to_string(error) + ")");
        }
        if (read == 0) break;
        response.body.insert(response.body.end(), buffer, buffer + read);
    }
    if (cancelled(token)) return failure(content_http_error_code::cancelled, "Release fetch cancelled");
    const auto validation = validate_content_http_release_response(request, response, limits);
    if (!validation.succeeded()) return {.response = std::nullopt, .validation = validation};
    return {.response = std::move(response), .validation = {}};
}

#endif

} // namespace

jobs::job_future<content_http_fetch_result> fetch_content_http(jobs::job_system& jobs, content_http_request request,
                                                               content_http_limits limits,
                                                               jobs::cancellation_token cancellation)
{
    return jobs.submit_future({.name = "content-http-fetch",
                               .priority = jobs::job_priority::normal,
                               .affinity = jobs::job_affinity::io_thread},
                              [request = std::move(request), limits, cancellation]() -> content_http_fetch_result
                              {
                                  const auto preflight = validate_content_http_request(request, limits);
                                  if (!preflight.succeeded())
                                      return {.response = std::nullopt, .validation = preflight};
                                  if (cancellation.stop_requested())
                                      return failure(content_http_error_code::cancelled, "HTTP fetch cancelled");
#if defined(_WIN32)
                                  return winhttp_fetch(request, limits, cancellation);
#elif defined(ARC_CONTENT_HTTP_CURL) && defined(__linux__) && !defined(__ANDROID__)
                                  return curl_fetch_content_http(request, limits, cancellation);
#else
                                  return failure(content_http_error_code::transport_unavailable,
                                                 "Native HTTP/TLS backend not available for this platform");
#endif
                              });
}

jobs::job_future<content_http_fetch_result> fetch_content_http_release(jobs::job_system& jobs,
                                                                       content_http_release_request request,
                                                                       content_http_limits limits,
                                                                       jobs::cancellation_token cancellation)
{
    return jobs.submit_future({.name = "content-http-release-fetch",
                               .priority = jobs::job_priority::normal,
                               .affinity = jobs::job_affinity::io_thread},
                              [request = std::move(request), limits, cancellation]() -> content_http_fetch_result
                              {
                                  const auto preflight = validate_content_http_release_request(request, limits);
                                  if (!preflight.succeeded())
                                      return {.response = std::nullopt, .validation = preflight};
                                  if (cancellation.stop_requested())
                                      return failure(content_http_error_code::cancelled, "Release lookup cancelled");
#if defined(_WIN32)
                                  return winhttp_fetch_release(request, limits, cancellation);
#elif defined(ARC_CONTENT_HTTP_CURL) && defined(__linux__) && !defined(__ANDROID__)
                                  return curl_fetch_content_http_release(request, limits, cancellation);
#else
            return failure(content_http_error_code::transport_unavailable,
                           "Native release HTTP/TLS backend unavailable on this platform");
#endif
                              });
}

} // namespace arc::http
