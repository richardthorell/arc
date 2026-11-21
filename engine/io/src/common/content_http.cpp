#include <arc/io/content_http.h>

#include <algorithm>
#include <cctype>
#include <charconv>
#include <limits>
#include <utility>
#include <string>
#include <string_view>

namespace arc::io
{
namespace
{

content_http_validation invalid(content_http_error_code code, std::string message)
{
    return {.code = code, .message = std::move(message)};
}

bool ascii_digit(char c) noexcept
{
    return c >= '0' && c <= '9';
}

bool parse_decimal(std::string_view text, std::uint64_t& value) noexcept
{
    if (text.empty() || (text.size() > 1 && text.front() == '0')) return false;
    if (!std::all_of(text.begin(), text.end(), ascii_digit)) return false;
    const auto [end, error] = std::from_chars(text.data(), text.data() + text.size(), value);
    return error == std::errc{} && end == text.data() + text.size();
}

bool digest_path(std::string_view text) noexcept
{
    constexpr std::string_view prefix = "objects/sha256/";
    if (!text.starts_with(prefix) || text.size() != prefix.size() + 64) return false;
    const auto digest = text.substr(prefix.size());
    return std::all_of(digest.begin(), digest.end(), [](char c) { return ascii_digit(c) || (c >= 'a' && c <= 'f'); });
}

bool loopback_host(std::string_view host) noexcept
{
    return host == "localhost" || host == "127.0.0.1" || host == "[::1]";
}

bool hostname(std::string_view host) noexcept
{
    if (host.empty() || host.size() > 253 || host.front() == '.' || host.back() == '.') return false;
    return std::all_of(host.begin(), host.end(),
                       [](unsigned char c)
                       {
                           return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') ||
                                  ascii_digit(static_cast<char>(c)) || c == '-' || c == '.';
                       });
}

bool valid_origin(std::string_view origin, bool dev_loopback) noexcept
{
    const bool https = origin.starts_with("https://");
    const bool http = origin.starts_with("http://");
    if (!https && !http) return false;
    auto authority = origin.substr(https ? 8 : 7);
    if (authority.empty() || authority.find_first_of("/?#@\\%") != std::string_view::npos) return false;
    std::string_view host = authority;
    if (authority.front() == '[')
    {
        const auto close = authority.find(']');
        if (close == std::string_view::npos) return false;
        host = authority.substr(0, close + 1);
        if (close + 1 < authority.size())
        {
            if (authority[close + 1] != ':') return false;
            std::uint64_t port{};
            if (!parse_decimal(authority.substr(close + 2), port) || port == 0 || port > 65535) return false;
        }
    }
    else if (const auto colon = authority.find(':'); colon != std::string_view::npos)
    {
        host = authority.substr(0, colon);
        std::uint64_t port{};
        if (!parse_decimal(authority.substr(colon + 1), port) || port == 0 || port > 65535) return false;
    }
    if (!loopback_host(host) && !hostname(host)) return false;
    return https || (dev_loopback && loopback_host(host));
}

bool strong_etag(std::string_view etag) noexcept
{
    if (etag.empty()) return true;
    if (etag.size() < 2 || etag.front() != '"' || etag.back() != '"') return false;
    return std::all_of(etag.begin() + 1, etag.end() - 1,
                       [](unsigned char c) { return c >= 0x21u && c <= 0x7eu && c != '"'; });
}

bool response_content_range(std::string_view text, std::uint64_t start, std::uint64_t length,
                            std::uint64_t total) noexcept
{
    if (!text.starts_with("bytes ")) return false;
    text.remove_prefix(6);
    const auto dash = text.find('-');
    const auto slash = text.find('/');
    if (dash == std::string_view::npos || slash == std::string_view::npos || dash >= slash) return false;
    std::uint64_t given_start{}, given_end{}, given_total{};
    if (!parse_decimal(text.substr(0, dash), given_start) ||
        !parse_decimal(text.substr(dash + 1, slash - dash - 1), given_end) ||
        !parse_decimal(text.substr(slash + 1), given_total))
        return false;
    return given_start == start && given_total == total && given_end >= given_start &&
           given_end - given_start == length - 1;
}

} // namespace

content_http_validation validate_content_http_request(const content_http_request& request,
                                                      const content_http_limits& limits)
{
    if (!valid_origin(request.origin, limits.allow_loopback_http))
        return invalid(content_http_error_code::invalid_origin,
                       "Remote content requires HTTPS or explicit loopback HTTP");
    if (!digest_path(request.object_path) || !strong_etag(request.if_match_etag) || limits.timeout.count() <= 0 ||
        limits.maximum_object_bytes == 0 || request.object_size == 0 ||
        request.object_size > limits.maximum_object_bytes)
        return invalid(content_http_error_code::invalid_request, "Unsafe content object, ETag or request limits");
    if (request.range_start.has_value() != request.range_length.has_value())
        return invalid(content_http_error_code::invalid_range, "Range start and length must be provided together");
    if (request.range_start)
    {
        const auto start = *request.range_start;
        const auto length = *request.range_length;
        if (length == 0 || start >= request.object_size || length > request.object_size - start)
            return invalid(content_http_error_code::invalid_range, "Range exceeds declared immutable object");
    }
    return {};
}

content_http_validation validate_content_http_response(const content_http_request& request,
                                                       const content_http_response& response,
                                                       const content_http_limits& limits)
{
    auto checked = validate_content_http_request(request, limits);
    if (!checked.succeeded()) return checked;
    // Never follow 30x: a future backend must disable redirects, including
    // cross-origin redirects and signed-URL handoffs unless explicitly trusted.
    if ((response.status >= 300 && response.status < 400) || !response.redirect_location.empty())
        return invalid(content_http_error_code::rejected_redirect, "Untrusted HTTP redirect");
    const bool ranged = request.range_start.has_value();
    if (response.status != (ranged ? 206u : 200u))
        return invalid(content_http_error_code::invalid_status, "Unexpected HTTP status for immutable content");
    const auto expected_bytes = ranged ? *request.range_length : request.object_size;
    if (expected_bytes > limits.maximum_object_bytes || response.body.size() > limits.maximum_object_bytes)
        return invalid(content_http_error_code::oversized_response, "Remote content exceeds configured byte budget");
    std::uint64_t declared_bytes{};
    if (!parse_decimal(response.content_length, declared_bytes) || declared_bytes != expected_bytes ||
        response.body.size() != expected_bytes)
        return invalid(content_http_error_code::inconsistent_length, "HTTP body or Content-Length mismatch");
    if (ranged)
    {
        if (!response_content_range(response.content_range, *request.range_start, *request.range_length,
                                    request.object_size))
            return invalid(content_http_error_code::invalid_range, "HTTP Content-Range does not match request");
    }
    else if (!response.content_range.empty())
        return invalid(content_http_error_code::invalid_range, "Full response unexpectedly declares Content-Range");
    if (!strong_etag(response.etag) || (!request.if_match_etag.empty() && response.etag != request.if_match_etag))
        return invalid(content_http_error_code::etag_mismatch, "ETag is weak, unsafe or changed");
    return {};
}

namespace
{
constexpr std::uint64_t maximum_release_descriptor_bytes = 8ull * 1024ull * 1024ull;

bool valid_release_guid(std::string_view guid) noexcept
{
    if (guid.size() != 36) return false;
    for (std::size_t index = 0; index < guid.size(); ++index)
    {
        const auto c = guid[index];
        if (index == 8 || index == 13 || index == 18 || index == 23)
        {
            if (c != '-') return false;
        }
        else if (!(ascii_digit(c) || (c >= 'a' && c <= 'f')))
            return false;
    }
    return true;
}

bool valid_release_profile(std::string_view profile) noexcept
{
    if (profile.empty() || profile.size() > 128 || profile == "." || profile == ".." ||
        !std::isalnum(static_cast<unsigned char>(profile.front())))
        return false;
    return std::all_of(
        profile.begin(), profile.end(), [](unsigned char c)
        { return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9') || c == '_' || c == '-'; });
}
} // namespace

content_http_validation validate_content_http_release_request(const content_http_release_request& request,
                                                              const content_http_limits& limits)
{
    if (!valid_origin(request.origin, limits.allow_loopback_http))
        return invalid(content_http_error_code::invalid_origin, "Release lookup requires trusted HTTPS or loopback");
    if (!valid_release_guid(request.project_guid) || !valid_release_profile(request.target_profile) ||
        limits.timeout.count() <= 0 || limits.maximum_object_bytes == 0)
        return invalid(content_http_error_code::invalid_request, "Invalid release project/profile or transfer limits");
    return {};
}

content_http_validation validate_content_http_release_response(const content_http_release_request& request,
                                                               const content_http_response& response,
                                                               const content_http_limits& limits)
{
    auto preflight = validate_content_http_release_request(request, limits);
    if (!preflight.succeeded()) return preflight;

    if ((response.status >= 300 && response.status < 400) || !response.redirect_location.empty())
        return invalid(content_http_error_code::rejected_redirect, "Mutable release lookup cannot redirect");
    if (response.status == 401 || response.status == 403)
        return invalid(content_http_error_code::unauthorized, "Release descriptor access denied");
    if (response.status == 404) return invalid(content_http_error_code::not_found, "No release descriptor for target");
    if (response.status == 429 || response.status == 503)
        return invalid(content_http_error_code::rate_limited, "Release descriptor server temporarily unavailable");
    if (response.status != 200)
        return invalid(content_http_error_code::invalid_status, "Unexpected release descriptor HTTP status");
    if (!response.content_range.empty())
        return invalid(content_http_error_code::invalid_range, "Release descriptor cannot be ranged");

    const auto budget = std::min(maximum_release_descriptor_bytes, limits.maximum_object_bytes);
    if (response.body.empty() || response.body.size() > budget)
        return invalid(content_http_error_code::oversized_response, "Release descriptor is empty or exceeds limit");

    if (!response.content_length.empty())
    {
        std::uint64_t declared{};
        if (!parse_decimal(response.content_length, declared) || declared != response.body.size())
            return invalid(content_http_error_code::inconsistent_length, "Release descriptor length mismatch");
    }
    return {};
}

} // namespace arc::io
