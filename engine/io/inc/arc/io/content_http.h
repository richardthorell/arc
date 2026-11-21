#pragma once

/** @file
 * @brief C9.2a bounded remote cooked-object HTTP transport boundary.
 *
 * This transport-neutral contract validates HTTPS origins, immutable-object
 * addresses and exact HTTP response framing before C9.2b supplies platform
 * HTTP/TLS backends. No sockets, credentials or game-facing URLs leak into
 * cooked asset/VFS consumers.
 */

#include <arc/io/io.h>

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <string_view>

namespace arc::io
{

struct content_http_limits
{
    std::uint64_t maximum_object_bytes{256ull * 1024ull * 1024ull};
    std::chrono::milliseconds timeout{15000};
    bool allow_loopback_http{}; ///< Development only; never permit LAN plaintext.
};

struct content_http_request
{
    std::string origin;
    std::string object_path; ///< objects/sha256/<lowercase 64-character digest>
    std::uint64_t object_size{};
    std::optional<std::uint64_t> range_start;
    std::optional<std::uint64_t> range_length;
    std::string if_match_etag; ///< Optional strong ETag, not an authentication token.
};

struct content_http_response
{
    unsigned int status{};
    std::string content_length;
    std::string content_range;
    std::string etag;
    std::string redirect_location;
    file_buffer body;
};

enum class content_http_error_code : std::uint8_t
{
    none,
    invalid_origin,
    invalid_request,
    rejected_redirect,
    invalid_status,
    invalid_range,
    oversized_response,
    inconsistent_length,
    etag_mismatch
};

struct [[nodiscard]] content_http_validation
{
    content_http_error_code code{content_http_error_code::none};
    std::string message;

    [[nodiscard]] bool succeeded() const noexcept
    {
        return code == content_http_error_code::none;
    }
};

/** Reject untrusted or ambiguous origins before issuing network operations. */
[[nodiscard]] content_http_validation validate_content_http_request(const content_http_request& request,
                                                                    const content_http_limits& limits);

/** Verify a completed response against the immutable object and requested
 * byte interval. No redirect following or implicit full-body range fallback.
 * SHA-256 verification and cache commits are performed by C9.2b/C9.2c.
 */
[[nodiscard]] content_http_validation validate_content_http_response(const content_http_request& request,
                                                                     const content_http_response& response,
                                                                     const content_http_limits& limits);

} // namespace arc::io
