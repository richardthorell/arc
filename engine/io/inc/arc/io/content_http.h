#pragma once

/** @file
 * @brief Bounded remote cooked-object HTTP transport boundary and async fetch.
 *
 * This transport-neutral contract validates HTTPS origins, immutable-object
 * addresses and exact HTTP response framing; an optional platform backend
 * performs native TLS and bounded transfers off the render thread. No
 * sockets, credentials or game-facing URLs reach cooked asset/VFS consumers.
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
    etag_mismatch,
    cancelled,
    timeout,
    transport_unavailable,
    network_failure,
    unauthorized,
    not_found,
    rate_limited
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

struct [[nodiscard]] content_http_fetch_result
{
    std::optional<content_http_response> response;
    content_http_validation validation;

    [[nodiscard]] bool succeeded() const noexcept
    {
        return response.has_value() && validation.succeeded();
    }
};

/** Perform a bounded GET on an ARC IO worker. Native Windows uses WinHTTP
 * with platform TLS/certificate/hostname verification and redirects disabled.
 * Linux uses libcurl with system CA/hostname verification when available at
 * build time. Android and other unsupported platforms explicitly return
 * transport_unavailable. The job system must outlive the returned future.
 */
[[nodiscard]] jobs::job_future<content_http_fetch_result>
fetch_content_http(jobs::job_system& jobs, content_http_request request, content_http_limits limits = {},
                   jobs::cancellation_token cancellation = {});

/** C9.2d2: a mutable release pointer is NOT an immutable SHA object.
 * Only one fixed endpoint is allowed:
 * releases/<project-guid>/<target-profile>/current.json
 * The origin must pass the same strict HTTPS/loopback validation as C9.2b.
 */
struct content_http_release_request
{
    std::string origin;
    std::string project_guid;
    std::string target_profile;
};

/** Never treat a release pointer as immutable content or trust HTTP cache
 * freshness. The native backend requests no-cache and rejects redirects.
 */
[[nodiscard]] content_http_validation validate_content_http_release_request(const content_http_release_request& request,
                                                                            const content_http_limits& limits);

[[nodiscard]] content_http_validation
validate_content_http_release_response(const content_http_release_request& request,
                                       const content_http_response& response, const content_http_limits& limits);

/** Perform a bounded GET for the current release descriptor on an IO worker.
 * Windows uses WinHTTP, Linux uses libcurl when available, and all backends
 * enforce system TLS/hostname verification. Unsupported platforms fail
 * explicitly. A release is valid only after C9.1 parse_remote_content_release.
 */
[[nodiscard]] jobs::job_future<content_http_fetch_result>
fetch_content_http_release(jobs::job_system& jobs, content_http_release_request request,
                           content_http_limits limits = {}, jobs::cancellation_token cancellation = {});

} // namespace arc::io
