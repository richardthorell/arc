#pragma once

/** @file
 * @brief C9.2d2 discovery of a mutable current.json release pointer.
 *
 * Only the existing C9.1 descriptor is trusted to declare immutable objects.
 * This step never downloads cooked blobs or mounts content. A newer, complete,
 * compatible release is returned for subsequent C9.2d1 acquisition.
 */

#include <arc/assets/remote_content_transfer.h>

#include <functional>
#include <optional>
#include <string>

namespace arc::assets
{

struct [[nodiscard]] remote_release_discovery_result
{
    std::optional<remote_content_release> release;
    http::content_http_error_code transport_error{http::content_http_error_code::none};
    std::string error;

    [[nodiscard]] bool succeeded() const noexcept
    {
        return release.has_value() && error.empty();
    }
};

using remote_release_fetch_executor = std::function<jobs::job_future<http::content_http_fetch_result>(
    jobs::job_system&, http::content_http_release_request, http::content_http_limits, jobs::cancellation_token)>;

/** Request the fixed, no-cache current release endpoint on an IO worker,
 * validate the HTTP framing, then use the existing C9.1 descriptor parser to
 * enforce project/engine/build/target and strictly increasing generation.
 *
 * Runs the coordinator on a general worker. Production execution uses the
 * native WinHTTP backend; injected executors are for deterministic tests.
 * No catalog is mutated, even when the candidate release is valid.
 */
[[nodiscard]] jobs::job_future<remote_release_discovery_result>
discover_remote_release(jobs::job_system& jobs, std::string origin, remote_content_expectations expectations,
                        http::content_http_limits limits = {}, jobs::cancellation_token cancellation = {},
                        remote_release_fetch_executor executor = {});

} // namespace arc::assets
