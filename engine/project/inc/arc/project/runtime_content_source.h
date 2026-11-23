#pragma once

/** @file
 * @brief C9.3 host-neutral, Development-only cooked-content endpoint selection.
 *
 * These are launch-time inputs, not part of the D0 distribution manifest.
 * C9.4/6 owns HTTP serving and actual verified generation activation.
 */

#include <arc/project/runtime_launch.h>

#include <filesystem>
#include <optional>
#include <span>
#include <string>
#include <string_view>

namespace arc::project
{

enum class runtime_content_mode : std::uint8_t
{
    packaged,
    remote,
    hybrid
};

enum class runtime_content_origin : std::uint8_t
{
    distribution,
    project_development,
    trusted_launch
};

/** A field omitted by the host cannot override a lower-precedence setting.
 * Endpoints are never serialized into the signed D0 distribution manifest.
 */
struct runtime_content_options
{
    std::optional<std::string> server;
    std::optional<runtime_content_mode> mode;
};

struct runtime_content_launch
{
    runtime_content_options project_development;
    runtime_content_options trusted_launch;
};

/** C9.3 records the requested data source. It does not imply a live release
 * has been acquired or mounted; C9.4/6 must report that separately.
 */
struct runtime_content_selection
{
    runtime_content_mode mode{runtime_content_mode::packaged};
    runtime_content_origin origin{runtime_content_origin::distribution};
    std::string server_origin; ///< Validated URL origin only, never credentials or a resource path.
    std::string project_guid;
    std::string target_id;
    std::string cook_profile;
    std::filesystem::path cache_root; ///< OS-private writable storage root / RemoteContent.
};

using runtime_content_options_result = core::result<runtime_content_options, project_error>;
using runtime_content_selection_result = core::result<runtime_content_selection, project_error>;

/** Scan argv (without program name). Ignore unrelated host flags such as
 * --ci-smoke; reject duplicate, malformed, or unknown --content-* options.
 * Supports both --content-server URL and --content-server=URL, similarly mode.
 */
[[nodiscard]] runtime_content_options_result
parse_runtime_content_arguments(std::span<const std::string_view> arguments);

/** Android NativeActivity supplies non-secret Intent metadata rather than
 * command-line arguments. Empty optional means absent; present-but-empty is
 * invalid. Both hosts use the same selector and Shipping policy.
 */
[[nodiscard]] runtime_content_options_result parse_android_content_metadata(std::optional<std::string_view> server,
                                                                            std::optional<std::string_view> mode);

/** Only callable after manifest and compiled host identity have been checked.
 * Shipping always uses packaged content, rejecting even a "packaged" override.
 * Endpoint policy reuses arc-http's strict HTTPS / opt-in loopback origin rules.
 */
[[nodiscard]] runtime_content_selection_result
select_runtime_content(const distribution_manifest& manifest, const runtime_content_launch& config,
                       const std::filesystem::path& writable_storage_root);

[[nodiscard]] std::string_view runtime_content_mode_name(runtime_content_mode mode) noexcept;
[[nodiscard]] std::string_view runtime_content_origin_name(runtime_content_origin origin) noexcept;

} // namespace arc::project
