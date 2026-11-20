#pragma once

/** @file
 * @brief C9.1 native transport-independent remote cooked-content descriptor contract.
 *
 * Validates publisher metadata before C9.2 downloads immutable objects or
 * mounts the existing cooked asset catalog. No networking or JSON dependency
 * is exposed through this header.
 */

#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace arc::assets
{

struct remote_content_target
{
    std::string platform;
    std::string architecture;
    std::string renderer;
    std::string cook_profile;
    std::string texture_family;
};

struct remote_content_expectations
{
    std::string project_guid;
    std::string engine_version;
    std::string base_build_id;
    remote_content_target target;
    std::uint64_t minimum_generation{};
};

struct remote_content_object
{
    std::string path;
    std::string sha256;
    std::uint64_t size{};
};

struct remote_content_chunk
{
    std::string name;
    remote_content_object object;
};

struct remote_content_release
{
    std::string project_guid;
    std::string engine_version;
    std::string base_build_id;
    remote_content_target target;
    std::uint64_t generation{};
    std::string cook_manifest_name;
    remote_content_object cook_manifest;
    std::vector<remote_content_chunk> chunks;
};

struct remote_content_parse_result
{
    std::optional<remote_content_release> release;
    std::string error;
    [[nodiscard]] bool succeeded() const noexcept
    {
        return release.has_value();
    }
};

/** Fail-closed parse/validation of C9.1 version 1 (max 8 MiB metadata).
 * project/engine/build/target must exactly match the runtime expectations;
 * generation must be strictly greater than minimum_generation.
 */
[[nodiscard]] remote_content_parse_result parse_remote_content_release(std::string_view json_bytes,
                                                                       const remote_content_expectations& expected);

enum class remote_content_feed_action : std::uint8_t
{
    replay,
    full_snapshot,
    up_to_date,
    reject_regression,
    invalid
};

/** Decide whether the optional dev change feed can replay from last ACK. */
[[nodiscard]] remote_content_feed_action remote_content_reconnect(std::uint64_t last_acknowledged,
                                                                  std::uint64_t oldest_available,
                                                                  std::uint64_t newest_available) noexcept;

} // namespace arc::assets
