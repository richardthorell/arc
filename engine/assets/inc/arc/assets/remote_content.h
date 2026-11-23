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
    /** Optional C9 v1 cooked source identity. Empty in in-memory legacy
     * descriptors means the same ID as the packaged runtime base build.
     */
    std::string cook_build_id;
    remote_content_target target;
    std::uint64_t generation{};
    std::string cook_manifest_name;
    remote_content_object cook_manifest;
    std::vector<remote_content_chunk> chunks;
};

struct [[nodiscard]] remote_content_parse_result
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

/** Replay is bounded so an old client always gets a full snapshot rather
 * than an unbounded event list. The default is part of C9 v1 wire policy.
 */
[[nodiscard]] remote_content_feed_action remote_content_reconnect(std::uint64_t last_acknowledged,
                                                                  std::uint64_t oldest_available,
                                                                  std::uint64_t newest_available,
                                                                  std::uint64_t maximum_replay_events = 256) noexcept;

/** A committed immutable release descriptor address in a development feed.
 * This is a descriptor SHA reference, not a mutable current.json pointer.
 * C9.4 publishes these bytes before notifying, and C9.6 validates them
 * against the existing parse_remote_content_release() before activation.
 */
struct remote_content_feed_event
{
    std::uint64_t sequence{};
    std::uint64_t generation{};
    remote_content_object release;
    std::vector<std::string> changed_guids;
    std::vector<std::string> removed_guids;
};

/** Bounded C9 v1 feed response. Never contains source bytes or credentials. */
struct remote_content_feed
{
    remote_content_feed_action action{remote_content_feed_action::invalid};
    std::uint64_t oldest_available{};
    std::uint64_t newest_available{};
    std::uint64_t current_generation{};
    remote_content_object current_release;
    std::vector<remote_content_feed_event> events;
};

struct [[nodiscard]] remote_content_feed_parse_result
{
    std::optional<remote_content_feed> feed;
    std::string error;
    [[nodiscard]] bool succeeded() const noexcept
    {
        return feed.has_value();
    }
};

/** Parse the complete dev change-feed response (maximum 8 MiB). Requires
 * exact target/build/session cursor compatibility, a consistent replay/reset
 * mode, contiguous events, increasing published generations and canonical
 * immutable release-descriptor hashes. A gap always requires full snapshot;
 * no client may accidentally apply an incomplete generation.
 *
 * The caller independently authenticates the transport/session. This parser
 * never performs I/O or modifies the active cooked catalog.
 */
[[nodiscard]] remote_content_feed_parse_result parse_remote_content_feed(std::string_view json_bytes,
                                                                         const remote_content_expectations& expected,
                                                                         std::uint64_t last_acknowledged);

} // namespace arc::assets
