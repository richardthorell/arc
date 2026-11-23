#pragma once

/** L1.5 Development-only inspection messages and safe-frame mailbox.
 * No ECS types or world pointers cross the socket/runtime thread boundary.
 */
#include <cstdint>
#include <deque>
#include <mutex>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace arc::network::live_play_dev
{
struct inspection_entity
{
    bool authored{};
    std::string id;
    friend bool operator==(const inspection_entity&, const inspection_entity&) = default;
};

struct hierarchy_item
{
    inspection_entity entity;
    std::optional<inspection_entity> parent;
    std::string name;
    friend bool operator==(const hierarchy_item&, const hierarchy_item&) = default;
};

struct hierarchy_snapshot
{
    std::uint64_t world_revision{};
    std::vector<hierarchy_item> entities;
    bool complete{true};
};

enum class hierarchy_change_kind : std::uint8_t
{
    upsert,
    removed
};

struct hierarchy_change
{
    hierarchy_change_kind kind{hierarchy_change_kind::upsert};
    hierarchy_item item;
};

struct hierarchy_delta
{
    std::uint64_t base_revision{};
    std::uint64_t world_revision{};
    std::vector<hierarchy_change> changes;
};

/** Returns null when a bounded, coherent delta is unavailable; sender must
 * transmit a complete paged snapshot instead. No ECS access or sockets here.
 */
[[nodiscard]] std::optional<hierarchy_delta> diff_hierarchy(const hierarchy_snapshot& baseline,
                                                            const hierarchy_snapshot& current);

struct inspection_request
{
    std::string request_id;
    inspection_entity entity;
    std::string component;
    std::uint32_t schema_version{};
};

enum class inspection_value_kind : std::uint8_t
{
    numeric_array,
    text,
    boolean
};

struct inspection_property
{
    std::string path;
    inspection_value_kind kind{inspection_value_kind::numeric_array};
    std::vector<float> numbers;
    std::string text;
    bool flag{};
};

struct component_inspection
{
    inspection_request request;
    std::uint64_t runtime_revision{};
    std::vector<inspection_property> properties;
};

struct inspection_outcome
{
    std::string request_id;
    std::optional<component_inspection> result;
    std::string error_code;
};

/** Strict v1 editor inspect request; payload is validated before it reaches ECS. */
[[nodiscard]] std::optional<inspection_request> decode_inspection_request(std::string_view request_id,
                                                                          std::string_view payload_json);

/** Bounded queue from authenticated socket worker to host frame thread and back. */
class inspection_mailbox final
{
public:
    [[nodiscard]] bool enqueue(inspection_request request);
    [[nodiscard]] std::vector<inspection_request> take_requests();
    [[nodiscard]] bool report(inspection_outcome result);
    [[nodiscard]] std::vector<inspection_outcome> take_results();
    /** Coalesce hierarchy updates: only latest immutable snapshot survives. */
    void publish(hierarchy_snapshot snapshot);
    [[nodiscard]] std::optional<hierarchy_snapshot> take_snapshot();
    void clear();

private:
    std::mutex mutex_;
    std::deque<inspection_request> requests_;
    std::deque<inspection_outcome> results_;
    std::optional<hierarchy_snapshot> snapshot_;
};
} // namespace arc::network::live_play_dev
