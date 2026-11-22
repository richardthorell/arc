#pragma once

/** Development-only Live Play property edit transport and thread handoff.
 * The game loop owns ECS application; the socket worker owns all client I/O.
 */
#include <array>
#include <cstdint>
#include <deque>
#include <mutex>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace arc::network::live_play_dev
{

struct property_edit
{
    std::string request_id;
    std::string entity_guid;
    std::string component;
    std::string property;
    std::string phase;
    std::uint32_t schema_version{};
    std::uint64_t authoring_revision{};
    std::uint64_t expected_runtime_revision{};
    std::array<float, 4> values{};
    std::uint8_t value_count{};
};

struct edit_result
{
    std::string request_id;
    bool applied{};
    std::uint64_t applied_revision{};
    std::string code;
    std::string message;
};

/** Parse strictly the L1.1 entity.property.set payload; no ECS effects. */
[[nodiscard]] std::optional<property_edit> decode_property_edit(std::string_view request_id,
                                                                std::string_view payload_json);

/** Bounded network-worker <-> runtime-frame handoff. Never owns an ECS world. */
class edit_mailbox final
{
public:
    [[nodiscard]] bool enqueue(property_edit edit);
    [[nodiscard]] std::vector<property_edit> take_edits();
    [[nodiscard]] bool report(edit_result result);
    [[nodiscard]] std::vector<edit_result> take_results();
    void clear();

private:
    static constexpr std::size_t maximum_pending_edits = 256;
    static constexpr std::size_t maximum_results = 512;
    std::mutex mutex_;
    std::deque<property_edit> pending_;
    std::deque<edit_result> results_;
};

} // namespace arc::network::live_play_dev
