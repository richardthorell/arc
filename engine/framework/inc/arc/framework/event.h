#pragma once

#include <cstdint>

namespace arc
{

/**
 * @brief High-level event categories emitted by platform hosts.
 */
enum class event_type : std::uint8_t
{
    none,
    close_requested,
    resized,
    key_down,
    key_up,
    mouse_button_down,
    mouse_button_up,
    mouse_moved,
    mouse_wheel,
    focus_gained,
    focus_lost
};

/**
 * @brief Mouse buttons reported by platform hosts.
 */
enum class mouse_button : std::uint8_t
{
    unknown,
    left,
    right,
    middle,
    x1,
    x2
};

/**
 * @brief Platform-neutral event payload.
 *
 * Fields are populated according to `type`. Unused fields remain zero/default.
 */
struct event
{
    event_type type{ event_type::none };
    std::uint32_t width{};
    std::uint32_t height{};
    int key_code{};
    std::uint32_t modifiers{};
    mouse_button button{ mouse_button::unknown };
    int x{};
    int y{};
    float wheel_delta{};
    bool repeat{};
};

} // namespace arc
