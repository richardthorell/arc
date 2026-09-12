#pragma once

#include <arc/input/input.h>

#include <cstdint>

namespace arc::input
{

/**
 * @brief Layout-neutral gamepad buttons.
 *
 * Face buttons use physical positions instead of platform labels so mappings
 * remain portable between Xbox, PlayStation, Nintendo, and generic controllers.
 * Extended buttons are only reported by devices that physically expose them.
 */
enum class gamepad_button : std::uint8_t
{
    south,
    east,
    west,
    north,
    auxiliary_1,
    auxiliary_2,
    dpad_up,
    dpad_down,
    dpad_left,
    dpad_right,
    left_shoulder,
    right_shoulder,
    left_trigger_button,
    right_trigger_button,
    left_stick,
    right_stick,
    left_stick_up,
    left_stick_down,
    left_stick_left,
    left_stick_right,
    right_stick_up,
    right_stick_down,
    right_stick_left,
    right_stick_right,
    paddle_left_1,
    paddle_left_2,
    paddle_right_1,
    paddle_right_2,
    view,
    menu,
    guide,
    share
};

/**
 * @brief Normalized scalar gamepad axes.
 *
 * Stick axes use [-1, 1]. Triggers use [0, 1].
 */
enum class gamepad_axis : std::uint8_t
{
    left_x,
    left_y,
    right_x,
    right_y,
    left_trigger,
    right_trigger
};

[[nodiscard]] constexpr input_control make_gamepad_button_control(gamepad_button value) noexcept
{
    return {.kind = input_control_kind::gamepad_button, .code = static_cast<std::uint16_t>(value)};
}

[[nodiscard]] constexpr input_control make_gamepad_axis_control(gamepad_axis value) noexcept
{
    return {.kind = input_control_kind::gamepad_axis, .code = static_cast<std::uint16_t>(value)};
}

} // namespace arc::input
