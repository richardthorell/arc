#pragma once

#include <arc/input/input.h>

#include <cstdint>

namespace arc::input
{

/**
 * @brief Platform-neutral scalar controls derived from the primary touch contact.
 *
 * The primary contact is the active contact with the lowest runtime contact ID.
 * Positions and pressure are normalized to [0, 1]. Pressure is zero when the
 * backend does not expose it.
 */
enum class touch_control : std::uint8_t
{
    primary_down,
    primary_x,
    primary_y,
    primary_pressure
};

[[nodiscard]] constexpr input_control make_touch_control(touch_control value) noexcept
{
    return {.kind = input_control_kind::touch, .code = static_cast<std::uint16_t>(value)};
}

} // namespace arc::input
