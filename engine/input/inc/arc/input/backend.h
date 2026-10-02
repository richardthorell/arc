#pragma once

#include <arc/input/input.h>

#include <cstdint>
#include <string_view>

namespace arc::input
{

/**
 * @brief Platform family that owns an input backend implementation.
 *
 * This identifies backend availability for diagnostics and platform startup only.
 * Gameplay remains coupled to ARC semantic input rather than platform families.
 */
enum class input_platform : std::uint8_t
{
    unknown,
    windows,
    android,
    linux,
    macos,
    ios
};

/**
 * @brief Device families and optional services implemented by a platform backend.
 *
 * Unsupported capabilities remain false so callers never need to infer support
 * from the target platform. Device-specific capabilities are still reported by
 * input_device_descriptor after a device is connected.
 */
struct input_backend_capabilities
{
    bool keyboard{};
    bool mouse{};
    bool gamepad{};
    bool touch{};
    bool pen{};
    bool wheel{};
    bool flight_stick{};
    bool motion_controller{};
    bool motion_sensors{};
    bool battery{};
    bool output{};
};

/**
 * @brief Platform-neutral description supplied by a native input backend.
 */
struct input_backend_descriptor
{
    input_platform platform{input_platform::unknown};
    input_backend_type backend{input_backend_type::unknown};
    std::string_view name;
    input_backend_capabilities capabilities{};
};

/** @brief Return whether a backend explicitly advertises a device family. */
[[nodiscard]] constexpr bool supports_device(const input_backend_capabilities& capabilities,
                                             input_device_type type) noexcept
{
    switch (type)
    {
    case input_device_type::keyboard: return capabilities.keyboard;
    case input_device_type::mouse: return capabilities.mouse;
    case input_device_type::gamepad: return capabilities.gamepad;
    case input_device_type::touch: return capabilities.touch;
    case input_device_type::pen: return capabilities.pen;
    case input_device_type::wheel: return capabilities.wheel;
    case input_device_type::flight_stick: return capabilities.flight_stick;
    case input_device_type::motion_controller: return capabilities.motion_controller;
    case input_device_type::unknown: return false;
    }
    return false;
}

} // namespace arc::input
