#pragma once

#include <cstdint>

namespace arc::framework
{

/** @brief Operating-system family reported for diagnostics and platform integration. */
enum class platform_family : std::uint8_t
{
    unknown,
    windows,
    android,
    linux,
    macos,
    ios
};

/** @brief Physical form factor used as one input to device-profile resolution. */
enum class device_form_factor : std::uint8_t
{
    unknown,
    desktop,
    handheld,
    tablet,
    console,
    server
};

/**
 * @brief Backend-neutral capabilities supplied by the active platform host.
 *
 * Platform identity is diagnostic only. High-level systems select behavior from
 * the capability flags and limits rather than branching on platform_family.
 */
struct platform_capabilities
{
    platform_family family{platform_family::unknown};
    device_form_factor form_factor{device_form_factor::unknown};
    std::uint32_t logical_processor_count{};
    std::uint64_t system_memory_bytes{};
    bool window_system{};
    bool high_dpi{};
    bool multiple_windows{};
    bool dynamic_libraries{};
    bool persistent_local_storage{};
    bool native_package_assets{};
};

} // namespace arc::framework
