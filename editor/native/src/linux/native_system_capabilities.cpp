#include "../native_system_capabilities.h"

#include <unistd.h>

#include <cstdint>
#include <thread>

namespace arc::editor
{

framework::platform_capabilities query_native_system_capabilities() noexcept
{
    const auto physical_pages = sysconf(_SC_PHYS_PAGES);
    const auto page_size = sysconf(_SC_PAGE_SIZE);
    const auto system_memory_bytes = physical_pages > 0 && page_size > 0 ? static_cast<std::uint64_t>(physical_pages) *
                                                                               static_cast<std::uint64_t>(page_size)
                                                                         : 0u;
    return {.family = framework::platform_family::linux_os,
            .form_factor = framework::device_form_factor::desktop,
            .logical_processor_count = std::thread::hardware_concurrency(),
            .system_memory_bytes = system_memory_bytes,
            .window_system = true,
            .high_dpi = true,
            .multiple_windows = true,
            .dynamic_libraries = true,
            .persistent_local_storage = true,
            .native_package_assets = false};
}

} // namespace arc::editor
