#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif

#include "../native_system_capabilities.h"

#include <windows.h>

namespace arc::editor
{

framework::platform_capabilities query_native_system_capabilities() noexcept
{
    SYSTEM_INFO system{};
    GetNativeSystemInfo(&system);
    MEMORYSTATUSEX memory{};
    memory.dwLength = sizeof(memory);
    const bool memory_available = GlobalMemoryStatusEx(&memory) == TRUE;
    return {.family = framework::platform_family::windows,
            .form_factor = framework::device_form_factor::desktop,
            .logical_processor_count = system.dwNumberOfProcessors,
            .system_memory_bytes = memory_available ? memory.ullTotalPhys : 0u,
            .window_system = true,
            .high_dpi = true,
            .multiple_windows = true,
            .dynamic_libraries = true,
            .persistent_local_storage = true,
            .native_package_assets = false};
}

} // namespace arc::editor
