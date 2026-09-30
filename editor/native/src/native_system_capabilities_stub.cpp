#include "native_system_capabilities.h"

#include <thread>

namespace arc::editor
{

framework::platform_capabilities query_native_system_capabilities() noexcept
{
    return {.logical_processor_count = std::thread::hardware_concurrency(), .persistent_local_storage = true};
}

} // namespace arc::editor
