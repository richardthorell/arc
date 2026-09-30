#pragma once

#include <arc/framework/capabilities.h>

namespace arc::editor
{

[[nodiscard]] framework::platform_capabilities query_native_system_capabilities() noexcept;

} // namespace arc::editor
