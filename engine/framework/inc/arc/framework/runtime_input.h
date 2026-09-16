#pragma once

#include <arc/framework/service.h>

namespace arc::framework
{

/** Runtime service slot exposing the runtime-owned physical input system to ECS systems. */
inline constexpr runtime_service_id runtime_input_service_id = make_runtime_service_id("arc.framework.input");

} // namespace arc::framework
