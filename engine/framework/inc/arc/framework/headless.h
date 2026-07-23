#pragma once

#include <arc/framework/application.h>
#include <arc/framework/runtime_world.h>

#include <cstdint>
#include <string>

namespace arc
{

struct headless_runtime_options
{
    std::uint64_t maximum_ticks{};
    std::uint64_t seed{ 0x4152435f53455256ull };
    runtime_world_role default_world_role{ runtime_world_role::server };
    bool sleep_to_clock{ true };
    bool enable_debug_time_controls{};
};

struct headless_runtime_result
{
    bool succeeded{};
    std::uint64_t completed_ticks{};
    std::string error;
};

/** Run an application without a window, input device, renderer, or platform host. */
headless_runtime_result run_headless(application& app, headless_runtime_options options = {});

} // namespace arc
