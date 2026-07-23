#pragma once

#include <arc/ecs/simulation.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>

namespace arc
{

using simulation_tick_id = ecs::simulation_tick_id;
using runtime_world_role = ecs::runtime_world_role;
using runtime_service_id = ecs::runtime_service_id;
using random_stream_id = ecs::random_stream_id;
using random_stream = ecs::random_stream;
using simulation_input_kind = ecs::simulation_input_kind;
using simulation_input_action = ecs::simulation_input_action;
using simulation_input_command = ecs::simulation_input_command;
using simulation_input_snapshot = ecs::simulation_input_snapshot;

enum class simulation_overrun_policy : std::uint8_t
{
    discard_excess,
    preserve_debt
};

struct simulation_config
{
    double fixed_tick_rate{ 60.0 };
    double maximum_frame_delta_seconds{ 0.25 };
    std::uint32_t maximum_catch_up_ticks{ 8 };
    double time_scale{ 1.0 };
    std::uint64_t process_seed{ 0x4152435f53454544ull };
    simulation_overrun_policy overrun_policy{ simulation_overrun_policy::discard_excess };
    std::size_t snapshot_budget_bytes{};
    bool headless{};
    bool presentation_enabled{ true };
    bool allow_headless_time_controls{};
    runtime_world_role default_world_role{ runtime_world_role::client };

    double fixed_delta_seconds() const noexcept
    {
        return fixed_tick_rate > 0.0 ? 1.0 / fixed_tick_rate : 1.0 / 60.0;
    }
};

struct simulation_tick
{
    simulation_tick_id id{};
    double delta_seconds{ 1.0 / 60.0 };
    double total_seconds{};
};

inline bool valid_simulation_config(const simulation_config& value) noexcept
{
    return std::isfinite(value.fixed_tick_rate) &&
        value.fixed_tick_rate >= 1.0 &&
        value.fixed_tick_rate <= 1000.0 &&
        std::isfinite(value.maximum_frame_delta_seconds) &&
        value.maximum_frame_delta_seconds > 0.0 &&
        value.maximum_catch_up_ticks > 0 &&
        std::isfinite(value.time_scale) &&
        value.time_scale >= 0.0 &&
        value.time_scale <= 16.0;
}

} // namespace arc
