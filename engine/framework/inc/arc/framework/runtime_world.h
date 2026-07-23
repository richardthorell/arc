#pragma once

#include <arc/ecs/ecs.h>
#include <arc/framework/service.h>
#include <arc/framework/simulation.h>

#include <cstddef>
#include <compare>
#include <cstdint>
#include <deque>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace arc
{

struct runtime_world_id
{
    std::uint64_t value{};

    constexpr bool valid() const noexcept { return value != 0; }
    friend constexpr bool operator==(runtime_world_id, runtime_world_id) noexcept = default;
    friend constexpr auto operator<=>(runtime_world_id, runtime_world_id) noexcept = default;
};

enum class runtime_world_state : std::uint8_t
{
    stopped,
    starting,
    running,
    paused,
    faulted,
    stopping
};

struct runtime_world_descriptor
{
    std::string name{ "World" };
    runtime_world_role role{ runtime_world_role::client };
    std::uint64_t seed{};
    memory_budget memory{};
    bool install_placeholder_systems{ true };
    bool presentation_enabled{ true };
};

struct world_snapshot_id
{
    std::uint64_t value{};

    constexpr bool valid() const noexcept { return value != 0; }
    friend constexpr bool operator==(world_snapshot_id, world_snapshot_id) noexcept = default;
};

struct world_snapshot_metadata
{
    world_snapshot_id id{};
    runtime_world_id world{};
    simulation_tick_id tick{};
    double simulation_time_seconds{};
    std::uint64_t world_epoch{};
    std::size_t estimated_bytes{};
    std::string label;
};

struct world_snapshot_result
{
    bool succeeded{};
    world_snapshot_metadata metadata;
    std::string error;
};

struct runtime_world_run_result
{
    std::size_t systems_executed{};
    std::vector<ecs::system_schedule_error> errors;

    bool succeeded() const noexcept { return errors.empty(); }
};

class runtime_world
{
public:
    runtime_world(memory_system& memory, runtime_world_id id, runtime_world_descriptor descriptor);

    runtime_world_id id() const noexcept { return id_; }
    std::string_view name() const noexcept { return descriptor_.name; }
    runtime_world_role role() const noexcept { return descriptor_.role; }
    runtime_world_state state() const noexcept { return state_; }
    std::uint64_t seed() const noexcept { return descriptor_.seed; }
    std::uint64_t epoch() const noexcept { return epoch_; }
    simulation_tick_id last_completed_tick() const noexcept { return last_completed_tick_; }
    const std::string& fault_message() const noexcept { return fault_message_; }

    ecs::world& entities() noexcept { return *entities_; }
    const ecs::world& entities() const noexcept { return *entities_; }
    ecs::system_scheduler& systems() noexcept { return systems_; }
    const ecs::system_scheduler& systems() const noexcept { return systems_; }
    ecs::world_partition& partition() noexcept { return partition_; }
    const ecs::world_partition& partition() const noexcept { return partition_; }

    void start();
    void stop() noexcept;
    void pause() noexcept;
    void resume() noexcept;
    void clear_fault() noexcept;
    bool attach_entities(ecs::world& entities) noexcept;

    runtime_world_run_result run_fixed(
        job_system& jobs,
        const simulation_tick& tick,
        runtime_service_provider* services,
        const simulation_input_snapshot* input,
        tick_arena& tick_memory,
        frame_arena& frame_memory,
        std::uint64_t process_seed);
    runtime_world_run_result run_presentation(
        job_system& jobs,
        const simulation_tick& tick,
        float frame_delta_seconds,
        float interpolation_alpha,
        runtime_service_provider* services,
        const simulation_input_snapshot* input,
        tick_arena& tick_memory,
        frame_arena& frame_memory,
        std::uint64_t process_seed);

private:
    friend class runtime_world_manager;
    void install_placeholder_systems();
    runtime_world_run_result run_phase(
        job_system& jobs,
        ecs::system_phase phase,
        const ecs::system_execution_info& execution);
    void fault(std::string message);

    runtime_world_id id_{};
    runtime_world_descriptor descriptor_;
    runtime_world_state state_{ runtime_world_state::stopped };
    ecs::world owned_entities_;
    ecs::world* entities_{};
    ecs::system_scheduler systems_;
    ecs::world_partition partition_;
    simulation_tick_id last_completed_tick_{};
    std::uint64_t epoch_{ 1 };
    std::string fault_message_;
};

class runtime_world_manager
{
public:
    explicit runtime_world_manager(memory_system& memory);

    runtime_world& create(runtime_world_descriptor descriptor = {});
    bool destroy(runtime_world_id id);
    runtime_world* find(runtime_world_id id) noexcept;
    const runtime_world* find(runtime_world_id id) const noexcept;
    std::vector<runtime_world_id> ordered_worlds() const;
    std::size_t size() const noexcept { return worlds_.size(); }
    bool executing() const noexcept { return executing_; }

    runtime_world_run_result run_fixed(
        job_system& jobs,
        const simulation_tick& tick,
        runtime_service_provider* services,
        const simulation_input_snapshot* input,
        tick_arena& tick_memory,
        frame_arena& frame_memory,
        std::uint64_t process_seed);
    runtime_world_run_result run_presentation(
        job_system& jobs,
        const simulation_tick& tick,
        float frame_delta_seconds,
        float interpolation_alpha,
        runtime_service_provider* services,
        const simulation_input_snapshot* input,
        tick_arena& tick_memory,
        frame_arena& frame_memory,
        std::uint64_t process_seed);

    void start_all();
    void stop_all() noexcept;
    void pause_all() noexcept;
    void resume_all() noexcept;

    void set_snapshot_budget(std::size_t bytes) noexcept;
    world_snapshot_result capture_snapshot(
        runtime_world_id world,
        simulation_tick_id tick,
        double simulation_time_seconds = 0.0,
        std::string label = {},
        runtime_service_registry* services = nullptr);
    world_snapshot_result restore_snapshot(
        world_snapshot_id snapshot,
        runtime_service_registry* services = nullptr);
    std::vector<world_snapshot_metadata> snapshots() const;

private:
    struct stored_snapshot
    {
        world_snapshot_metadata metadata;
        runtime_world_descriptor descriptor;
        ecs::world entities;
        ecs::world_partition partition;
        std::vector<runtime_service_snapshot> services;
        simulation_tick_id last_completed_tick{};
    };

    const std::vector<runtime_world*>& execution_order() const noexcept;
    void rebuild_execution_order();
    void trim_snapshots();

    memory_system* memory_{};
    std::vector<std::unique_ptr<runtime_world>> worlds_;
    std::vector<runtime_world*> execution_order_;
    std::deque<stored_snapshot> snapshots_;
    std::uint64_t next_world_id_{ 1 };
    std::uint64_t next_snapshot_id_{ 1 };
    std::size_t snapshot_budget_bytes_{};
    std::size_t snapshot_bytes_{};
    bool executing_{};
};

} // namespace arc
