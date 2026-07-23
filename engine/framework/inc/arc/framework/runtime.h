#pragma once

#include <arc/framework/application.h>
#include <arc/jobs/jobs.h>
#include <arc/framework/module.h>
#include <arc/framework/runtime_world.h>
#include <arc/framework/service.h>

#include <chrono>
#include <vector>

namespace arc
{

/**
 * @brief Platform-neutral application runtime.
 *
 * The runtime owns lifecycle ordering, frame timing, config normalization, and
 * event dispatch. Platform entry code owns windows and operating-system loops.
 */
class runtime
{
public:
    explicit runtime(application& app);
    runtime(application& app, application_config config);
    ~runtime();

    /**
     * @brief Normalize user-provided configuration into usable defaults.
     */
    static application_config normalize_config(application_config config);

    /**
     * @brief Start the application lifecycle if it has not already started.
     */
    void start();

    /**
     * @brief Advance one frame and call `application::on_update`.
     */
    frame_time tick();

    /**
     * @brief Advance one outer frame by an explicit wall-clock delta.
     *
     * This is the deterministic entry point used by headless hosts and tests.
     */
    frame_time advance(double wall_delta_seconds);

    /**
     * @brief Dispatch a platform-neutral event to the application.
     */
    void dispatch(const event& value);

    /**
     * @brief Request that the runtime stop after the current platform loop iteration.
     */
    void request_stop() noexcept;

    /**
     * @brief Shut down the application lifecycle if it is active.
     */
    void shutdown();

    /**
     * @brief Return whether the runtime should continue ticking.
     */
    bool running() const noexcept;

    /**
     * @brief Return whether the lifecycle has been started.
     */
    bool started() const noexcept;

    /**
     * @brief Return normalized application configuration.
     */
    const application_config& config() const noexcept;

    /**
     * @brief Return the shared runtime job system.
     */
    job_system& jobs() noexcept;

    /**
     * @brief Return the runtime-owned memory service.
     */
    memory_system& memory() noexcept;

    /**
     * @brief Return transient CPU arenas with frame and tick lifetimes.
     */
    frame_arena& frame_memory() noexcept;
    tick_arena& tick_memory() noexcept;

    /**
     * @brief Return the shared runtime module manager.
     */
    module_manager& modules() noexcept;
    runtime_service_registry& services() noexcept;
    runtime_world_manager& worlds() noexcept;
    const runtime_world_manager& worlds() const noexcept;

    void pause() noexcept;
    void resume() noexcept;
    bool paused() const noexcept;
    bool step(std::uint32_t ticks = 1) noexcept;
    bool set_time_scale(double value) noexcept;
    double time_scale() const noexcept;
    simulation_tick current_tick() const noexcept;
    std::uint64_t discarded_ticks() const noexcept;

    world_snapshot_result capture_snapshot(runtime_world_id world, std::string label = {});
    world_snapshot_result restore_snapshot(world_snapshot_id snapshot);

private:
    using clock = std::chrono::steady_clock;

    application* app_{};
    application_config config_{};
    memory_system memory_{};
    system_memory_resource frame_memory_resource_{ memory_, memory_domain::frame, make_memory_tag("runtime.frame") };
    system_memory_resource tick_memory_resource_{ memory_, memory_domain::tick, make_memory_tag("runtime.tick") };
    frame_arena frame_arena_{ 256u * 1024u, &frame_memory_resource_ };
    tick_arena tick_arena_{ 128u * 1024u, &tick_memory_resource_ };
    job_system jobs_{};
    runtime_service_registry services_;
    runtime_world_manager worlds_{ memory_ };
    module_context module_context_;
    module_manager modules_;
    bool modules_registered_{};
    bool services_registered_{};
    bool worlds_registered_{};
    bool started_{};
    bool running_{};
    bool paused_{};
    std::uint32_t pending_steps_{};
    double accumulator_seconds_{};
    double explicit_total_seconds_{};
    std::uint64_t discarded_ticks_{};
    simulation_tick current_tick_{};
    clock::time_point start_time_{};
    clock::time_point last_frame_time_{};
    frame_time current_time_{};
    std::vector<simulation_input_command> pending_input_;
    std::vector<simulation_input_command> sampled_input_;
    std::uint64_t input_revision_{};
};

} // namespace arc
