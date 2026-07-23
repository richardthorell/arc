#pragma once

#include <arc/framework/event.h>
#include <arc/framework/simulation.h>

#include <cstdint>
#include <memory>
#include <string>

namespace arc
{

class module_registry;
class runtime_service_registry;
class runtime_world_manager;

/**
 * @brief Startup configuration requested by an application.
 */
struct application_config
{
    std::string title{ "ARC Application" };
    std::uint32_t initial_width{ 1280 };
    std::uint32_t initial_height{ 720 };
    bool resizable{ true };
    bool visible{ true };
    bool start_focused{ true };
    bool maximized{ false };
    simulation_config simulation{};
};

/**
 * @brief Timing data passed to application update callbacks.
 */
struct frame_time
{
    double delta_seconds{};
    double total_seconds{};
    std::uint64_t frame_index{};
    simulation_tick_id last_completed_tick{};
    std::uint32_t completed_ticks{};
    double interpolation_alpha{};
    std::uint64_t discarded_ticks{};
};

/**
 * @brief Base class implemented by engine applications, tools, and samples.
 */
class application
{
public:
    virtual ~application();

    /**
     * @brief Return the desired startup configuration.
     */
    virtual application_config configure() const;

    /**
     * @brief Called once when the runtime starts.
     */
    virtual void on_start();

    /**
     * @brief Register runtime modules before startup.
     */
    virtual void register_modules(module_registry& registry);

    /**
     * @brief Register lifecycle services before modules and worlds start.
     */
    virtual void register_services(runtime_service_registry& services);

    /**
     * @brief Describe simulation worlds before module startup.
     */
    virtual void register_worlds(runtime_world_manager& worlds);

    /**
     * @brief Called once per frame while the runtime is running.
     */
    virtual void on_update(const frame_time& time);

    /**
     * @brief Called when the platform host emits an event.
     */
    virtual void on_event(const event& value);

    /**
     * @brief Called once when the runtime shuts down.
     */
    virtual void on_shutdown();
};

using application_ptr = std::unique_ptr<application>;

/**
 * @brief Factory implemented by editor, sample, or game executable code.
 */
application_ptr create_application();

} // namespace arc
