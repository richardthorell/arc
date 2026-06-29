#pragma once

#include <arc/event.h>

#include <cstdint>
#include <memory>
#include <string>

namespace arc
{

class module_registry;

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
};

/**
 * @brief Timing data passed to application update callbacks.
 */
struct frame_time
{
    double delta_seconds{};
    double total_seconds{};
    std::uint64_t frame_index{};
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
