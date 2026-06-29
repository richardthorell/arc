#pragma once

#include <arc/framework/event.h>
#include <arc/jobs/jobs.h>
#include <arc/diagnostics/log.h>
#include <arc/memory/memory.h>

#include <memory>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace arc
{

struct frame_time;

/**
 * @brief Services exposed to engine modules during lifecycle callbacks.
 */
class module_context
{
public:
    module_context(job_system& jobs, logger& diagnostics, tracked_memory_resource& memory) noexcept;

    /**
     * @brief Return the shared engine job system.
     */
    job_system& jobs() const noexcept;

    /**
     * @brief Return the shared diagnostics logger.
     */
    logger& diagnostics() const noexcept;

    /**
     * @brief Return the shared tracked memory resource.
     */
    tracked_memory_resource& memory() const noexcept;

private:
    job_system* jobs_{};
    logger* diagnostics_{};
    tracked_memory_resource* memory_{};
};

/**
 * @brief Base class for engine systems managed by the runtime.
 */
class module
{
public:
    virtual ~module();

    /**
     * @brief Return the unique runtime name for this module.
     */
    virtual std::string_view name() const = 0;

    /**
     * @brief Return module names that must start before this module.
     */
    virtual std::vector<std::string> dependencies() const;

    /**
     * @brief Called once when the runtime starts modules.
     */
    virtual void on_start(module_context& context);

    /**
     * @brief Called once per runtime frame.
     */
    virtual void on_update(module_context& context, const frame_time& time);

    /**
     * @brief Called when the runtime dispatches a platform-neutral event.
     */
    virtual void on_event(module_context& context, const event& value);

    /**
     * @brief Called once when the runtime shuts modules down.
     */
    virtual void on_shutdown(module_context& context);
};

/**
 * @brief Mutable collection used by applications to register modules.
 */
class module_registry
{
public:
    /**
     * @brief Add a module instance owned by the registry.
     */
    void add(std::unique_ptr<module> value);

    /**
     * @brief Construct and add a module in-place.
     */
    template <class Module, class... Args>
    Module& emplace(Args&&... args)
    {
        auto value = std::make_unique<Module>(std::forward<Args>(args)...);
        Module& reference = *value;
        add(std::move(value));
        return reference;
    }

    /**
     * @brief Return the number of registered modules.
     */
    std::size_t size() const noexcept;

    /**
     * @brief Return whether no modules have been registered.
     */
    bool empty() const noexcept;

    /**
     * @brief Return all registered modules.
     */
    const std::vector<std::unique_ptr<module>>& modules() const noexcept;

private:
    std::vector<std::unique_ptr<module>> modules_;
};

/**
 * @brief Orders and drives registered modules through runtime lifecycle callbacks.
 */
class module_manager
{
public:
    /**
     * @brief Return the mutable module registry.
     */
    module_registry& registry() noexcept;

    /**
     * @brief Return the immutable module registry.
     */
    const module_registry& registry() const noexcept;

    /**
     * @brief Start modules in dependency order.
     */
    void start(module_context& context);

    /**
     * @brief Update started modules in dependency order.
     */
    void update(module_context& context, const frame_time& time);

    /**
     * @brief Dispatch an event to started modules in dependency order.
     */
    void dispatch(module_context& context, const event& value);

    /**
     * @brief Shut down started modules in reverse dependency order.
     */
    void shutdown(module_context& context);

    /**
     * @brief Return whether modules are currently started.
     */
    bool started() const noexcept;

    /**
     * @brief Return module names in resolved dependency order.
     */
    std::vector<std::string_view> start_order() const;

private:
    void resolve_order();

    module_registry registry_;
    std::vector<std::size_t> order_;
    bool ordered_{};
    bool started_{};
};

} // namespace arc
