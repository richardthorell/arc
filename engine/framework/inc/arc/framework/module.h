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

namespace arc::framework
{

struct frame_time;
class runtime_service_registry;
class runtime_world_manager;

/**
 * @brief Services exposed to engine modules during lifecycle callbacks.
 */
class module_context
{
public:
    module_context(jobs::job_system& jobs, diagnostics::logger& diagnostics,
                   memory::tracked_memory_resource& memory) noexcept;
    module_context(jobs::job_system& jobs, diagnostics::logger& diagnostics, memory::memory_system& memory,
                   memory::tracked_memory_resource& compatibility_memory, runtime_service_registry* services = nullptr,
                   runtime_world_manager* worlds = nullptr) noexcept;

    /**
     * @brief Return the shared engine job system.
     */
    [[nodiscard]] jobs::job_system& jobs() const noexcept;

    /**
     * @brief Return the shared diagnostics diagnostics::logger.
     */
    [[nodiscard]] diagnostics::logger& diagnostics() const noexcept;

    /**
     * @brief Return the shared tracked memory resource.
     */
    [[nodiscard]] memory::tracked_memory_resource& memory() const noexcept;

    /**
     * @brief Return the engine memory service used for budgets, tags, and arenas.
     */
    [[nodiscard]] memory::memory_system& memory_service() const noexcept;
    [[nodiscard]] runtime_service_registry* services() const noexcept;
    [[nodiscard]] runtime_world_manager* worlds() const noexcept;

private:
    jobs::job_system* jobs_{};
    diagnostics::logger* diagnostics_{};
    memory::tracked_memory_resource* memory_{};
    memory::memory_system* memory_service_{};
    runtime_service_registry* services_{};
    runtime_world_manager* worlds_{};
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
    [[nodiscard]] virtual std::string_view name() const = 0;

    /**
     * @brief Return module names that must start before this module.
     */
    [[nodiscard]] virtual std::vector<std::string> dependencies() const;

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
    template <class Module, class... Args> Module& emplace(Args&&... args)
    {
        auto value = std::make_unique<Module>(std::forward<Args>(args)...);
        Module& reference = *value;
        add(std::move(value));
        return reference;
    }

    /**
     * @brief Return the number of registered modules.
     */
    [[nodiscard]] std::size_t size() const noexcept;

    /**
     * @brief Return whether no modules have been registered.
     */
    [[nodiscard]] bool empty() const noexcept;

    /**
     * @brief Return all registered modules.
     */
    [[nodiscard]] const std::vector<std::unique_ptr<module>>& modules() const noexcept;

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
    [[nodiscard]] module_registry& registry() noexcept;

    /**
     * @brief Return the immutable module registry.
     */
    [[nodiscard]] const module_registry& registry() const noexcept;

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
    [[nodiscard]] bool started() const noexcept;

    /**
     * @brief Return module names in resolved dependency order.
     */
    [[nodiscard]] std::vector<std::string_view> start_order() const;

private:
    void resolve_order();

    module_registry registry_;
    std::vector<std::size_t> order_;
    bool ordered_{};
    bool started_{};
};

} // namespace arc::framework
