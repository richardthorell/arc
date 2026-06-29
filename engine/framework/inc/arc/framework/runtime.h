#pragma once

#include <arc/framework/application.h>
#include <arc/jobs/jobs.h>
#include <arc/framework/module.h>

#include <chrono>

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
     * @brief Return the shared runtime module manager.
     */
    module_manager& modules() noexcept;

private:
    using clock = std::chrono::steady_clock;

    application* app_{};
    application_config config_{};
    job_system jobs_{};
    module_context module_context_;
    module_manager modules_;
    bool modules_registered_{};
    bool started_{};
    bool running_{};
    clock::time_point start_time_{};
    clock::time_point last_frame_time_{};
    frame_time current_time_{};
};

} // namespace arc
