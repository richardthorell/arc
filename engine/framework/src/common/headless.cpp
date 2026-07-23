#include <arc/framework/headless.h>

#include <arc/framework/runtime.h>

#include <chrono>
#include <exception>
#include <thread>

namespace arc
{

headless_runtime_result run_headless(application& app, headless_runtime_options options)
{
    application_config config = app.configure();
    config.visible = false;
    config.start_focused = false;
    config.simulation.headless = true;
    config.simulation.presentation_enabled = false;
    config.simulation.process_seed = options.seed;
    config.simulation.overrun_policy = simulation_overrun_policy::preserve_debt;
    config.simulation.allow_headless_time_controls = options.enable_debug_time_controls;
    config.simulation.time_scale = 1.0;
    config.simulation.default_world_role = options.default_world_role;

    runtime host(app, std::move(config));

    try
    {
        host.start();
        const double fixed_delta = host.config().simulation.fixed_delta_seconds();
        using clock = std::chrono::steady_clock;
        auto deadline = clock::now();

        while (host.running() &&
            (options.maximum_ticks == 0 ||
                host.current_tick().id.value < options.maximum_ticks))
        {
            host.advance(fixed_delta);
            if (options.sleep_to_clock)
            {
                deadline += std::chrono::duration_cast<clock::duration>(
                    std::chrono::duration<double>(fixed_delta));
                std::this_thread::sleep_until(deadline);
            }
        }

        const std::uint64_t completed = host.current_tick().id.value;
        std::string world_failure;
        for (const runtime_world_id id : host.worlds().ordered_worlds())
        {
            const runtime_world* world = host.worlds().find(id);
            if (world && world->state() == runtime_world_state::faulted)
            {
                world_failure = std::string(world->name()) + ": " + world->fault_message();
                break;
            }
        }
        host.shutdown();
        return {
            .succeeded = world_failure.empty(),
            .completed_ticks = completed,
            .error = std::move(world_failure)
        };
    }
    catch (const std::exception& error)
    {
        const std::string message = error.what();
        const std::uint64_t completed = host.current_tick().id.value;
        try
        {
            host.shutdown();
        }
        catch (...)
        {
            // Preserve the failure that stopped the headless loop.
        }
        return { false, completed, message };
    }
    catch (...)
    {
        const std::uint64_t completed = host.current_tick().id.value;
        try
        {
            host.shutdown();
        }
        catch (...)
        {
        }
        return { false, completed, "unknown headless runtime failure" };
    }
}

} // namespace arc
