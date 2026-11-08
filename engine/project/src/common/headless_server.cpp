#include <arc/project/headless_server.h>

#include <chrono>
#include <exception>
#include <string>
#include <thread>
#include <utility>

namespace arc::project
{
namespace
{

project_error failed(std::string message)
{
    return {.code = project_error_code::process_failed, .field = "serverLoop", .message = std::move(message)};
}

} // namespace

packaged_headless_server_result run_packaged_headless_server(packaged_headless_server_options options)
{
    if (options.runtime.boot.role != runtime_launch_role::server ||
        options.runtime.boot.expected.renderer != "none")
        return packaged_headless_server_result::failure(
            {.code = project_error_code::invalid_descriptor,
             .field = "role",
             .message = "Packaged headless server requires server role and a compiled-in renderer expectation of none"});

    // Preserve simulated time debt rather than silently dropping authoritative
    // server ticks under load; never accept client/presentation time controls.
    auto& simulation = options.runtime.config.simulation;
    simulation.headless = true;
    simulation.presentation_enabled = false;
    simulation.overrun_policy = framework::simulation_overrun_policy::preserve_debt;
    simulation.allow_headless_time_controls = false;
    simulation.time_scale = 1.0;

    auto started = start_packaged_runtime(std::move(options.runtime));
    if (!started) return packaged_headless_server_result::failure(std::move(started).error());
    auto session = std::move(started).value();

    // Avoid reading the runtime pointer after shutdown or a fatal frame error.
    const double fixed_delta = session->runtime()->config().simulation.fixed_delta_seconds();
    using clock = std::chrono::steady_clock;
    auto deadline = clock::now();
    std::uint64_t completed_ticks{};

    auto fail = [&](project_error error) -> packaged_headless_server_result
    {
        static_cast<void>(session->shutdown());
        return packaged_headless_server_result::failure(std::move(error));
    };

    try
    {
        while (session->state() == packaged_runtime_state::running &&
               (options.maximum_ticks == 0 || completed_ticks < options.maximum_ticks))
        {
            if (options.stop_requested && options.stop_requested())
            {
                session->request_stop();
                break;
            }

            auto frame = session->advance(fixed_delta);
            if (!frame) return fail(std::move(frame).error());
            completed_ticks = session->runtime()->current_tick().id.value;

            if (options.sleep_to_clock)
            {
                deadline += std::chrono::duration_cast<clock::duration>(std::chrono::duration<double>(fixed_delta));
                std::this_thread::sleep_until(deadline);
            }
        }
    }
    catch (const std::exception& error)
    {
        return fail(failed("Packaged server loop failed: " + std::string(error.what())));
    }
    catch (...)
    {
        return fail(failed("Packaged server loop failed with an unknown exception"));
    }

    auto shutdown = session->shutdown();
    if (!shutdown) return packaged_headless_server_result::failure(std::move(shutdown).error());
    return packaged_headless_server_result::success(completed_ticks);
}

} // namespace arc::project
