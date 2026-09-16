#include "native_viewport_controller.h"

#include <arc/editor/arc_host.h>
#include <arc/jobs/jobs.h>
#include <arc/memory/memory.h>
#include <arc/render/render.h>

#include <chrono>
#include <iostream>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <utility>
#include <variant>
#include <vector>

namespace
{

const char* job_priority_name(arc::jobs::job_priority value) noexcept
{
    switch (value)
    {
        case arc::jobs::job_priority::critical:
            return "critical";
        case arc::jobs::job_priority::high:
            return "high";
        case arc::jobs::job_priority::normal:
            return "normal";
        case arc::jobs::job_priority::low:
            return "low";
        case arc::jobs::job_priority::background:
            return "background";
        case arc::jobs::job_priority::count:
            break;
    }
    return "unknown";
}

const char* job_affinity_name(arc::jobs::job_affinity value) noexcept
{
    switch (value)
    {
        case arc::jobs::job_affinity::any_worker:
            return "worker";
        case arc::jobs::job_affinity::main_thread:
            return "main";
        case arc::jobs::job_affinity::render_thread:
            return "render";
        case arc::jobs::job_affinity::io_thread:
            return "io";
    }
    return "unknown";
}

const char* job_status_name(arc::jobs::job_status value) noexcept
{
    switch (value)
    {
        case arc::jobs::job_status::invalid:
            return "invalid";
        case arc::jobs::job_status::waiting_dependencies:
            return "dependencies";
        case arc::jobs::job_status::queued:
            return "queued";
        case arc::jobs::job_status::running:
            return "running";
        case arc::jobs::job_status::waiting_children:
            return "children";
        case arc::jobs::job_status::succeeded:
            return "succeeded";
        case arc::jobs::job_status::failed:
            return "failed";
        case arc::jobs::job_status::cancelled:
            return "cancelled";
    }
    return "unknown";
}

arc::editor::host_profiler_snapshot make_profiler_snapshot(const arc::jobs::job_system_snapshot& jobs,
                                                           const arc::memory::memory_snapshot& memory)
{
    arc::editor::host_profiler_snapshot result;
    result.timestamp_nanoseconds = static_cast<std::uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch())
            .count());
    result.memory_bytes = memory.global_bytes_outstanding;
    result.memory_soft_limit = memory.global_budget.soft_limit;
    result.memory_hard_limit = memory.global_budget.hard_limit;
    result.memory_pressure_events = memory.pressure_event_count;
    result.jobs_submitted = jobs.submitted;
    result.jobs_completed = jobs.completed;
    result.jobs_stolen = jobs.stolen;
    result.jobs_cancelled = jobs.cancelled;
    result.jobs_failed = jobs.failed;
    result.jobs_queued = jobs.queued_general + jobs.queued_main + jobs.queued_render + jobs.queued_io;
    result.dropped_profile_events = jobs.dropped_profile_events;
    result.memory_domains.reserve(memory.domains.size());
    for (const auto& domain : memory.domains)
    {
        result.memory_domains.push_back({.domain = std::string(arc::memory::to_string(domain.domain)),
                                         .bytes_outstanding = domain.stats.bytes_outstanding,
                                         .peak_bytes = domain.stats.peak_bytes_outstanding,
                                         .soft_limit = domain.budget.soft_limit,
                                         .hard_limit = domain.budget.hard_limit,
                                         .pressure = domain.soft_limit_exceeded});
    }
    result.allocation_groups.reserve(memory.allocation_groups.size());
    for (const auto& group : memory.allocation_groups)
    {
        result.allocation_groups.push_back({.domain = std::string(arc::memory::to_string(group.domain)),
                                            .tag = std::string(group.tag.name),
                                            .world_id = group.world_id,
                                            .thread_id = group.thread_id,
                                            .stack_id = group.stack_id,
                                            .allocation_count = group.allocation_count,
                                            .bytes_outstanding = group.bytes_outstanding});
    }
    result.jobs.reserve(jobs.recent_events.size());
    for (const auto& job : jobs.recent_events)
    {
        result.jobs.push_back({.sequence = job.sequence,
                               .name = job.name,
                               .priority = job_priority_name(job.priority),
                               .affinity = job_affinity_name(job.affinity),
                               .status = job_status_name(job.status),
                               .thread_id = job.thread_id,
                               .queued_nanoseconds = job.queued_nanoseconds,
                               .started_nanoseconds = job.started_nanoseconds,
                               .completed_nanoseconds = job.completed_nanoseconds});
    }
    return result;
}

} // namespace

int main()
{
    auto& memory = arc::memory::default_memory_system();
    arc::jobs::job_system jobs({.worker_count = 0,
                                .run_inline = false,
                                .io_worker_count = 2,
                                .enable_render_thread = true,
                                .profile_event_capacity = 8192,
                                .memory = &memory});
    jobs.register_main_thread();
    auto host = std::make_shared<arc::editor::arc_host>(std::make_unique<arc::render::renderer>());
    std::mutex host_mutex;
    std::mutex output_mutex;
    auto native_viewport = arc::editor::make_native_viewport_controller(host, host_mutex, output_mutex, jobs);
    const auto write_response = [&](const arc::editor::host_response& response)
    {
        std::lock_guard output_lock(output_mutex);
        std::cout << arc::editor::to_json(response) << '\n';
        std::cout.flush();
    };
    std::jthread event_pump(
        [&](std::stop_token stop)
        {
            auto next_profiler_sample = std::chrono::steady_clock::now();
            std::uint64_t profiler_sequence = std::uint64_t{1} << 63u;
            while (!stop.stop_requested())
            {
                std::vector<arc::editor::host_event> events;
                {
                    std::lock_guard host_lock(host_mutex);
                    events = host->poll_events();
                }
                if (!events.empty())
                {
                    std::lock_guard output_lock(output_mutex);
                    for (const auto& event : events)
                        std::cout << arc::editor::to_json(event) << '\n';
                    std::cout.flush();
                }
                const auto now = std::chrono::steady_clock::now();
                if (now >= next_profiler_sample)
                {
                    const auto snapshot = make_profiler_snapshot(jobs.snapshot(true), memory.snapshot());
                    const arc::editor::host_event event{.sequence = profiler_sequence++,
                                                        .event_type = arc::editor::host_event_type::profiler_snapshot,
                                                        .message = "Profiler snapshot",
                                                        .payload_json = arc::editor::to_json(snapshot)};
                    std::lock_guard output_lock(output_mutex);
                    std::cout << arc::editor::to_json(event) << '\n';
                    std::cout.flush();
                    next_profiler_sample = now + std::chrono::milliseconds(100);
                }
                std::this_thread::sleep_for(std::chrono::milliseconds(16));
            }
        });

    std::string line;
    while (std::getline(std::cin, line))
    {
        jobs.pump_main_thread();
        if (line.empty()) continue;

        std::string error;
        if (line.find("\"kind\":\"query\"") != std::string::npos ||
            line.find("\"kind\": \"query\"") != std::string::npos)
        {
            arc::editor::host_query_envelope query;
            if (!arc::editor::from_json(line, query, error))
            {
                std::cerr << "arc_host_process query parse error: " << error << '\n';
                write_response(
                    arc::editor::host_response{.request_id = query.request_id, .succeeded = false, .error = error});
                continue;
            }
            arc::editor::host_response response;
            {
                std::lock_guard lock(host_mutex);
                response = host->query(query);
            }
            write_response(response);
        }
        else
        {
            arc::editor::host_command_envelope command;
            if (!arc::editor::from_json(line, command, error))
            {
                std::cerr << "arc_host_process command parse error: " << error << '\n';
                write_response(
                    arc::editor::host_response{.request_id = command.request_id, .succeeded = false, .error = error});
                continue;
            }
            arc::editor::host_response response;
            if (const auto* create = std::get_if<arc::editor::host_viewport_create_command>(&command.payload);
                create && create->output == arc::editor::host_viewport_output_type::shared_texture)
            {
                {
                    std::lock_guard lock(host_mutex);
                    response = host->execute(command);
                }
                if (response.succeeded)
                {
                    std::string setup_error;
                    if (!native_viewport->create_shared(create->viewport_id, create->consumer_process_id, create->width,
                                                        create->height, setup_error))
                    {
                        response.succeeded = false;
                        response.error = std::move(setup_error);
                    }
                }
            }
            else
            {
                std::lock_guard lock(host_mutex);
                response = host->execute(command);
            }
            write_response(response);

            if (response.succeeded)
            {
                if (const auto* attach = std::get_if<arc::editor::host_viewport_attach_command>(&command.payload))
                    native_viewport->attach(attach->viewport_id, attach->native_handle, attach->x, attach->y,
                                            attach->width, attach->height);
                else if (const auto* resize = std::get_if<arc::editor::host_viewport_resize_command>(&command.payload))
                    native_viewport->resize(resize->viewport_id, resize->x, resize->y, resize->width, resize->height);
                else if (const auto* detach = std::get_if<arc::editor::host_viewport_detach_command>(&command.payload))
                    native_viewport->detach(detach->viewport_id);
                else if (const auto* release =
                             std::get_if<arc::editor::host_viewport_frame_released_command>(&command.payload))
                    native_viewport->release_frame(release->viewport_id, release->generation, release->frame_id,
                                                   release->consumer_handle);
                else if (const auto* visibility =
                             std::get_if<arc::editor::host_viewport_set_visibility_command>(&command.payload))
                    native_viewport->set_visible(visibility->viewport_id, visibility->visible);
                else if (const auto* pointer =
                             std::get_if<arc::editor::host_viewport_pointer_command>(&command.payload))
                    native_viewport->pointer(*pointer);
                else if (const auto* key = std::get_if<arc::editor::host_viewport_key_command>(&command.payload))
                    native_viewport->key(*key);
            }
        }
    }

    event_pump.request_stop();
    native_viewport->stop();
    return 0;
}
