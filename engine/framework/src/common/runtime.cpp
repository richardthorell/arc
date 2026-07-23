#include <arc/framework/runtime.h>

#include <arc/diagnostics/log.h>

#include <algorithm>
#include <cmath>
#include <exception>
#include <limits>
#include <string>
#include <utility>

namespace arc
{

application::~application() = default;

application_config application::configure() const
{
    return {};
}

void application::on_start()
{
}

void application::register_modules(module_registry&)
{
}

void application::register_services(runtime_service_registry&)
{
}

void application::register_worlds(runtime_world_manager&)
{
}

void application::on_update(const frame_time&)
{
}

void application::on_event(const event&)
{
}

void application::on_shutdown()
{
}

runtime::runtime(application& app)
    : runtime(app, app.configure())
{
}

runtime::runtime(application& app, application_config config)
    : app_(&app)
    , config_(normalize_config(std::move(config)))
    , jobs_({ .memory = &memory_ })
    , module_context_(
        jobs_,
        default_logger(),
        memory_,
        default_tracked_memory_resource(),
        &services_,
        &worlds_)
{
    jobs_.register_main_thread();
    worlds_.set_snapshot_budget(config_.simulation.snapshot_budget_bytes);
}

runtime::~runtime()
{
    try
    {
        shutdown();
    }
    catch (...)
    {
        // Destructors must not leak application shutdown failures.
    }
}

application_config runtime::normalize_config(application_config config)
{
    if (config.title.empty())
        config.title = "ARC Application";
    if (config.initial_width == 0)
        config.initial_width = 1280;
    if (config.initial_height == 0)
        config.initial_height = 720;
    if (!valid_simulation_config(config.simulation))
        config.simulation = simulation_config{};
    if (config.simulation.headless)
    {
        config.visible = false;
        config.start_focused = false;
        if (!config.simulation.allow_headless_time_controls)
            config.simulation.time_scale = 1.0;
    }
    return config;
}

void runtime::start()
{
    if (started_)
        return;

    try
    {
        if (!services_registered_)
        {
            app_->register_services(services_);
            services_registered_ = true;
        }
        services_.start();

        if (!worlds_registered_)
        {
            app_->register_worlds(worlds_);
            if (worlds_.size() == 0)
            {
                runtime_world_descriptor descriptor{};
                descriptor.role = config_.simulation.default_world_role;
                descriptor.name = descriptor.role == runtime_world_role::server
                    ? "Server World"
                    : descriptor.role == runtime_world_role::editor_preview
                        ? "Editor Preview World"
                        : "Client World";
                descriptor.seed = config_.simulation.process_seed;
                descriptor.presentation_enabled =
                    config_.simulation.presentation_enabled && !config_.simulation.headless;
                worlds_.create(std::move(descriptor));
            }
            worlds_registered_ = true;
        }

        if (!modules_registered_)
        {
            app_->register_modules(modules_.registry());
            modules_registered_ = true;
        }
        modules_.start(module_context_);
        worlds_.start_all();
        app_->on_start();
    }
    catch (...)
    {
        worlds_.stop_all();
        modules_.shutdown(module_context_);
        services_.shutdown();
        started_ = false;
        running_ = false;
        throw;
    }

    started_ = true;
    running_ = true;
    paused_ = false;
    pending_steps_ = 0;
    accumulator_seconds_ = 0.0;
    explicit_total_seconds_ = 0.0;
    discarded_ticks_ = 0;
    current_tick_ = {};
    start_time_ = clock::now();
    last_frame_time_ = start_time_;
    current_time_ = {};
}

frame_time runtime::tick()
{
    if (!started_)
        start();
    if (!running_)
        return current_time_;

    const auto now = clock::now();
    const double delta_seconds = std::chrono::duration<double>(now - last_frame_time_).count();
    last_frame_time_ = now;
    return advance(delta_seconds);
}

frame_time runtime::advance(double wall_delta_seconds)
{
    if (!started_)
        start();
    if (!running_)
        return current_time_;

    if (!std::isfinite(wall_delta_seconds) || wall_delta_seconds < 0.0)
        wall_delta_seconds = 0.0;
    explicit_total_seconds_ += wall_delta_seconds;
    const double clamped_delta = config_.simulation.headless
        ? wall_delta_seconds
        : std::min(wall_delta_seconds, config_.simulation.maximum_frame_delta_seconds);
    current_time_.delta_seconds = clamped_delta;
    current_time_.total_seconds = explicit_total_seconds_;
    current_time_.completed_ticks = 0;

    if (!pending_input_.empty())
    {
        sampled_input_.insert(
            sampled_input_.end(),
            pending_input_.begin(),
            pending_input_.end());
        pending_input_.clear();
        ++input_revision_;
    }
    const simulation_input_snapshot input_snapshot{
        .revision = input_revision_,
        .commands = sampled_input_
    };

    if (!paused_)
        accumulator_seconds_ += clamped_delta * config_.simulation.time_scale;

    const double fixed_delta = config_.simulation.fixed_delta_seconds();
    const auto execute_tick = [&]() {
        current_tick_.id.value += 1;
        current_tick_.delta_seconds = fixed_delta;
        current_tick_.total_seconds =
            static_cast<double>(current_tick_.id.value) * fixed_delta;
        const runtime_world_run_result result = worlds_.run_fixed(
            jobs_,
            current_tick_,
            &services_,
            &input_snapshot,
            tick_arena_,
            frame_arena_,
            config_.simulation.process_seed);
        tick_arena_.reset();
        ++current_time_.completed_ticks;
        if (!result.succeeded() && config_.simulation.headless)
            request_stop();
    };

    if (paused_ && pending_steps_ > 0)
    {
        const std::uint32_t steps = pending_steps_;
        pending_steps_ = 0;
        worlds_.resume_all();
        for (std::uint32_t index = 0; index < steps && running_; ++index)
            execute_tick();
        worlds_.pause_all();
    }
    else if (!paused_)
    {
        std::uint32_t catch_up_ticks{};
        while (accumulator_seconds_ + 1e-12 >= fixed_delta &&
            catch_up_ticks < config_.simulation.maximum_catch_up_ticks &&
            running_)
        {
            accumulator_seconds_ -= fixed_delta;
            execute_tick();
            ++catch_up_ticks;
        }

        if (accumulator_seconds_ >= fixed_delta &&
            config_.simulation.overrun_policy == simulation_overrun_policy::discard_excess)
        {
            const auto discarded = static_cast<std::uint64_t>(accumulator_seconds_ / fixed_delta);
            discarded_ticks_ += discarded;
            accumulator_seconds_ = std::fmod(accumulator_seconds_, fixed_delta);
            warn("runtime", "Simulation catch-up limit reached; excess fixed ticks were discarded");
        }
    }

    current_time_.last_completed_tick = current_tick_.id;
    current_time_.discarded_ticks = discarded_ticks_;
    current_time_.interpolation_alpha = fixed_delta > 0.0
        ? std::clamp(accumulator_seconds_ / fixed_delta, 0.0, 1.0)
        : 0.0;
    if (config_.simulation.presentation_enabled)
    {
        worlds_.run_presentation(
            jobs_,
            current_tick_,
            static_cast<float>(clamped_delta),
            static_cast<float>(current_time_.interpolation_alpha),
            &services_,
            &input_snapshot,
            tick_arena_,
            frame_arena_,
            config_.simulation.process_seed);
    }
    modules_.update(module_context_, current_time_);
    app_->on_update(current_time_);
    jobs_.pump_main_thread();
    frame_arena_.reset();
    if (current_time_.completed_ticks != 0)
        sampled_input_.clear();
    ++current_time_.frame_index;
    return current_time_;
}

void runtime::dispatch(const event& value)
{
    if (!started_)
        start();

    modules_.dispatch(module_context_, value);
    app_->on_event(value);
    simulation_input_command input{};
    bool is_input{ true };
    input.modifiers = value.modifiers;
    input.x = value.x;
    input.y = value.y;
    input.repeat = value.repeat;
    switch (value.type)
    {
    case event_type::key_down:
        input.kind = simulation_input_kind::key;
        input.action = simulation_input_action::pressed;
        input.code = value.key_code;
        break;
    case event_type::key_up:
        input.kind = simulation_input_kind::key;
        input.action = simulation_input_action::released;
        input.code = value.key_code;
        break;
    case event_type::mouse_button_down:
        input.kind = simulation_input_kind::mouse_button;
        input.action = simulation_input_action::pressed;
        input.code = static_cast<std::int32_t>(value.button);
        break;
    case event_type::mouse_button_up:
        input.kind = simulation_input_kind::mouse_button;
        input.action = simulation_input_action::released;
        input.code = static_cast<std::int32_t>(value.button);
        break;
    case event_type::mouse_moved:
        input.kind = simulation_input_kind::mouse_position;
        input.action = simulation_input_action::changed;
        break;
    case event_type::mouse_wheel:
        input.kind = simulation_input_kind::mouse_wheel;
        input.action = simulation_input_action::changed;
        input.value = value.wheel_delta;
        break;
    case event_type::focus_gained:
    case event_type::focus_lost:
        input.kind = simulation_input_kind::focus;
        input.action = simulation_input_action::changed;
        input.value = value.type == event_type::focus_gained ? 1.0f : 0.0f;
        break;
    default:
        is_input = false;
        break;
    }
    if (is_input)
        pending_input_.push_back(input);
    if (value.type == event_type::close_requested)
        request_stop();
}

void runtime::request_stop() noexcept
{
    running_ = false;
}

void runtime::shutdown()
{
    if (!started_)
        return;

    std::exception_ptr failure;
    const auto capture_failure = [&failure](auto&& operation) {
        try
        {
            operation();
        }
        catch (...)
        {
            if (!failure)
                failure = std::current_exception();
        }
    };

    running_ = false;
    capture_failure([this] { app_->on_shutdown(); });
    capture_failure([this] { worlds_.stop_all(); });
    capture_failure([this] { modules_.shutdown(module_context_); });
    capture_failure([this] { services_.shutdown(); });
    started_ = false;
    paused_ = false;
    pending_steps_ = 0;
    if (failure)
        std::rethrow_exception(failure);
}

bool runtime::running() const noexcept
{
    return running_;
}

bool runtime::started() const noexcept
{
    return started_;
}

const application_config& runtime::config() const noexcept
{
    return config_;
}

job_system& runtime::jobs() noexcept
{
    return jobs_;
}

memory_system& runtime::memory() noexcept
{
    return memory_;
}

frame_arena& runtime::frame_memory() noexcept
{
    return frame_arena_;
}

tick_arena& runtime::tick_memory() noexcept
{
    return tick_arena_;
}

module_manager& runtime::modules() noexcept
{
    return modules_;
}

runtime_service_registry& runtime::services() noexcept
{
    return services_;
}

runtime_world_manager& runtime::worlds() noexcept
{
    return worlds_;
}

const runtime_world_manager& runtime::worlds() const noexcept
{
    return worlds_;
}

void runtime::pause() noexcept
{
    if (!started_ || (config_.simulation.headless && !config_.simulation.allow_headless_time_controls))
        return;
    paused_ = true;
    worlds_.pause_all();
}

void runtime::resume() noexcept
{
    if (!started_ || (config_.simulation.headless && !config_.simulation.allow_headless_time_controls))
        return;
    paused_ = false;
    pending_steps_ = 0;
    worlds_.resume_all();
}

bool runtime::paused() const noexcept
{
    return paused_;
}

bool runtime::step(std::uint32_t ticks) noexcept
{
    if (!started_ || !paused_ || ticks == 0 ||
        (config_.simulation.headless && !config_.simulation.allow_headless_time_controls))
        return false;
    pending_steps_ = std::min<std::uint32_t>(
        std::numeric_limits<std::uint32_t>::max() - pending_steps_,
        ticks) + pending_steps_;
    return true;
}

bool runtime::set_time_scale(double value) noexcept
{
    if (!std::isfinite(value) || value < 0.0 || value > 16.0 ||
        (config_.simulation.headless && !config_.simulation.allow_headless_time_controls))
        return false;
    config_.simulation.time_scale = value;
    return true;
}

double runtime::time_scale() const noexcept
{
    return config_.simulation.time_scale;
}

simulation_tick runtime::current_tick() const noexcept
{
    return current_tick_;
}

std::uint64_t runtime::discarded_ticks() const noexcept
{
    return discarded_ticks_;
}

world_snapshot_result runtime::capture_snapshot(runtime_world_id world, std::string label)
{
    return worlds_.capture_snapshot(
        world,
        current_tick_.id,
        current_tick_.total_seconds,
        std::move(label),
        &services_);
}

world_snapshot_result runtime::restore_snapshot(world_snapshot_id snapshot)
{
    return worlds_.restore_snapshot(snapshot, &services_);
}

} // namespace arc
