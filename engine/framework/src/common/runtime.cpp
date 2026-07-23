#include <arc/framework/runtime.h>

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
    , module_context_(jobs_, default_logger(), memory_, default_tracked_memory_resource())
{
    jobs_.register_main_thread();
}

application_config runtime::normalize_config(application_config config)
{
    if (config.title.empty())
        config.title = "ARC Application";
    if (config.initial_width == 0)
        config.initial_width = 1280;
    if (config.initial_height == 0)
        config.initial_height = 720;
    return config;
}

void runtime::start()
{
    if (started_)
        return;

    started_ = true;
    running_ = true;
    start_time_ = clock::now();
    last_frame_time_ = start_time_;
    current_time_ = {};
    if (!modules_registered_)
    {
        app_->register_modules(modules_.registry());
        modules_registered_ = true;
    }
    modules_.start(module_context_);
    app_->on_start();
}

frame_time runtime::tick()
{
    if (!started_)
        start();
    if (!running_)
        return current_time_;

    const auto now = clock::now();
    current_time_.delta_seconds = std::chrono::duration<double>(now - last_frame_time_).count();
    current_time_.total_seconds = std::chrono::duration<double>(now - start_time_).count();
    last_frame_time_ = now;

    modules_.update(module_context_, current_time_);
    app_->on_update(current_time_);
    jobs_.pump_main_thread();
    tick_arena_.reset();
    frame_arena_.reset();
    ++current_time_.frame_index;
    return current_time_;
}

void runtime::dispatch(const event& value)
{
    if (!started_)
        start();

    modules_.dispatch(module_context_, value);
    app_->on_event(value);
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

    app_->on_shutdown();
    modules_.shutdown(module_context_);
    started_ = false;
    running_ = false;
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

} // namespace arc
