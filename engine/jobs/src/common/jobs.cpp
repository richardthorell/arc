#include <arc/jobs.h>

#include <algorithm>
#include <chrono>
#include <stdexcept>

namespace arc
{

job_handle::job_handle(std::shared_future<void> future)
    : future_(std::move(future))
{
}

void job_handle::wait() const
{
    if (future_.valid())
        future_.get();
}

bool job_handle::valid() const noexcept
{
    return future_.valid();
}

bool job_handle::ready() const
{
    if (!future_.valid())
        return false;

    return future_.wait_for(std::chrono::seconds(0)) == std::future_status::ready;
}

std::size_t job_system::default_worker_count() noexcept
{
    const auto hardware_count = std::thread::hardware_concurrency();
    if (hardware_count <= 1)
        return 1;
    return static_cast<std::size_t>(hardware_count - 1);
}

job_system_config job_system::single_threaded_config() noexcept
{
    return { .worker_count = 0, .run_inline = true };
}

job_system::job_system(job_system_config config)
    : worker_count_(config.worker_count == 0 && !config.run_inline ? default_worker_count() : config.worker_count)
    , run_inline_(config.run_inline)
{
    if (run_inline_)
        return;

    workers_.reserve(worker_count_);
    for (std::size_t index = 0; index < worker_count_; ++index)
        workers_.emplace_back([this]() { worker_loop(); });
}

job_system::~job_system()
{
    shutdown();
}

void job_system::wait_all(const std::vector<job_handle>& handles) const
{
    for (const auto& handle : handles)
        handle.wait();
}

void job_system::shutdown()
{
    {
        std::lock_guard lock(mutex_);
        if (stopping_)
            return;
        stopping_ = true;
    }

    wake_.notify_all();
    for (auto& worker : workers_)
    {
        if (worker.joinable())
            worker.join();
    }
    workers_.clear();
}

std::size_t job_system::worker_count() const noexcept
{
    return worker_count_;
}

bool job_system::run_inline() const noexcept
{
    return run_inline_;
}

void job_system::enqueue(std::function<void()> function)
{
    if (run_inline_)
    {
        function();
        return;
    }

    {
        std::lock_guard lock(mutex_);
        if (stopping_)
            throw std::runtime_error("cannot submit to stopped job_system");
        queue_.push(std::move(function));
    }
    wake_.notify_one();
}

void job_system::worker_loop()
{
    for (;;)
    {
        std::function<void()> function;
        {
            std::unique_lock lock(mutex_);
            wake_.wait(lock, [this]() { return stopping_ || !queue_.empty(); });
            if (queue_.empty())
            {
                if (stopping_)
                    return;
                continue;
            }

            function = std::move(queue_.front());
            queue_.pop();
        }

        function();
    }
}

void wait_all(const std::vector<job_handle>& handles)
{
    for (const auto& handle : handles)
        handle.wait();
}

} // namespace arc
