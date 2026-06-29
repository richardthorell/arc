#pragma once

#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <future>
#include <mutex>
#include <queue>
#include <thread>
#include <type_traits>
#include <utility>
#include <vector>

namespace arc
{

/**
 * @brief Configuration used when creating a job system.
 */
struct job_system_config
{
    std::size_t worker_count{};
    bool run_inline{};
};

/**
 * @brief Handle returned by submitted asynchronous jobs.
 */
class job_handle
{
public:
    job_handle() = default;

    /**
     * @brief Wait until the job completes and rethrow any job exception.
     */
    void wait() const;

    /**
     * @brief Return whether this handle refers to a submitted job.
     */
    bool valid() const noexcept;

    /**
     * @brief Return whether the job has completed.
     */
    bool ready() const;

private:
    explicit job_handle(std::shared_future<void> future);

    std::shared_future<void> future_;

    friend class job_system;
};

/**
 * @brief Shared worker-pool service for engine systems and modules.
 */
class job_system
{
public:
    /**
     * @brief Return the default worker count for the current machine.
     */
    static std::size_t default_worker_count() noexcept;

    /**
     * @brief Return a deterministic inline configuration useful for tests.
     */
    static job_system_config single_threaded_config() noexcept;

    explicit job_system(job_system_config config = {});
    ~job_system();

    job_system(const job_system&) = delete;
    job_system& operator=(const job_system&) = delete;

    /**
     * @brief Submit one callable for asynchronous execution.
     */
    template <class Function>
    job_handle submit(Function&& function)
    {
        auto promise = std::make_shared<std::promise<void>>();
        auto future = promise->get_future().share();

        enqueue([task = std::forward<Function>(function), promise]() mutable {
            try
            {
                std::invoke(task);
                promise->set_value();
            }
            catch (...)
            {
                promise->set_exception(std::current_exception());
            }
        });

        return job_handle(std::move(future));
    }

    /**
     * @brief Wait for every handle in the provided list.
     */
    void wait_all(const std::vector<job_handle>& handles) const;

    /**
     * @brief Execute a range in chunks, waiting for all chunks before returning.
     */
    template <class Function>
    void parallel_for(std::size_t begin, std::size_t end, std::size_t grain_size, Function&& function)
    {
        if (begin >= end)
            return;

        if (grain_size == 0)
            grain_size = 1;

        std::vector<job_handle> handles;
        for (std::size_t chunk_begin = begin; chunk_begin < end; chunk_begin += grain_size)
        {
            const std::size_t chunk_end = (chunk_begin + grain_size < end) ? chunk_begin + grain_size : end;
            handles.push_back(submit([chunk_begin, chunk_end, &function]() {
                std::invoke(function, chunk_begin, chunk_end);
            }));
        }

        wait_all(handles);
    }

    /**
     * @brief Stop workers and drain already queued work.
     */
    void shutdown();

    /**
     * @brief Return the configured number of worker threads.
     */
    std::size_t worker_count() const noexcept;

    /**
     * @brief Return whether this job system executes work inline on submit.
     */
    bool run_inline() const noexcept;

private:
    void enqueue(std::function<void()> function);
    void worker_loop();

    std::size_t worker_count_{};
    bool run_inline_{};
    mutable std::mutex mutex_;
    std::condition_variable wake_;
    std::queue<std::function<void()>> queue_;
    std::vector<std::thread> workers_;
    bool stopping_{};
};

/**
 * @brief Submit one callable to a job system.
 */
template <class Function>
job_handle submit(job_system& jobs, Function&& function)
{
    return jobs.submit(std::forward<Function>(function));
}

/**
 * @brief Wait for every handle in the provided list.
 */
void wait_all(const std::vector<job_handle>& handles);

/**
 * @brief Execute a range in chunks on a job system.
 */
template <class Function>
void parallel_for(job_system& jobs, std::size_t begin, std::size_t end, std::size_t grain_size, Function&& function)
{
    jobs.parallel_for(begin, end, grain_size, std::forward<Function>(function));
}

} // namespace arc
