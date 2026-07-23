#pragma once

#include <arc/memory/memory.h>

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <coroutine>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

namespace arc
{

enum class job_priority : std::uint8_t
{
    critical,
    high,
    normal,
    low,
    background,
    count
};

enum class job_affinity : std::uint8_t
{
    any_worker,
    main_thread,
    render_thread,
    io_thread
};

enum class job_status : std::uint8_t
{
    invalid,
    waiting_dependencies,
    queued,
    running,
    waiting_children,
    succeeded,
    failed,
    cancelled
};

constexpr bool job_status_complete(job_status status) noexcept
{
    return status == job_status::succeeded || status == job_status::failed || status == job_status::cancelled;
}

enum class job_dependency_policy : std::uint8_t
{
    cancel_on_failure,
    run_always
};

enum class job_shutdown_mode : std::uint8_t
{
    drain,
    cancel_pending
};

class job_cancelled final : public std::runtime_error
{
public:
    job_cancelled();
};

namespace detail
{
struct cancellation_state;
struct job_state;

class task_callable
{
public:
    task_callable() = default;

    template <class Function>
    explicit task_callable(Function&& function)
    {
        using stored_type = std::decay_t<Function>;
        if constexpr (sizeof(stored_type) <= inline_capacity &&
            alignof(stored_type) <= alignof(std::max_align_t) &&
            std::is_nothrow_move_constructible_v<stored_type>)
        {
            new (storage_) stored_type(std::forward<Function>(function));
            object_ = storage_;
            move_ = [](void* destination, void* source) noexcept {
                new (destination) stored_type(std::move(*static_cast<stored_type*>(source)));
                static_cast<stored_type*>(source)->~stored_type();
            };
            destroy_ = [](void* object) noexcept { static_cast<stored_type*>(object)->~stored_type(); };
            heap_ = false;
        }
        else
        {
            object_ = new stored_type(std::forward<Function>(function));
            destroy_ = [](void* object) noexcept { delete static_cast<stored_type*>(object); };
            heap_ = true;
        }
        invoke_ = [](void* object) { std::invoke(*static_cast<stored_type*>(object)); };
    }

    ~task_callable()
    {
        reset();
    }

    task_callable(task_callable&& other) noexcept
    {
        move_from(std::move(other));
    }

    task_callable& operator=(task_callable&& other) noexcept
    {
        if (this != &other)
        {
            reset();
            move_from(std::move(other));
        }
        return *this;
    }

    task_callable(const task_callable&) = delete;
    task_callable& operator=(const task_callable&) = delete;

    void operator()()
    {
        invoke_(object_);
    }

    explicit operator bool() const noexcept
    {
        return invoke_ != nullptr;
    }

private:
    static constexpr std::size_t inline_capacity = 64;
    using invoke_fn = void (*)(void*);
    using destroy_fn = void (*)(void*) noexcept;
    using move_fn = void (*)(void*, void*) noexcept;

    void reset() noexcept
    {
        if (object_ && destroy_)
            destroy_(object_);
        object_ = nullptr;
        invoke_ = nullptr;
        destroy_ = nullptr;
        move_ = nullptr;
        heap_ = false;
    }

    void move_from(task_callable&& other) noexcept
    {
        invoke_ = other.invoke_;
        destroy_ = other.destroy_;
        move_ = other.move_;
        heap_ = other.heap_;
        if (!other.object_)
        {
            object_ = nullptr;
            return;
        }
        if (heap_)
        {
            object_ = other.object_;
        }
        else
        {
            move_(storage_, other.object_);
            object_ = storage_;
        }
        other.object_ = nullptr;
        other.invoke_ = nullptr;
        other.destroy_ = nullptr;
        other.move_ = nullptr;
        other.heap_ = false;
    }

    alignas(std::max_align_t) std::byte storage_[inline_capacity]{};
    void* object_{};
    invoke_fn invoke_{};
    destroy_fn destroy_{};
    move_fn move_{};
    bool heap_{};
};

}

class cancellation_token
{
public:
    cancellation_token() = default;

    bool valid() const noexcept;
    bool stop_requested() const noexcept;

private:
    explicit cancellation_token(std::shared_ptr<detail::cancellation_state> state);
    std::shared_ptr<detail::cancellation_state> state_;
    friend class cancellation_source;
    friend class job_handle;
};

class cancellation_source
{
public:
    cancellation_source();

    cancellation_token token() const noexcept;
    bool request_cancel() noexcept;
    bool stop_requested() const noexcept;

private:
    std::shared_ptr<detail::cancellation_state> state_;
};

struct job_wait_result
{
    job_status status{ job_status::invalid };
    std::exception_ptr exception;

    bool succeeded() const noexcept { return status == job_status::succeeded; }
};

class job_system;

class job_handle
{
public:
    job_handle() = default;

    void wait() const;
    job_wait_result wait_result() const noexcept;
    bool valid() const noexcept;
    bool ready() const;
    job_status status() const noexcept;
    cancellation_token cancellation() const noexcept;
    bool request_cancel() const noexcept;

#if defined(ARC_ENABLE_JOB_COROUTINES)
    struct awaiter
    {
        std::shared_ptr<detail::job_state> state;
        bool await_ready() const noexcept;
        void await_suspend(std::coroutine_handle<> continuation);
        void await_resume() const;
    };
    awaiter operator co_await() const noexcept;
#endif

private:
    explicit job_handle(std::shared_ptr<detail::job_state> state);
    std::shared_ptr<detail::job_state> state_;
    friend class job_system;
    friend struct job_descriptor;
};

struct job_descriptor
{
    std::string_view name{ "unnamed job" };
    job_priority priority{ job_priority::normal };
    job_affinity affinity{ job_affinity::any_worker };
    std::vector<job_handle> dependencies;
    std::span<const job_handle> dependency_view;
    job_handle parent;
    cancellation_token cancellation;
    job_dependency_policy dependency_policy{ job_dependency_policy::cancel_on_failure };
};

struct job_profile_event
{
    std::uint64_t sequence{};
    std::string name;
    job_priority priority{ job_priority::normal };
    job_affinity affinity{ job_affinity::any_worker };
    job_status status{ job_status::invalid };
    std::uint64_t thread_id{};
    std::uint64_t queued_nanoseconds{};
    std::uint64_t started_nanoseconds{};
    std::uint64_t completed_nanoseconds{};
};

struct job_system_snapshot
{
    std::uint64_t sequence{};
    std::size_t worker_count{};
    std::size_t io_worker_count{};
    std::size_t queued_general{};
    std::size_t queued_main{};
    std::size_t queued_render{};
    std::size_t queued_io{};
    std::uint64_t submitted{};
    std::uint64_t completed{};
    std::uint64_t stolen{};
    std::uint64_t cancelled{};
    std::uint64_t failed{};
    std::uint64_t dropped_profile_events{};
    std::vector<job_profile_event> recent_events;
};

struct job_system_config
{
    std::size_t worker_count{};
    bool run_inline{};
    std::size_t io_worker_count{ 2 };
    bool enable_render_thread{ true };
    std::size_t profile_event_capacity{ 8192 };
    memory_system* memory{};
};

template <class T>
class job_future
{
private:
    struct value_state
    {
        std::mutex mutex;
        std::optional<T> value;
    };

public:
    job_future() = default;

    bool valid() const noexcept { return handle_.valid(); }
    bool ready() const { return handle_.ready(); }
    job_status status() const noexcept { return handle_.status(); }
    const job_handle& handle() const noexcept { return handle_; }

    T get() const
    {
        handle_.wait();
        std::lock_guard lock(value_->mutex);
        return std::move(*value_->value);
    }

#if defined(ARC_ENABLE_JOB_COROUTINES)
    struct awaiter
    {
        job_handle handle;
        std::shared_ptr<value_state> value;
        bool await_ready() const { return handle.ready(); }
        void await_suspend(std::coroutine_handle<> continuation)
        {
            handle.operator co_await().await_suspend(continuation);
        }
        T await_resume()
        {
            handle.wait();
            std::lock_guard lock(value->mutex);
            return std::move(*value->value);
        }
    };
    awaiter operator co_await() const { return { handle_, value_ }; }
#endif

private:
    job_future(job_handle handle, std::shared_ptr<value_state> value)
        : handle_(std::move(handle))
        , value_(std::move(value))
    {
    }

    job_handle handle_;
    std::shared_ptr<value_state> value_;
    friend class job_system;
};

template <>
class job_future<void>
{
public:
    job_future() = default;
    explicit job_future(job_handle handle) : handle_(std::move(handle)) {}

    bool valid() const noexcept { return handle_.valid(); }
    bool ready() const { return handle_.ready(); }
    job_status status() const noexcept { return handle_.status(); }
    const job_handle& handle() const noexcept { return handle_; }
    void get() const { handle_.wait(); }

#if defined(ARC_ENABLE_JOB_COROUTINES)
    auto operator co_await() const noexcept { return handle_.operator co_await(); }
#endif

private:
    job_handle handle_;
};

#if defined(ARC_ENABLE_JOB_COROUTINES)
template <class T = void>
class job_task;

template <class T>
class job_task
{
public:
    struct promise_type
    {
        std::optional<T> value;
        std::exception_ptr exception;

        job_task get_return_object() noexcept
        {
            return job_task(std::coroutine_handle<promise_type>::from_promise(*this));
        }
        std::suspend_never initial_suspend() const noexcept { return {}; }
        std::suspend_always final_suspend() const noexcept { return {}; }
        void unhandled_exception() noexcept { exception = std::current_exception(); }
        template <class Value>
        void return_value(Value&& result) { value.emplace(std::forward<Value>(result)); }
    };

    job_task() = default;
    ~job_task() { if (coroutine_) coroutine_.destroy(); }
    job_task(job_task&& other) noexcept : coroutine_(std::exchange(other.coroutine_, {})) {}
    job_task& operator=(job_task&& other) noexcept
    {
        if (this != &other)
        {
            if (coroutine_) coroutine_.destroy();
            coroutine_ = std::exchange(other.coroutine_, {});
        }
        return *this;
    }

    T get()
    {
        while (coroutine_ && !coroutine_.done())
            std::this_thread::yield();
        if (coroutine_.promise().exception)
            std::rethrow_exception(coroutine_.promise().exception);
        return std::move(*coroutine_.promise().value);
    }

private:
    explicit job_task(std::coroutine_handle<promise_type> coroutine) : coroutine_(coroutine) {}
    std::coroutine_handle<promise_type> coroutine_{};
};

template <>
class job_task<void>
{
public:
    struct promise_type
    {
        std::exception_ptr exception;
        job_task get_return_object() noexcept
        {
            return job_task(std::coroutine_handle<promise_type>::from_promise(*this));
        }
        std::suspend_never initial_suspend() const noexcept { return {}; }
        std::suspend_always final_suspend() const noexcept { return {}; }
        void unhandled_exception() noexcept { exception = std::current_exception(); }
        void return_void() noexcept {}
    };

    job_task() = default;
    ~job_task() { if (coroutine_) coroutine_.destroy(); }
    job_task(job_task&& other) noexcept : coroutine_(std::exchange(other.coroutine_, {})) {}
    job_task& operator=(job_task&& other) noexcept
    {
        if (this != &other)
        {
            if (coroutine_) coroutine_.destroy();
            coroutine_ = std::exchange(other.coroutine_, {});
        }
        return *this;
    }
    void get()
    {
        while (coroutine_ && !coroutine_.done())
            std::this_thread::yield();
        if (coroutine_.promise().exception)
            std::rethrow_exception(coroutine_.promise().exception);
    }

private:
    explicit job_task(std::coroutine_handle<promise_type> coroutine) : coroutine_(coroutine) {}
    std::coroutine_handle<promise_type> coroutine_{};
};
#endif

class job_system
{
public:
    struct implementation;

    static std::size_t default_worker_count() noexcept;
    static job_system_config single_threaded_config() noexcept;

    explicit job_system(job_system_config config = {});
    ~job_system();

    job_system(const job_system&) = delete;
    job_system& operator=(const job_system&) = delete;

    template <class Function>
    job_handle submit(Function&& function)
    {
        return submit({}, std::forward<Function>(function));
    }

    template <class Function>
    job_handle submit(job_descriptor descriptor, Function&& function)
    {
        return submit_erased(std::move(descriptor), detail::task_callable(std::forward<Function>(function)), false);
    }

    template <class Function>
    job_handle submit_child(job_descriptor descriptor, Function&& function)
    {
        descriptor.parent = current_job();
        if (!descriptor.parent.valid())
            throw std::logic_error("submit_child must be called while executing a job");
        return submit(std::move(descriptor), std::forward<Function>(function));
    }

    template <class Function>
    job_handle submit_child(Function&& function)
    {
        return submit_child({}, std::forward<Function>(function));
    }

    template <class Function>
    void dispatch(Function&& function)
    {
        dispatch({}, std::forward<Function>(function));
    }

    template <class Function>
    void dispatch(job_descriptor descriptor, Function&& function)
    {
        (void)submit_erased(
            std::move(descriptor),
            detail::task_callable(std::forward<Function>(function)),
            true);
    }

    template <class Function>
    auto submit_future(job_descriptor descriptor, Function&& function)
    {
        using result_type = std::invoke_result_t<std::decay_t<Function>>;
        if constexpr (std::is_void_v<result_type>)
        {
            return job_future<void>(submit(std::move(descriptor), std::forward<Function>(function)));
        }
        else
        {
            using future_type = job_future<result_type>;
            auto value = std::make_shared<typename future_type::value_state>();
            auto handle = submit(std::move(descriptor), [
                function = std::forward<Function>(function),
                value
            ]() mutable {
                result_type result = std::invoke(function);
                std::lock_guard lock(value->mutex);
                value->value.emplace(std::move(result));
            });
            return future_type(std::move(handle), std::move(value));
        }
    }

    template <class Function>
    auto submit_future(Function&& function)
    {
        return submit_future({}, std::forward<Function>(function));
    }

    void wait_all(const std::vector<job_handle>& handles) const;

    template <class Function>
    void parallel_for(std::size_t begin, std::size_t end, std::size_t grain_size, Function&& function)
    {
        if (begin >= end)
            return;
        grain_size = std::max<std::size_t>(grain_size, 1);
        std::vector<job_handle> handles;
        handles.reserve((end - begin + grain_size - 1) / grain_size);
        for (std::size_t chunk_begin = begin; chunk_begin < end; chunk_begin += grain_size)
        {
            const std::size_t chunk_end = std::min(end, chunk_begin + grain_size);
            handles.push_back(submit({
                .name = "parallel_for",
                .priority = job_priority::normal
            }, [chunk_begin, chunk_end, &function]() {
                std::invoke(function, chunk_begin, chunk_end);
            }));
        }
        wait_all(handles);
    }

    std::size_t pump_main_thread(std::size_t maximum_jobs = static_cast<std::size_t>(-1));
    std::size_t pump_render_thread(std::size_t maximum_jobs = static_cast<std::size_t>(-1));
    void register_main_thread() noexcept;
    bool is_main_thread() const noexcept;
    job_handle current_job() const noexcept;

    void shutdown(job_shutdown_mode mode = job_shutdown_mode::drain);
    std::size_t worker_count() const noexcept;
    std::size_t io_worker_count() const noexcept;
    bool run_inline() const noexcept;
    job_system_snapshot snapshot(bool consume_events = false) const;

private:
    job_handle submit_erased(job_descriptor descriptor, detail::task_callable function, bool detached);
    job_wait_result wait_for(const std::shared_ptr<detail::job_state>& state) const noexcept;
    void add_coroutine_continuation(
        const std::shared_ptr<detail::job_state>& state,
        std::coroutine_handle<> continuation);

    std::unique_ptr<implementation> implementation_;
    friend class job_handle;
};

template <class Function>
job_handle submit(job_system& jobs, Function&& function)
{
    return jobs.submit(std::forward<Function>(function));
}

void wait_all(const std::vector<job_handle>& handles);

template <class Function>
void parallel_for(job_system& jobs, std::size_t begin, std::size_t end, std::size_t grain_size, Function&& function)
{
    jobs.parallel_for(begin, end, grain_size, std::forward<Function>(function));
}

} // namespace arc
