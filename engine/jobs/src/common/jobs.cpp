#include <arc/jobs/jobs.h>
#include <arc/diagnostics/log.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <deque>
#include <limits>
#include <random>
#include <thread>
#include <unordered_set>

namespace arc
{
namespace
{

constexpr std::size_t priority_count = static_cast<std::size_t>(job_priority::count);
constexpr std::size_t fairness_quota = 8;
using clock_type = std::chrono::steady_clock;

std::uint64_t now_nanoseconds() noexcept
{
    return static_cast<std::uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(clock_type::now().time_since_epoch()).count());
}

std::uint64_t thread_id_value() noexcept
{
    return static_cast<std::uint64_t>(std::hash<std::thread::id>{}(std::this_thread::get_id()));
}

struct thread_worker_context
{
    job_system* scheduler{};
    std::size_t worker_index{ static_cast<std::size_t>(-1) };
    job_affinity affinity{ job_affinity::any_worker };
};

thread_local thread_worker_context worker_context{};
thread_local std::weak_ptr<detail::job_state> current_job_state;

}

namespace detail
{

struct cancellation_state
{
    std::atomic_bool cancelled{};
};

struct job_state
{
    job_system* scheduler{};
    std::string name;
    job_priority priority{ job_priority::normal };
    job_affinity affinity{ job_affinity::any_worker };
    job_dependency_policy dependency_policy{ job_dependency_policy::cancel_on_failure };
    cancellation_token cancellation;
    task_callable function;
    std::weak_ptr<job_state> parent;
    std::atomic_size_t unfinished{ 1 };
    std::atomic_size_t pending_dependencies{};
    std::atomic_bool dependency_failed{};
    std::atomic<job_status> status{ job_status::waiting_dependencies };
    std::exception_ptr exception;
    mutable std::mutex mutex;
    std::condition_variable completion;
    std::vector<std::shared_ptr<job_state>> dependents;
    std::vector<std::coroutine_handle<>> continuations;
    bool detached{};
    std::uint64_t sequence{};
    std::uint64_t queued_time{};
    std::uint64_t started_time{};
    std::uint64_t completed_time{};
    std::uint64_t execution_thread{};
};

}

struct job_system::implementation
{
    struct work_queue
    {
        mutable std::mutex mutex;
        std::array<std::deque<std::shared_ptr<detail::job_state>>, priority_count> priorities;

        void push(std::shared_ptr<detail::job_state> state, bool local)
        {
            std::lock_guard lock(mutex);
            auto& queue = priorities[static_cast<std::size_t>(state->priority)];
            if (local)
                queue.push_back(std::move(state));
            else
                queue.push_front(std::move(state));
        }

        std::shared_ptr<detail::job_state> pop_local(std::size_t& high_priority_streak)
        {
            std::lock_guard lock(mutex);
            std::size_t first = 0;
            if (high_priority_streak >= fairness_quota)
            {
                for (std::size_t priority = 1; priority < priority_count; ++priority)
                {
                    if (!priorities[priority].empty())
                    {
                        high_priority_streak = 0;
                        auto value = std::move(priorities[priority].back());
                        priorities[priority].pop_back();
                        return value;
                    }
                }
            }
            for (std::size_t priority = first; priority < priority_count; ++priority)
            {
                if (priorities[priority].empty())
                    continue;
                high_priority_streak = priority <= static_cast<std::size_t>(job_priority::high)
                    ? high_priority_streak + 1
                    : 0;
                auto value = std::move(priorities[priority].back());
                priorities[priority].pop_back();
                return value;
            }
            return {};
        }

        std::shared_ptr<detail::job_state> steal()
        {
            std::lock_guard lock(mutex);
            for (auto& queue : priorities)
            {
                if (queue.empty())
                    continue;
                auto value = std::move(queue.front());
                queue.pop_front();
                return value;
            }
            return {};
        }

        std::size_t size() const
        {
            std::lock_guard lock(mutex);
            std::size_t result{};
            for (const auto& queue : priorities)
                result += queue.size();
            return result;
        }

        std::vector<std::shared_ptr<detail::job_state>> take_all()
        {
            std::lock_guard lock(mutex);
            std::vector<std::shared_ptr<detail::job_state>> result;
            for (auto& queue : priorities)
            {
                while (!queue.empty())
                {
                    result.push_back(std::move(queue.front()));
                    queue.pop_front();
                }
            }
            return result;
        }
    };

    explicit implementation(job_system& owner, job_system_config value)
        : scheduler(&owner)
        , config(value)
        , memory(value.memory ? value.memory : &default_memory_system())
    {
    }

    job_system* scheduler{};
    job_system_config config;
    memory_system* memory{};
    std::vector<std::unique_ptr<work_queue>> workers;
    work_queue injection;
    work_queue main;
    work_queue render;
    work_queue io;
    std::vector<std::thread> worker_threads;
    std::vector<std::thread> io_threads;
    std::thread render_thread;
    std::thread::id main_thread;
    mutable std::mutex wake_mutex;
    std::condition_variable wake;
    std::atomic_bool stopping{};
    std::atomic<job_shutdown_mode> shutdown_mode{ job_shutdown_mode::drain };
    std::atomic_uint64_t next_sequence{};
    std::atomic_uint64_t submitted{};
    std::atomic_uint64_t completed{};
    std::atomic_uint64_t stolen{};
    std::atomic_uint64_t cancelled{};
    std::atomic_uint64_t failed{};
    std::atomic_size_t active{};
    std::atomic_uint64_t snapshot_sequence{};
    mutable std::mutex profile_mutex;
    mutable std::deque<job_profile_event> profile_events;
    std::atomic_uint64_t dropped_profile_events{};

    bool queues_empty() const
    {
        if (injection.size() || main.size() || render.size() || io.size())
            return false;
        for (const auto& worker : workers)
            if (worker->size())
                return false;
        return true;
    }
};

namespace
{

class job_pool_resource final : public std::pmr::memory_resource
{
public:
    explicit job_pool_resource(memory_system& memory)
        : backing_(memory, memory_domain::jobs, make_memory_tag("jobs.state"))
        , pool_(std::array<std::size_t, 3>{ 256, 512, 1024 }, 64, &backing_)
    {
    }

private:
    void* do_allocate(std::size_t bytes, std::size_t alignment) override
    {
        if (bytes <= 1024 && alignment <= alignof(std::max_align_t))
        {
            if (void* pointer = pool_.try_allocate(bytes, alignment))
                return pointer;
            throw std::bad_alloc();
        }
        return backing_.allocate(bytes, alignment);
    }

    void do_deallocate(void* pointer, std::size_t bytes, std::size_t alignment) override
    {
        if (bytes <= 1024 && alignment <= alignof(std::max_align_t))
            pool_.deallocate(pointer, bytes);
        else
            backing_.deallocate(pointer, bytes, alignment);
    }

    bool do_is_equal(const std::pmr::memory_resource& other) const noexcept override
    {
        return this == &other;
    }

    system_memory_resource backing_;
    fixed_block_pool pool_;
};

std::mutex pool_resources_mutex;
std::unordered_map<job_system::implementation*, std::shared_ptr<job_pool_resource>> pool_resources;

std::shared_ptr<job_pool_resource> pool_resource_for(job_system::implementation& implementation)
{
    std::lock_guard lock(pool_resources_mutex);
    auto& resource = pool_resources[&implementation];
    if (!resource)
        resource = std::make_shared<job_pool_resource>(*implementation.memory);
    return resource;
}

void release_pool_resource(job_system::implementation& implementation)
{
    std::lock_guard lock(pool_resources_mutex);
    pool_resources.erase(&implementation);
}

void record_profile(job_system::implementation& implementation, const std::shared_ptr<detail::job_state>& state)
{
    job_profile_event event{
        .sequence = state->sequence,
        .name = state->name,
        .priority = state->priority,
        .affinity = state->affinity,
        .status = state->status.load(std::memory_order_acquire),
        .thread_id = state->execution_thread,
        .queued_nanoseconds = state->queued_time,
        .started_nanoseconds = state->started_time,
        .completed_nanoseconds = state->completed_time
    };
    std::lock_guard lock(implementation.profile_mutex);
    if (implementation.profile_events.size() >= implementation.config.profile_event_capacity)
    {
        implementation.profile_events.pop_front();
        implementation.dropped_profile_events.fetch_add(1, std::memory_order_relaxed);
    }
    implementation.profile_events.push_back(std::move(event));
}

void finish_part(
    job_system::implementation& implementation,
    const std::shared_ptr<detail::job_state>& state,
    job_status requested_status);

void dependency_finished(
    job_system::implementation& implementation,
    const std::shared_ptr<detail::job_state>& dependent,
    job_status prerequisite_status);

void enqueue_ready(job_system::implementation& implementation, std::shared_ptr<detail::job_state> state)
{
    if (state->cancellation.stop_requested() ||
        (state->dependency_failed.load(std::memory_order_acquire) &&
            state->dependency_policy == job_dependency_policy::cancel_on_failure))
    {
        finish_part(implementation, state, job_status::cancelled);
        return;
    }

    state->status.store(job_status::queued, std::memory_order_release);
    state->queued_time = now_nanoseconds();
    if (implementation.config.run_inline)
    {
        state->status.store(job_status::running, std::memory_order_release);
        state->started_time = now_nanoseconds();
        state->execution_thread = thread_id_value();
        job_status completion = job_status::succeeded;
        const auto previous_job = current_job_state;
        current_job_state = state;
        try
        {
            state->function();
        }
        catch (...)
        {
            state->exception = std::current_exception();
            completion = job_status::failed;
        }
        current_job_state = previous_job;
        finish_part(implementation, state, completion);
        return;
    }

    switch (state->affinity)
    {
    case job_affinity::main_thread:
        implementation.main.push(std::move(state), false);
        break;
    case job_affinity::render_thread:
        implementation.render.push(std::move(state), false);
        break;
    case job_affinity::io_thread:
        implementation.io.push(std::move(state), false);
        break;
    case job_affinity::any_worker:
        if (worker_context.scheduler == implementation.scheduler &&
            worker_context.affinity == job_affinity::any_worker &&
            worker_context.worker_index < implementation.workers.size())
        {
            implementation.workers[worker_context.worker_index]->push(std::move(state), true);
        }
        else
        {
            implementation.injection.push(std::move(state), false);
        }
        break;
    }
    implementation.wake.notify_all();
}

void finish_part(
    job_system::implementation& implementation,
    const std::shared_ptr<detail::job_state>& state,
    job_status requested_status)
{
    if (requested_status == job_status::failed)
        state->status.store(job_status::failed, std::memory_order_release);
    else if (requested_status == job_status::cancelled &&
        state->status.load(std::memory_order_acquire) != job_status::failed)
        state->status.store(job_status::cancelled, std::memory_order_release);

    if (state->unfinished.fetch_sub(1, std::memory_order_acq_rel) != 1)
    {
        if (!job_status_complete(state->status.load(std::memory_order_acquire)))
            state->status.store(job_status::waiting_children, std::memory_order_release);
        return;
    }

    auto final_status = state->status.load(std::memory_order_acquire);
    if (final_status != job_status::failed && final_status != job_status::cancelled)
        final_status = requested_status == job_status::cancelled ? job_status::cancelled : job_status::succeeded;
    state->status.store(final_status, std::memory_order_release);
    state->completed_time = now_nanoseconds();
    if (final_status == job_status::failed)
    {
        implementation.failed.fetch_add(1, std::memory_order_relaxed);
        if (state->detached)
            error("jobs", std::string("Fire-and-forget job failed: ") + state->name);
    }
    if (final_status == job_status::cancelled)
        implementation.cancelled.fetch_add(1, std::memory_order_relaxed);
    implementation.completed.fetch_add(1, std::memory_order_relaxed);

    std::vector<std::shared_ptr<detail::job_state>> dependents;
    std::vector<std::coroutine_handle<>> continuations;
    {
        std::lock_guard lock(state->mutex);
        dependents.swap(state->dependents);
        continuations.swap(state->continuations);
    }
    state->completion.notify_all();
    record_profile(implementation, state);

    for (const auto& dependent : dependents)
        dependency_finished(implementation, dependent, final_status);
    for (const auto continuation : continuations)
        if (continuation)
            continuation.resume();
    if (auto parent = state->parent.lock())
        finish_part(implementation, parent, final_status == job_status::failed ? job_status::failed : job_status::succeeded);
}

void dependency_finished(
    job_system::implementation& implementation,
    const std::shared_ptr<detail::job_state>& dependent,
    job_status prerequisite_status)
{
    if (prerequisite_status == job_status::failed || prerequisite_status == job_status::cancelled)
        dependent->dependency_failed.store(true, std::memory_order_release);
    if (dependent->pending_dependencies.fetch_sub(1, std::memory_order_acq_rel) == 1)
        enqueue_ready(implementation, dependent);
}

void execute_state(job_system::implementation& implementation, const std::shared_ptr<detail::job_state>& state)
{
    if (!state)
        return;
    if (implementation.shutdown_mode.load(std::memory_order_acquire) == job_shutdown_mode::cancel_pending ||
        state->cancellation.stop_requested())
    {
        finish_part(implementation, state, job_status::cancelled);
        return;
    }

    job_status expected = job_status::queued;
    if (!state->status.compare_exchange_strong(expected, job_status::running, std::memory_order_acq_rel))
        return;
    state->started_time = now_nanoseconds();
    state->execution_thread = thread_id_value();
    implementation.active.fetch_add(1, std::memory_order_acq_rel);
    const auto previous_job = current_job_state;
    current_job_state = state;
    job_status completion = job_status::succeeded;
    try
    {
        allocation_tag_scope tag("jobs.execute");
        state->function();
    }
    catch (...)
    {
        state->exception = std::current_exception();
        completion = job_status::failed;
    }
    current_job_state = previous_job;
    finish_part(implementation, state, completion);
    implementation.active.fetch_sub(1, std::memory_order_acq_rel);
    implementation.wake.notify_all();
}

std::shared_ptr<detail::job_state> take_general(job_system::implementation& implementation, std::size_t worker_index)
{
    static thread_local std::size_t streak{};
    if (auto value = implementation.workers[worker_index]->pop_local(streak))
        return value;
    if (auto value = implementation.injection.pop_local(streak))
        return value;

    const auto count = implementation.workers.size();
    for (std::size_t offset = 1; offset < count; ++offset)
    {
        const auto victim = (worker_index + offset) % count;
        if (auto value = implementation.workers[victim]->steal())
        {
            implementation.stolen.fetch_add(1, std::memory_order_relaxed);
            return value;
        }
    }
    return {};
}

void general_worker_loop(job_system::implementation& implementation, std::size_t worker_index)
{
    worker_context = { .scheduler = implementation.scheduler, .worker_index = worker_index, .affinity = job_affinity::any_worker };
    for (;;)
    {
        if (auto state = take_general(implementation, worker_index))
        {
            execute_state(implementation, state);
            continue;
        }
        if (implementation.stopping.load(std::memory_order_acquire) &&
            implementation.active.load(std::memory_order_acquire) == 0 &&
            implementation.queues_empty())
            break;
        std::unique_lock lock(implementation.wake_mutex);
        implementation.wake.wait_for(lock, std::chrono::milliseconds(2));
    }
    worker_context = {};
}

void affinity_worker_loop(job_system::implementation& implementation, job_affinity affinity, std::size_t index)
{
    worker_context = { .scheduler = implementation.scheduler, .worker_index = index, .affinity = affinity };
    auto* queue = affinity == job_affinity::render_thread ? &implementation.render : &implementation.io;
    std::size_t streak{};
    for (;;)
    {
        if (auto state = queue->pop_local(streak))
        {
            execute_state(implementation, state);
            continue;
        }
        if (implementation.stopping.load(std::memory_order_acquire) &&
            implementation.active.load(std::memory_order_acquire) == 0 &&
            implementation.queues_empty())
            break;
        std::unique_lock lock(implementation.wake_mutex);
        implementation.wake.wait_for(lock, std::chrono::milliseconds(2));
    }
    worker_context = {};
}

}

job_cancelled::job_cancelled()
    : std::runtime_error("job was cancelled")
{
}

cancellation_token::cancellation_token(std::shared_ptr<detail::cancellation_state> state)
    : state_(std::move(state))
{
}

bool cancellation_token::valid() const noexcept
{
    return static_cast<bool>(state_);
}

bool cancellation_token::stop_requested() const noexcept
{
    return state_ && state_->cancelled.load(std::memory_order_acquire);
}

cancellation_source::cancellation_source()
    : state_(std::make_shared<detail::cancellation_state>())
{
}

cancellation_token cancellation_source::token() const noexcept
{
    return cancellation_token(state_);
}

bool cancellation_source::request_cancel() noexcept
{
    return !state_->cancelled.exchange(true, std::memory_order_acq_rel);
}

bool cancellation_source::stop_requested() const noexcept
{
    return state_->cancelled.load(std::memory_order_acquire);
}

job_handle::job_handle(std::shared_ptr<detail::job_state> state)
    : state_(std::move(state))
{
}

void job_handle::wait() const
{
    const auto result = wait_result();
    if (result.status == job_status::cancelled)
        throw job_cancelled();
    if (result.exception)
        std::rethrow_exception(result.exception);
}

job_wait_result job_handle::wait_result() const noexcept
{
    if (!state_)
        return {};
    return state_->scheduler->wait_for(state_);
}

bool job_handle::valid() const noexcept
{
    return static_cast<bool>(state_);
}

bool job_handle::ready() const
{
    return job_status_complete(status());
}

job_status job_handle::status() const noexcept
{
    return state_ ? state_->status.load(std::memory_order_acquire) : job_status::invalid;
}

cancellation_token job_handle::cancellation() const noexcept
{
    return state_ ? state_->cancellation : cancellation_token{};
}

bool job_handle::request_cancel() const noexcept
{
    if (!state_ || !state_->cancellation.state_)
        return false;
    return !state_->cancellation.state_->cancelled.exchange(true, std::memory_order_acq_rel);
}

#if defined(ARC_ENABLE_JOB_COROUTINES)
bool job_handle::awaiter::await_ready() const noexcept
{
    return !state || job_status_complete(state->status.load(std::memory_order_acquire));
}

void job_handle::awaiter::await_suspend(std::coroutine_handle<> continuation)
{
    state->scheduler->add_coroutine_continuation(state, continuation);
}

void job_handle::awaiter::await_resume() const
{
    job_handle(state).wait();
}

job_handle::awaiter job_handle::operator co_await() const noexcept
{
    return { state_ };
}
#endif

std::size_t job_system::default_worker_count() noexcept
{
    const auto hardware_count = std::thread::hardware_concurrency();
    if (hardware_count <= 4)
        return 1;
    return static_cast<std::size_t>(hardware_count - 4);
}

job_system_config job_system::single_threaded_config() noexcept
{
    return {
        .worker_count = 0,
        .run_inline = true,
        .io_worker_count = 0,
        .enable_render_thread = false
    };
}

job_system::job_system(job_system_config config)
    : implementation_(std::make_unique<implementation>(*this, config))
{
    auto& value = *implementation_;
    value.main_thread = std::this_thread::get_id();
    if (config.run_inline)
        return;

    const auto worker_count = config.worker_count == 0 ? default_worker_count() : config.worker_count;
    value.workers.reserve(worker_count);
    for (std::size_t index = 0; index < worker_count; ++index)
        value.workers.push_back(std::make_unique<implementation::work_queue>());
    for (std::size_t index = 0; index < worker_count; ++index)
        value.worker_threads.emplace_back([&value, index] { general_worker_loop(value, index); });
    for (std::size_t index = 0; index < config.io_worker_count; ++index)
        value.io_threads.emplace_back([&value, index] { affinity_worker_loop(value, job_affinity::io_thread, index); });
    if (config.enable_render_thread)
        value.render_thread = std::thread([&value] { affinity_worker_loop(value, job_affinity::render_thread, 0); });
}

job_system::~job_system()
{
    shutdown();
    release_pool_resource(*implementation_);
}

job_handle job_system::submit_erased(job_descriptor descriptor, detail::task_callable function, bool detached)
{
    if (!implementation_->config.run_inline &&
        descriptor.affinity == job_affinity::render_thread &&
        !implementation_->config.enable_render_thread)
        throw std::invalid_argument("render-affinity job submitted without a render executor");
    if (!implementation_->config.run_inline &&
        descriptor.affinity == job_affinity::io_thread &&
        implementation_->config.io_worker_count == 0)
        throw std::invalid_argument("IO-affinity job submitted without an IO executor");

    auto pool = pool_resource_for(*implementation_);
    void* state_memory = pool->allocate(sizeof(detail::job_state), alignof(detail::job_state));
    detail::job_state* raw_state{};
    try
    {
        raw_state = new (state_memory) detail::job_state();
    }
    catch (...)
    {
        pool->deallocate(state_memory, sizeof(detail::job_state), alignof(detail::job_state));
        throw;
    }
    auto state = std::shared_ptr<detail::job_state>(raw_state, [pool = std::move(pool)](detail::job_state* value) {
        value->~job_state();
        pool->deallocate(value, sizeof(detail::job_state), alignof(detail::job_state));
    });
    state->scheduler = this;
    state->name = descriptor.name.empty() ? "unnamed job" : std::string(descriptor.name);
    state->priority = descriptor.priority;
    state->affinity = descriptor.affinity;
    state->dependency_policy = descriptor.dependency_policy;
    if (descriptor.cancellation.valid())
    {
        state->cancellation = descriptor.cancellation;
    }
    else
    {
        cancellation_source source;
        state->cancellation = source.token();
    }
    state->function = std::move(function);
    state->detached = detached;
    state->sequence = implementation_->next_sequence.fetch_add(1, std::memory_order_relaxed) + 1;
    implementation_->submitted.fetch_add(1, std::memory_order_relaxed);

    if (descriptor.parent.valid())
    {
        for (const auto& dependency : descriptor.dependencies)
        {
            if (!dependency.valid())
                continue;
            for (auto ancestor = descriptor.parent.state_; ancestor; ancestor = ancestor->parent.lock())
            {
                if (ancestor.get() == dependency.state_.get())
                    throw std::invalid_argument("a child job cannot depend on an ancestor that waits for it");
            }
        }
    }

    if (descriptor.parent.valid())
    {
        auto parent = descriptor.parent.state_;
        auto unfinished = parent->unfinished.load(std::memory_order_acquire);
        while (unfinished != 0 &&
            !parent->unfinished.compare_exchange_weak(unfinished, unfinished + 1, std::memory_order_acq_rel))
        {
        }
        if (unfinished == 0)
            throw std::invalid_argument("cannot attach a child to a completed job");
        state->parent = parent;
    }

    std::unordered_set<detail::job_state*> unique_dependencies;
    for (const auto& dependency : descriptor.dependencies)
    {
        if (!dependency.valid() || dependency.state_.get() == state.get())
            continue;
        if (!unique_dependencies.insert(dependency.state_.get()).second)
            continue;

        bool completed{};
        job_status dependency_status{};
        {
            std::lock_guard lock(dependency.state_->mutex);
            dependency_status = dependency.state_->status.load(std::memory_order_acquire);
            completed = job_status_complete(dependency_status);
            if (!completed)
            {
                state->pending_dependencies.fetch_add(1, std::memory_order_relaxed);
                dependency.state_->dependents.push_back(state);
            }
        }
        if (completed && (dependency_status == job_status::failed || dependency_status == job_status::cancelled))
            state->dependency_failed.store(true, std::memory_order_release);
    }

    if (state->pending_dependencies.load(std::memory_order_acquire) == 0)
        enqueue_ready(*implementation_, state);
    return job_handle(std::move(state));
}

void job_system::wait_all(const std::vector<job_handle>& handles) const
{
    for (const auto& handle : handles)
        handle.wait();
}

job_wait_result job_system::wait_for(const std::shared_ptr<detail::job_state>& state) const noexcept
{
    if (!state)
        return {};

    while (!job_status_complete(state->status.load(std::memory_order_acquire)))
    {
        bool helped{};
        if (!implementation_->config.run_inline &&
            worker_context.scheduler == this &&
            worker_context.affinity == job_affinity::any_worker &&
            worker_context.worker_index < implementation_->workers.size())
        {
            if (auto work = take_general(*implementation_, worker_context.worker_index))
            {
                execute_state(*implementation_, work);
                helped = true;
            }
        }
        else if (is_main_thread())
        {
            helped = const_cast<job_system*>(this)->pump_main_thread(1) != 0;
            if (!helped && !implementation_->workers.empty())
            {
                if (auto work = implementation_->injection.steal())
                {
                    execute_state(*implementation_, work);
                    helped = true;
                }
            }
        }
        if (!helped)
        {
            std::unique_lock lock(state->mutex);
            state->completion.wait_for(lock, std::chrono::milliseconds(1), [&] {
                return job_status_complete(state->status.load(std::memory_order_acquire));
            });
        }
    }
    return {
        .status = state->status.load(std::memory_order_acquire),
        .exception = state->exception
    };
}

std::size_t job_system::pump_main_thread(std::size_t maximum_jobs)
{
    if (!is_main_thread() && !implementation_->config.run_inline)
        return 0;
    std::size_t executed{};
    std::size_t streak{};
    while (executed < maximum_jobs)
    {
        auto state = implementation_->main.pop_local(streak);
        if (!state)
            break;
        execute_state(*implementation_, state);
        ++executed;
    }
    return executed;
}

void job_system::register_main_thread() noexcept
{
    implementation_->main_thread = std::this_thread::get_id();
}

bool job_system::is_main_thread() const noexcept
{
    return implementation_->config.run_inline || implementation_->main_thread == std::this_thread::get_id();
}

job_handle job_system::current_job() const noexcept
{
    auto state = current_job_state.lock();
    if (!state || state->scheduler != this)
        return {};
    return job_handle(std::move(state));
}

void job_system::shutdown(job_shutdown_mode mode)
{
    if (!implementation_ || implementation_->stopping.exchange(true, std::memory_order_acq_rel))
        return;
    implementation_->shutdown_mode.store(mode, std::memory_order_release);

    if (mode == job_shutdown_mode::cancel_pending)
    {
        auto cancel_queue = [&](implementation::work_queue& queue) {
            for (auto& state : queue.take_all())
                finish_part(*implementation_, state, job_status::cancelled);
        };
        cancel_queue(implementation_->injection);
        cancel_queue(implementation_->main);
        cancel_queue(implementation_->render);
        cancel_queue(implementation_->io);
        for (auto& worker : implementation_->workers)
            cancel_queue(*worker);
    }
    else if (is_main_thread())
    {
        while (pump_main_thread() != 0)
        {
        }
    }

    implementation_->wake.notify_all();
    for (auto& worker : implementation_->worker_threads)
        if (worker.joinable()) worker.join();
    for (auto& worker : implementation_->io_threads)
        if (worker.joinable()) worker.join();
    if (implementation_->render_thread.joinable())
        implementation_->render_thread.join();
    implementation_->worker_threads.clear();
    implementation_->io_threads.clear();
}

std::size_t job_system::worker_count() const noexcept
{
    return implementation_->workers.size();
}

std::size_t job_system::io_worker_count() const noexcept
{
    return implementation_->io_threads.size();
}

bool job_system::run_inline() const noexcept
{
    return implementation_->config.run_inline;
}

job_system_snapshot job_system::snapshot(bool consume_events) const
{
    job_system_snapshot result{
        .sequence = implementation_->snapshot_sequence.fetch_add(1, std::memory_order_relaxed) + 1,
        .worker_count = implementation_->workers.size(),
        .io_worker_count = implementation_->io_threads.size(),
        .queued_general = implementation_->injection.size(),
        .queued_main = implementation_->main.size(),
        .queued_render = implementation_->render.size(),
        .queued_io = implementation_->io.size(),
        .submitted = implementation_->submitted.load(std::memory_order_relaxed),
        .completed = implementation_->completed.load(std::memory_order_relaxed),
        .stolen = implementation_->stolen.load(std::memory_order_relaxed),
        .cancelled = implementation_->cancelled.load(std::memory_order_relaxed),
        .failed = implementation_->failed.load(std::memory_order_relaxed),
        .dropped_profile_events = implementation_->dropped_profile_events.load(std::memory_order_relaxed)
    };
    for (const auto& worker : implementation_->workers)
        result.queued_general += worker->size();
    std::lock_guard lock(implementation_->profile_mutex);
    result.recent_events.assign(implementation_->profile_events.begin(), implementation_->profile_events.end());
    if (consume_events)
        implementation_->profile_events.clear();
    return result;
}

void job_system::add_coroutine_continuation(
    const std::shared_ptr<detail::job_state>& state,
    std::coroutine_handle<> continuation)
{
    bool resume_immediately{};
    {
        std::lock_guard lock(state->mutex);
        if (job_status_complete(state->status.load(std::memory_order_acquire)))
            resume_immediately = true;
        else
            state->continuations.push_back(continuation);
    }
    if (resume_immediately)
        continuation.resume();
}

void wait_all(const std::vector<job_handle>& handles)
{
    for (const auto& handle : handles)
        handle.wait();
}

} // namespace arc
