#include <arc/jobs/jobs.h>

#include <catch2/catch_test_macros.hpp>

#include <array>
#include <atomic>
#include <chrono>
#include <mutex>
#include <numeric>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>

namespace
{

arc::jobs::job_system_config worker_config(std::size_t workers, std::size_t io_workers = 0, bool render_thread = false)
{
    return {.worker_count = workers,
            .run_inline = false,
            .io_worker_count = io_workers,
            .enable_render_thread = render_thread,
            .profile_event_capacity = 8192,
            .memory = nullptr};
}

arc::jobs::job_descriptor job(std::string_view name, arc::jobs::job_priority priority = arc::jobs::job_priority::normal,
                              arc::jobs::job_affinity affinity = arc::jobs::job_affinity::any_worker)
{
    return {.name = name,
            .priority = priority,
            .affinity = affinity,
            .dependencies = {},
            .dependency_view = {},
            .parent = {},
            .cancellation = {},
            .dependency_policy = arc::jobs::job_dependency_policy::cancel_on_failure};
}

arc::jobs::job_descriptor
dependent_job(std::string_view name, std::vector<arc::jobs::job_handle> dependencies,
              arc::jobs::job_dependency_policy policy = arc::jobs::job_dependency_policy::cancel_on_failure)
{
    auto descriptor = job(name);
    descriptor.dependencies = std::move(dependencies);
    descriptor.dependency_policy = policy;
    return descriptor;
}

arc::jobs::job_descriptor dependent_job(std::string_view name, std::span<const arc::jobs::job_handle> dependencies)
{
    auto descriptor = job(name);
    descriptor.dependency_view = dependencies;
    return descriptor;
}

} // namespace

TEST_CASE("job system runs submitted work")
{
    arc::jobs::job_system jobs(arc::jobs::job_system::single_threaded_config());
    int value = 0;

    auto handle = jobs.submit([&]() { value = 42; });
    handle.wait();

    REQUIRE(value == 42);
    REQUIRE(handle.valid());
    REQUIRE(handle.ready());
}

TEST_CASE("job handle rethrows task exceptions")
{
    arc::jobs::job_system jobs(arc::jobs::job_system::single_threaded_config());
    auto handle = jobs.submit([]() { throw std::runtime_error("boom"); });

    REQUIRE_THROWS_AS(handle.wait(), std::runtime_error);
}

TEST_CASE("parallel_for covers the requested range")
{
    arc::jobs::job_system jobs(arc::jobs::job_system::single_threaded_config());
    std::vector<int> values(17);

    arc::jobs::parallel_for(jobs, 0, values.size(), 4,
                            [&](std::size_t begin, std::size_t end)
                            {
                                for (std::size_t index = begin; index < end; ++index)
                                    values[index] = static_cast<int>(index + 1);
                            });

    REQUIRE(std::accumulate(values.begin(), values.end(), 0) == 153);
}

TEST_CASE("worker job system executes queued work")
{
    arc::jobs::job_system jobs(worker_config(2));
    std::atomic<int> count{0};
    std::vector<arc::jobs::job_handle> handles;

    for (int index = 0; index < 32; ++index)
        handles.push_back(jobs.submit([&]() { count.fetch_add(1); }));

    arc::jobs::wait_all(handles);

    REQUIRE(count.load() == 32);
}

TEST_CASE("jobs wait for every dependency before running")
{
    arc::jobs::job_system jobs(worker_config(3));
    std::mutex mutex;
    std::vector<int> order;

    auto first = jobs.submit(job("first"),
                             [&]
                             {
                                 std::lock_guard lock(mutex);
                                 order.push_back(1);
                             });
    auto second = jobs.submit(job("second"),
                              [&]
                              {
                                  std::lock_guard lock(mutex);
                                  order.push_back(2);
                              });
    auto joined = jobs.submit(dependent_job("joined", {first, second}),
                              [&]
                              {
                                  std::lock_guard lock(mutex);
                                  order.push_back(3);
                              });

    joined.wait();
    REQUIRE(order.size() == 3);
    REQUIRE(order.back() == 3);
}

TEST_CASE("dependency registration is atomic with fast prerequisites")
{
    arc::jobs::job_system jobs(worker_config(4));

    for (std::size_t iteration = 0; iteration < 512; ++iteration)
    {
        std::atomic_size_t completed{};
        std::array<arc::jobs::job_handle, 32> prerequisites;
        for (auto& prerequisite : prerequisites)
            prerequisite = jobs.submit([&] { completed.fetch_add(1, std::memory_order_relaxed); });

        std::atomic_bool ran_after_all{};
        const auto joined =
            jobs.submit(dependent_job("fast fan-in", std::span<const arc::jobs::job_handle>{prerequisites}), [&]
                        { ran_after_all.store(completed.load(std::memory_order_acquire) == prerequisites.size()); });
        joined.wait();
        REQUIRE(ran_after_all.load(std::memory_order_acquire));
    }
}

TEST_CASE("dependency failure cancels normal continuations and run always executes")
{
    arc::jobs::job_system jobs(worker_config(2));
    auto failed = jobs.submit(job("failure"), [] { throw std::runtime_error("expected"); });
    std::atomic_bool normal_ran{};
    std::atomic_bool cleanup_ran{};
    auto normal = jobs.submit(dependent_job("normal continuation", {failed}), [&] { normal_ran = true; });
    auto cleanup =
        jobs.submit(dependent_job("cleanup continuation", {failed}, arc::jobs::job_dependency_policy::run_always),
                    [&] { cleanup_ran = true; });

    REQUIRE(normal.wait_result().status == arc::jobs::job_status::cancelled);
    REQUIRE(cleanup.wait_result().status == arc::jobs::job_status::succeeded);
    REQUIRE_FALSE(normal_ran.load());
    REQUIRE(cleanup_ran.load());
}

TEST_CASE("parent completion includes dynamically submitted children")
{
    arc::jobs::job_system jobs(worker_config(3));
    std::atomic_int children{};
    auto parent = jobs.submit(job("parent"),
                              [&]
                              {
                                  for (int index = 0; index < 12; ++index)
                                      jobs.submit_child(job("child"), [&] { ++children; });
                              });

    parent.wait();
    REQUIRE(children.load() == 12);
    REQUIRE(parent.status() == arc::jobs::job_status::succeeded);
}

TEST_CASE("child dependencies cannot create an ancestor wait cycle")
{
    arc::jobs::job_system jobs(arc::jobs::job_system::single_threaded_config());
    bool rejected{};
    auto parent = jobs.submit(job("parent"),
                              [&]
                              {
                                  const auto current = jobs.current_job();
                                  try
                                  {
                                      jobs.submit_child(dependent_job("invalid child", {current}), [] {});
                                  }
                                  catch (const std::invalid_argument&)
                                  {
                                      rejected = true;
                                  }
                              });
    parent.wait();
    REQUIRE(rejected);
}

TEST_CASE("queued cancellation prevents task execution")
{
    arc::jobs::job_system jobs(worker_config(1));
    std::atomic_bool release{};
    auto blocker = jobs.submit(job("blocker", arc::jobs::job_priority::critical),
                               [&]
                               {
                                   while (!release.load())
                                       std::this_thread::yield();
                               });
    std::atomic_bool ran{};
    auto cancelled = jobs.submit(job("cancel me"), [&] { ran = true; });
    REQUIRE(cancelled.request_cancel());
    release = true;
    blocker.wait();
    REQUIRE(cancelled.wait_result().status == arc::jobs::job_status::cancelled);
    REQUIRE_FALSE(ran.load());
}

TEST_CASE("main render and IO affinities stay on their executors")
{
    arc::jobs::job_system jobs(worker_config(1, 1, true));
    jobs.register_main_thread();
    const auto main_id = std::this_thread::get_id();
    std::thread::id observed_main;
    std::thread::id observed_render;
    std::thread::id observed_io;

    auto main = jobs.submit(job("main", arc::jobs::job_priority::normal, arc::jobs::job_affinity::main_thread),
                            [&] { observed_main = std::this_thread::get_id(); });
    auto render = jobs.submit(job("render", arc::jobs::job_priority::normal, arc::jobs::job_affinity::render_thread),
                              [&] { observed_render = std::this_thread::get_id(); });
    auto io = jobs.submit(job("io", arc::jobs::job_priority::normal, arc::jobs::job_affinity::io_thread),
                          [&] { observed_io = std::this_thread::get_id(); });

    main.wait();
    render.wait();
    io.wait();
    REQUIRE(observed_main == main_id);
    REQUIRE(observed_render != main_id);
    REQUIRE(observed_io != main_id);
    REQUIRE(observed_render != observed_io);
}

TEST_CASE("typed futures return values and propagate failures")
{
    arc::jobs::job_system jobs(arc::jobs::job_system::single_threaded_config());
    auto value = jobs.submit_future(job("answer"), [] { return 42; });
    REQUIRE(value.get() == 42);

    auto failed = jobs.submit_future(job("typed failure"), []() -> int { throw std::runtime_error("typed"); });
    REQUIRE_THROWS_AS(failed.get(), std::runtime_error);
}

TEST_CASE("fire and forget tasks remain alive and are profiled")
{
    arc::jobs::job_system jobs(worker_config(2));
    std::atomic_int completed{};
    for (int index = 0; index < 64; ++index)
        jobs.dispatch(job("detached"), [&] { ++completed; });

    while (completed.load() != 64)
        std::this_thread::yield();
    jobs.shutdown();
    const auto snapshot = jobs.snapshot();
    REQUIRE(snapshot.submitted == 64);
    REQUIRE(snapshot.completed == 64);
    REQUIRE(snapshot.recent_events.size() == 64);
}

#if defined(ARC_ENABLE_JOB_COROUTINES)
namespace
{
arc::jobs::job_task<int> await_job(arc::jobs::job_system& jobs)
{
    auto future = jobs.submit_future(job("awaited"), [] { return 21; });
    const int value = co_await future;
    co_return value * 2;
}
} // namespace

TEST_CASE("job futures can be awaited by built in coroutine tasks")
{
    arc::jobs::job_system jobs(worker_config(2));
    auto task = await_job(jobs);
    REQUIRE(task.get() == 42);
}
#endif
