#include <arc/jobs/jobs.h>

#include <catch2/catch_test_macros.hpp>

#include <atomic>
#include <chrono>
#include <mutex>
#include <numeric>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

TEST_CASE("job system runs submitted work")
{
    arc::job_system jobs(arc::job_system::single_threaded_config());
    int value = 0;

    auto handle = jobs.submit([&]() { value = 42; });
    handle.wait();

    REQUIRE(value == 42);
    REQUIRE(handle.valid());
    REQUIRE(handle.ready());
}

TEST_CASE("job handle rethrows task exceptions")
{
    arc::job_system jobs(arc::job_system::single_threaded_config());
    auto handle = jobs.submit([]() { throw std::runtime_error("boom"); });

    REQUIRE_THROWS_AS(handle.wait(), std::runtime_error);
}

TEST_CASE("parallel_for covers the requested range")
{
    arc::job_system jobs(arc::job_system::single_threaded_config());
    std::vector<int> values(17);

    arc::parallel_for(jobs, 0, values.size(), 4, [&](std::size_t begin, std::size_t end) {
        for (std::size_t index = begin; index < end; ++index)
            values[index] = static_cast<int>(index + 1);
    });

    REQUIRE(std::accumulate(values.begin(), values.end(), 0) == 153);
}

TEST_CASE("worker job system executes queued work")
{
    arc::job_system jobs({ .worker_count = 2, .run_inline = false });
    std::atomic<int> count{ 0 };
    std::vector<arc::job_handle> handles;

    for (int index = 0; index < 32; ++index)
        handles.push_back(jobs.submit([&]() { count.fetch_add(1); }));

    arc::wait_all(handles);

    REQUIRE(count.load() == 32);
}

TEST_CASE("jobs wait for every dependency before running")
{
    arc::job_system jobs({ .worker_count = 3, .run_inline = false, .io_worker_count = 0, .enable_render_thread = false });
    std::mutex mutex;
    std::vector<int> order;

    auto first = jobs.submit({ .name = "first" }, [&] {
        std::lock_guard lock(mutex);
        order.push_back(1);
    });
    auto second = jobs.submit({ .name = "second" }, [&] {
        std::lock_guard lock(mutex);
        order.push_back(2);
    });
    auto joined = jobs.submit({
        .name = "joined",
        .dependencies = { first, second }
    }, [&] {
        std::lock_guard lock(mutex);
        order.push_back(3);
    });

    joined.wait();
    REQUIRE(order.size() == 3);
    REQUIRE(order.back() == 3);
}

TEST_CASE("dependency failure cancels normal continuations and run always executes")
{
    arc::job_system jobs({ .worker_count = 2, .run_inline = false, .io_worker_count = 0, .enable_render_thread = false });
    auto failed = jobs.submit({ .name = "failure" }, [] { throw std::runtime_error("expected"); });
    std::atomic_bool normal_ran{};
    std::atomic_bool cleanup_ran{};
    auto normal = jobs.submit({
        .name = "normal continuation",
        .dependencies = { failed }
    }, [&] { normal_ran = true; });
    auto cleanup = jobs.submit({
        .name = "cleanup continuation",
        .dependencies = { failed },
        .dependency_policy = arc::job_dependency_policy::run_always
    }, [&] { cleanup_ran = true; });

    REQUIRE(normal.wait_result().status == arc::job_status::cancelled);
    REQUIRE(cleanup.wait_result().status == arc::job_status::succeeded);
    REQUIRE_FALSE(normal_ran.load());
    REQUIRE(cleanup_ran.load());
}

TEST_CASE("parent completion includes dynamically submitted children")
{
    arc::job_system jobs({ .worker_count = 3, .run_inline = false, .io_worker_count = 0, .enable_render_thread = false });
    std::atomic_int children{};
    auto parent = jobs.submit({ .name = "parent" }, [&] {
        for (int index = 0; index < 12; ++index)
            jobs.submit_child({ .name = "child" }, [&] { ++children; });
    });

    parent.wait();
    REQUIRE(children.load() == 12);
    REQUIRE(parent.status() == arc::job_status::succeeded);
}

TEST_CASE("child dependencies cannot create an ancestor wait cycle")
{
    arc::job_system jobs(arc::job_system::single_threaded_config());
    bool rejected{};
    auto parent = jobs.submit({ .name = "parent" }, [&] {
        const auto current = jobs.current_job();
        try
        {
            jobs.submit_child({
                .name = "invalid child",
                .dependencies = { current }
            }, [] {});
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
    arc::job_system jobs({ .worker_count = 1, .run_inline = false, .io_worker_count = 0, .enable_render_thread = false });
    std::atomic_bool release{};
    auto blocker = jobs.submit({ .name = "blocker", .priority = arc::job_priority::critical }, [&] {
        while (!release.load())
            std::this_thread::yield();
    });
    std::atomic_bool ran{};
    auto cancelled = jobs.submit({ .name = "cancel me" }, [&] { ran = true; });
    REQUIRE(cancelled.request_cancel());
    release = true;
    blocker.wait();
    REQUIRE(cancelled.wait_result().status == arc::job_status::cancelled);
    REQUIRE_FALSE(ran.load());
}

TEST_CASE("main render and IO affinities stay on their executors")
{
    arc::job_system jobs({ .worker_count = 1, .run_inline = false, .io_worker_count = 1, .enable_render_thread = true });
    jobs.register_main_thread();
    const auto main_id = std::this_thread::get_id();
    std::thread::id observed_main;
    std::thread::id observed_render;
    std::thread::id observed_io;

    auto main = jobs.submit({ .name = "main", .affinity = arc::job_affinity::main_thread }, [&] {
        observed_main = std::this_thread::get_id();
    });
    auto render = jobs.submit({ .name = "render", .affinity = arc::job_affinity::render_thread }, [&] {
        observed_render = std::this_thread::get_id();
    });
    auto io = jobs.submit({ .name = "io", .affinity = arc::job_affinity::io_thread }, [&] {
        observed_io = std::this_thread::get_id();
    });

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
    arc::job_system jobs(arc::job_system::single_threaded_config());
    auto value = jobs.submit_future({ .name = "answer" }, [] { return 42; });
    REQUIRE(value.get() == 42);

    auto failed = jobs.submit_future({ .name = "typed failure" }, []() -> int {
        throw std::runtime_error("typed");
    });
    REQUIRE_THROWS_AS(failed.get(), std::runtime_error);
}

TEST_CASE("fire and forget tasks remain alive and are profiled")
{
    arc::job_system jobs({ .worker_count = 2, .run_inline = false, .io_worker_count = 0, .enable_render_thread = false });
    std::atomic_int completed{};
    for (int index = 0; index < 64; ++index)
        jobs.dispatch({ .name = "detached" }, [&] { ++completed; });

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
arc::job_task<int> await_job(arc::job_system& jobs)
{
    auto future = jobs.submit_future({ .name = "awaited" }, [] { return 21; });
    const int value = co_await future;
    co_return value * 2;
}
}

TEST_CASE("job futures can be awaited by built in coroutine tasks")
{
    arc::job_system jobs({ .worker_count = 2, .run_inline = false, .io_worker_count = 0, .enable_render_thread = false });
    auto task = await_job(jobs);
    REQUIRE(task.get() == 42);
}
#endif
