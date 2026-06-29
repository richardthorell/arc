#include <arc/jobs/jobs.h>

#include <catch2/catch_test_macros.hpp>

#include <atomic>
#include <numeric>
#include <stdexcept>
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
