#include <arc/memory/memory.h>

#include <catch2/catch_test_macros.hpp>

#include <memory_resource>
#include <vector>

TEST_CASE("tracked memory resource records allocations")
{
    arc::tracked_memory_resource resource("test");
    std::pmr::vector<int> values(&resource);

    values.resize(16);
    values.clear();
    values.shrink_to_fit();

    const auto stats = resource.stats();
    REQUIRE(resource.category() == "test");
    REQUIRE(stats.allocation_count >= 1);
    REQUIRE(stats.deallocation_count >= 1);
    REQUIRE(stats.bytes_allocated >= sizeof(int) * 16);
    REQUIRE(stats.bytes_outstanding == 0);
    REQUIRE(stats.peak_bytes_outstanding >= sizeof(int) * 16);
}

TEST_CASE("default tracked memory resource is available")
{
    auto& resource = arc::default_tracked_memory_resource();
    resource.reset_stats();

    void* pointer = resource.allocate(32, alignof(std::max_align_t));
    resource.deallocate(pointer, 32, alignof(std::max_align_t));

    const auto stats = arc::default_memory_stats();
    REQUIRE(stats.allocation_count == 1);
    REQUIRE(stats.deallocation_count == 1);
    REQUIRE(stats.bytes_outstanding == 0);
}
