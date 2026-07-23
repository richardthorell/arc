#include <arc/memory/memory.h>

#include <catch2/catch_test_macros.hpp>

#include <memory_resource>
#include <array>
#include <cstdint>
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

TEST_CASE("memory system tracks domains tags and world leaks")
{
    arc::memory_system system({
        .physical_memory_override = 1024 * 1024,
        .track_live_allocations = true,
        .capture_call_stacks = false
    });
    arc::system_memory_resource resource(
        system,
        arc::memory_domain::world,
        arc::make_memory_tag("scene.transforms"),
        42);

    void* pointer = resource.allocate(128, 16);
    const auto active = system.snapshot();
    REQUIRE(active.global_bytes_outstanding == 128);
    REQUIRE(active.tracked_allocation_count == 1);
    REQUIRE(active.tags.size() == 1);
    REQUIRE(active.tags.front().tag.name == "scene.transforms");
    REQUIRE(active.tags.front().domain == arc::memory_domain::world);

    const auto world_leaks = system.leaks(42);
    REQUIRE(world_leaks.size() == 1);
    REQUIRE(world_leaks.front().bytes == 128);

    resource.deallocate(pointer, 128, 16);
    REQUIRE(system.snapshot().global_bytes_outstanding == 0);
    REQUIRE(system.leaks().empty());
}

TEST_CASE("memory hard budgets invoke pressure handlers before rejecting")
{
    arc::memory_system system({
        .physical_memory_override = 1024,
        .cpu_soft_budget_fraction = 0.5f,
        .cpu_hard_budget_fraction = 0.75f,
        .track_live_allocations = false,
        .capture_call_stacks = false
    });
    system.set_budget(arc::memory_domain::assets, { .soft_limit = 64, .hard_limit = 96 });

    std::size_t pressure_count{};
    system.add_pressure_handler([&](arc::memory_pressure_level level, arc::memory_domain domain, std::size_t bytes) {
        REQUIRE(level == arc::memory_pressure_level::hard);
        REQUIRE(domain == arc::memory_domain::assets);
        REQUIRE(bytes == 128);
        ++pressure_count;
    });

    REQUIRE(system.try_allocate(
        128,
        alignof(std::max_align_t),
        arc::memory_domain::assets,
        arc::make_memory_tag("too-large")) == nullptr);
    REQUIRE(pressure_count == 1);
    REQUIRE(system.snapshot().pressure_event_count == 1);
}

TEST_CASE("linear arena growth preserves pointers and supports marks")
{
    arc::linear_arena arena(64);
    auto* first = static_cast<std::uint32_t*>(arena.allocate(sizeof(std::uint32_t), alignof(std::uint32_t)));
    *first = 0xdeadbeefu;
    const auto mark = arena.mark();

    REQUIRE(arena.allocate(4096, 64) != nullptr);
    REQUIRE(*first == 0xdeadbeefu);
    REQUIRE(arena.capacity() >= 4096);
    REQUIRE(arena.peak_used() >= 4096);
    REQUIRE(arena.rewind(mark));
    REQUIRE(*first == 0xdeadbeefu);

    const auto generation = arena.generation();
    arena.reset();
    REQUIRE(arena.used() == 0);
    REQUIRE(arena.generation() == generation + 1);
    REQUIRE_FALSE(arena.rewind(mark));
}

TEST_CASE("network packet pool serves every standard size class")
{
    arc::network_packet_pool pool;
    constexpr std::array<std::size_t, 5> sizes{ 128, 500, 1200, 4000, 60 * 1024 };
    std::array<void*, sizes.size()> pointers{};
    for (std::size_t index = 0; index < sizes.size(); ++index)
    {
        pointers[index] = pool.try_allocate(sizes[index]);
        REQUIRE(pointers[index] != nullptr);
    }
    REQUIRE(pool.outstanding_bytes() >= 128 + 500 + 1200 + 4000 + 60 * 1024);
    for (std::size_t index = 0; index < sizes.size(); ++index)
        pool.deallocate(pointers[index], sizes[index]);
    REQUIRE(pool.outstanding_bytes() == 0);
}

TEST_CASE("allocation tag scopes restore their previous value")
{
    const auto original = arc::current_memory_tag();
    {
        arc::allocation_tag_scope outer("outer");
        REQUIRE(arc::current_memory_tag().name == "outer");
        {
            arc::allocation_tag_scope inner("inner");
            REQUIRE(arc::current_memory_tag().name == "inner");
        }
        REQUIRE(arc::current_memory_tag().name == "outer");
    }
    REQUIRE(arc::current_memory_tag().id == original.id);
}
