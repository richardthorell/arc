#include <arc/render/render.h>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <atomic>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <thread>
#include <memory>
#include <vector>

#if !defined(ARC_RENDER_TEST_ASSET_ROOT)
#define ARC_RENDER_TEST_ASSET_ROOT "assets"
#endif

TEST_CASE("descriptor slots reject stale generations")
{
    arc::render::descriptor_slot_pool pool;
    const auto first = pool.allocate(arc::render::descriptor_resource_type::sampled_image);
    REQUIRE(first.valid());
    REQUIRE(pool.alive(first));

    REQUIRE(pool.release(first));
    REQUIRE_FALSE(pool.alive(first));

    const auto second = pool.allocate(arc::render::descriptor_resource_type::sampled_image);
    REQUIRE(second.index == first.index);
    REQUIRE(second.generation != first.generation);
    REQUIRE(pool.alive(second));
}

TEST_CASE("deferred resource releaser waits for completed frames")
{
    arc::render::deferred_resource_releaser releaser;
    int released = 0;
    releaser.defer(4, [&]() { released += 1; });
    releaser.defer(7, [&]() { released += 10; });

    REQUIRE(releaser.collect(3) == 0);
    REQUIRE(released == 0);
    REQUIRE(releaser.collect(4) == 1);
    REQUIRE(released == 1);
    REQUIRE(releaser.pending_count() == 1);
    REQUIRE(releaser.collect(8) == 1);
    REQUIRE(released == 11);
}

TEST_CASE("frame allocator resets transient allocations")
{
    arc::render::frame_allocator allocator(16);
    auto* first = static_cast<std::uint32_t*>(allocator.allocate(sizeof(std::uint32_t), alignof(std::uint32_t)));
    *first = 42;
    REQUIRE(allocator.used() >= sizeof(std::uint32_t));

    allocator.reset();
    REQUIRE(allocator.used() == 0);
    auto* second = static_cast<std::uint32_t*>(allocator.allocate(sizeof(std::uint32_t), alignof(std::uint32_t)));
    *second = 7;
    REQUIRE(*second == 7);
}

TEST_CASE("GPU upload arena retires ranges by completed frame")
{
    arc::render::gpu_upload_arena arena(256);
    arena.begin_frame(4);
    auto first = arena.try_allocate(80, 16);
    auto second = arena.try_allocate(80, 16);
    REQUIRE(first);
    REQUIRE(second);
    REQUIRE(first.offset % 16 == 0);
    REQUIRE(arena.used() >= 160);

    arena.begin_frame(5);
    auto third = arena.try_allocate(80, 16);
    REQUIRE(third);
    REQUIRE_FALSE(arena.try_allocate(80, 16));
    REQUIRE(arena.retire_completed(3) == 0);
    REQUIRE(arena.retire_completed(4) == 2);

    auto wrapped = arena.try_allocate(80, 16);
    REQUIRE(wrapped);
    REQUIRE(wrapped.frame == 5);
    REQUIRE(arena.peak_used() >= 240);
    REQUIRE(arena.retire_completed(5) == 2);
    REQUIRE(arena.used() == 0);
}

TEST_CASE("GPU upload arena can suballocate persistently mapped backend storage")
{
    std::array<std::byte, 128> mapped{};
    arc::render::gpu_upload_arena arena(mapped);
    arena.begin_frame(9);

    auto allocation = arena.try_allocate(24, 32);
    REQUIRE(allocation);
    REQUIRE(allocation.offset % 32 == 0);
    REQUIRE(allocation.bytes.data() == mapped.data() + allocation.offset);
    allocation.bytes.front() = std::byte{0x5a};
    REQUIRE(mapped[allocation.offset] == std::byte{0x5a});

    REQUIRE(arena.retire_completed(8) == 0);
    REQUIRE(arena.retire_completed(9) == 1);
    REQUIRE(arena.used() == 0);
}

TEST_CASE("pipeline handle cache reuses equivalent keys")
{
    arc::render::pipeline_handle_cache cache;
    arc::render::graphics_pipeline_key key{.vertex_shader = {.index = 1, .generation = 1},
                                           .fragment_shader = {.index = 2, .generation = 1},
                                           .vertex_layout = "pnu",
                                           .color_format = "rgba16f",
                                           .depth_format = "d32",
                                           .depth_test = true,
                                           .depth_write = true};
    arc::render::pipeline_handle pipeline{.index = 9, .generation = 3};

    REQUIRE_FALSE(cache.find(key).valid());
    cache.insert(key, pipeline);
    REQUIRE(cache.find(key) == pipeline);
    key.wireframe = true;
    REQUIRE_FALSE(cache.find(key).valid());
    key.wireframe = false;
    key.permutation.has_normal_texture = true;
    REQUIRE_FALSE(cache.find(key).valid());
}

TEST_CASE("shader permutation keys capture material features")
{
    arc::render::material_descriptor material;
    material.alpha_mode = arc::render::material_alpha_mode::blend;
    material.normal_texture = {.index = 1, .generation = 1};
    material.emissive_texture = {.index = 2, .generation = 1};
    material.clear_coat_factor = 0.5f;

    const auto key = arc::render::make_shader_permutation_key(material, 3, true);
    REQUIRE(key.alpha_mode == arc::render::material_alpha_mode::blend);
    REQUIRE(key.debug_view == 3);
    REQUIRE(key.has_normal_texture);
    REQUIRE(key.has_emissive_texture);
    REQUIRE(key.clear_coat);
    REQUIRE(key.wireframe);

    auto other = key;
    other.wireframe = false;
    REQUIRE(hash_shader_permutation_key(key) != hash_shader_permutation_key(other));
}
