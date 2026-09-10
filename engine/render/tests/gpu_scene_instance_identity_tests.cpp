#include <arc/render/gpu_scene.h>
#include <arc/render/render_world.h>

#include <catch2/catch_test_macros.hpp>

TEST_CASE("gpu scene supports multiple stable render instances for one object")
{
    arc::render::gpu_scene gpu_scene;
    arc::render::render_world_packet packet;
    packet.gpu_scene_world_id = 7u;
    packet.world_epoch = 1u;

    const arc::render::render_object_id object{.index = 42u, .generation = 3u};
    packet.items.push_back({.mesh = {.index = 1u, .generation = 1u}, .instance_id = 100u, .object_id = object});
    packet.items.push_back({.mesh = {.index = 2u, .generation = 1u}, .instance_id = 200u, .object_id = object});

    const auto first = gpu_scene.synchronize(packet, 1u);
    REQUIRE(first.active_instance_count == 2u);
    REQUIRE(packet.items[0].gpu_scene_instance.valid());
    REQUIRE(packet.items[1].gpu_scene_instance.valid());
    CHECK(packet.items[0].gpu_scene_instance != packet.items[1].gpu_scene_instance);

    const auto first_handle = packet.items[0].gpu_scene_instance;
    const auto second_handle = packet.items[1].gpu_scene_instance;
    packet.items[0].mesh = {.index = 3u, .generation = 2u};
    const auto second = gpu_scene.synchronize(packet, 2u);
    CHECK(second.active_instance_count == 2u);
    CHECK(packet.items[0].gpu_scene_instance == first_handle);
    CHECK(packet.items[1].gpu_scene_instance == second_handle);
}

TEST_CASE("gpu scene differentiates virtual roots from the same object by instance id")
{
    arc::render::gpu_scene gpu_scene;
    arc::render::render_world_packet packet;
    packet.gpu_scene_world_id = 9u;
    packet.world_epoch = 1u;

    const arc::render::render_object_id object{.index = 4u, .generation = 2u};
    packet.virtual_items.push_back(
        {.mesh = {.index = 10u, .generation = 1u}, .root_node = 0u, .instance_id = 11u, .object_id = object});
    packet.virtual_items.push_back(
        {.mesh = {.index = 11u, .generation = 1u}, .root_node = 0u, .instance_id = 12u, .object_id = object});

    const auto batch = gpu_scene.synchronize(packet, 1u);
    REQUIRE(batch.active_instance_count == 2u);
    CHECK(packet.virtual_items[0].gpu_scene_instance != packet.virtual_items[1].gpu_scene_instance);
}
