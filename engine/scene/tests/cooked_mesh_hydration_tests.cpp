#include <arc/scene/cooked_mesh_hydration.h>

#include <arc/ecs/hierarchy.h>
#include <arc/scene/components.h>

#include <catch2/catch_test_macros.hpp>

#include <array>

namespace
{
arc::render::mesh_data triangle()
{
    arc::render::mesh_data mesh;
    mesh.name = "cooked triangle";
    mesh.vertices.resize(3);
    mesh.vertices[0].position[0] = 1.0f;
    mesh.vertices[1].position[1] = 1.0f;
    mesh.vertices[2].position[2] = 1.0f;
    mesh.indices = {0, 1, 2};
    return mesh;
}

arc::ecs::entity make_entity(arc::ecs::world& world, arc::ecs::entity_guid id)
{
    const auto entity = world.create();
    world.emplace<arc::ecs::persistent_id_component>(entity, id);
    world.emplace<arc::scene::mesh_renderer_component>(entity);
    return entity;
}
} // namespace

TEST_CASE("V3.4d.1 uploads cooked meshes, binds their persistent entity and releases handles")
{
    arc::ecs::world world;
    arc::render::renderer renderer;
    const arc::ecs::entity_guid guid{1, 2};
    const auto entity = make_entity(world, guid);
    const std::array uploads{arc::scene::cooked_mesh_upload{.entity = guid, .mesh = triangle()}};
    arc::scene::cooked_mesh_residencies resources;

    CHECK(arc::scene::hydrate_cooked_scene_meshes(world, renderer, uploads, resources).empty());
    REQUIRE(resources.size() == 1);
    const auto handle = resources.front().mesh;
    REQUIRE(renderer.mesh_alive(handle));
    CHECK(world.get<arc::scene::mesh_renderer_component>(entity).mesh.conventional == handle);
    CHECK(world.get<arc::scene::mesh_renderer_component>(entity).mesh.conventional_lod_count == 1);
    CHECK(world.get<arc::scene::mesh_renderer_component>(entity).representation ==
          arc::render::geometry_representation_policy::conventional);

    arc::scene::release_cooked_scene_meshes(world, renderer, resources);
    CHECK(resources.empty());
    CHECK_FALSE(renderer.mesh_alive(handle));
    CHECK_FALSE(world.get<arc::scene::mesh_renderer_component>(entity).mesh.valid());
    CHECK(arc::scene::hydrate_cooked_scene_meshes(world, renderer, uploads, resources).empty());
    REQUIRE(resources.size() == 1);
    CHECK(renderer.mesh_alive(resources.front().mesh));
    arc::scene::release_cooked_scene_meshes(world, renderer, resources);
}

TEST_CASE("V3.4d.1 refuses missing or duplicated entity GUIDs without publishing partial residency")
{
    arc::ecs::world world;
    arc::render::renderer renderer;
    const arc::ecs::entity_guid known{3, 4};
    const auto entity = make_entity(world, known);
    arc::scene::cooked_mesh_residencies resources;

    const std::array missing{arc::scene::cooked_mesh_upload{.entity = {5, 6}, .mesh = triangle()}};
    CHECK_FALSE(arc::scene::hydrate_cooked_scene_meshes(world, renderer, missing, resources).empty());
    CHECK(resources.empty());
    CHECK_FALSE(world.get<arc::scene::mesh_renderer_component>(entity).mesh.valid());

    const std::array duplicate{arc::scene::cooked_mesh_upload{.entity = known, .mesh = triangle()},
                               arc::scene::cooked_mesh_upload{.entity = known, .mesh = triangle()}};
    CHECK_FALSE(arc::scene::hydrate_cooked_scene_meshes(world, renderer, duplicate, resources).empty());
    CHECK(resources.empty());
    CHECK_FALSE(world.get<arc::scene::mesh_renderer_component>(entity).mesh.valid());
}

TEST_CASE("V3.4d.1 refuses a second generation until the old mesh binding is released")
{
    arc::ecs::world world;
    arc::render::renderer renderer;
    const arc::ecs::entity_guid guid{7, 8};
    make_entity(world, guid);
    const std::array uploads{arc::scene::cooked_mesh_upload{.entity = guid, .mesh = triangle()}};
    arc::scene::cooked_mesh_residencies resources;
    REQUIRE(arc::scene::hydrate_cooked_scene_meshes(world, renderer, uploads, resources).empty());
    CHECK_FALSE(arc::scene::hydrate_cooked_scene_meshes(world, renderer, uploads, resources).empty());
    arc::scene::release_cooked_scene_meshes(world, renderer, resources);
}
