#include <arc/editor/arc_host.h>
#include <arc/editor/editor_state.h>
#include <arc/editor/procedural_mesh.h>
#include <arc/render/render.h>

#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <memory>
#include <string>
#include <variant>

TEST_CASE("procedural mesh variants generate denser geometry as subdivisions increase")
{
    arc::editor::procedural_mesh_component sphere{arc::editor::sphere_mesh_parameters{}};
    const auto default_mesh = arc::editor::make_procedural_mesh(sphere.parameters);

    REQUIRE(arc::editor::set_procedural_mesh_parameter(sphere, "segments", 64));
    REQUIRE(arc::editor::set_procedural_mesh_parameter(sphere, "rings", 32));
    const auto dense_mesh = arc::editor::make_procedural_mesh(sphere.parameters);

    CHECK(std::holds_alternative<arc::editor::sphere_mesh_parameters>(sphere.parameters));
    CHECK(std::get<arc::editor::sphere_mesh_parameters>(sphere.parameters).segments == 64);
    CHECK(std::get<arc::editor::sphere_mesh_parameters>(sphere.parameters).rings == 32);
    CHECK(dense_mesh.vertices.size() > default_mesh.vertices.size());
    CHECK(dense_mesh.indices.size() > default_mesh.indices.size());

    arc::editor::cube_mesh_parameters cube;
    cube.segments_x = 2;
    cube.segments_y = 3;
    cube.segments_z = 4;
    const auto subdivided_cube = arc::editor::make_procedural_mesh(cube);
    const auto default_cube = arc::editor::make_procedural_mesh(arc::editor::cube_mesh_parameters{});
    CHECK(subdivided_cube.vertices.size() > default_cube.vertices.size());
    CHECK(subdivided_cube.indices.size() > default_cube.indices.size());
}

TEST_CASE("native host edits procedural mesh parameters with history and persisted authoring data")
{
    auto renderer = std::make_unique<arc::render::renderer>();
    arc::editor::arc_host_manager manager;
    auto host = manager.acquire(std::move(renderer));

    REQUIRE(host->open_project({.name = "Procedural Mesh Parameters"}, {}).succeeded);
    REQUIRE(host
                ->execute({.request_id = 1,
                           .payload = arc::editor::host_create_entity_command{
                               .kind = arc::editor::host_create_entity_kind::sphere}})
                .succeeded);

    const auto initial = host->selected_entity_snapshot();
    const arc::ecs::entity entity{initial.entity.index, initial.entity.generation};
    auto* initial_component = host->scene_state().scene.try_get<arc::editor::procedural_mesh_component>(entity);
    REQUIRE(initial_component != nullptr);
    REQUIRE(std::holds_alternative<arc::editor::sphere_mesh_parameters>(initial_component->parameters));
    CHECK(std::get<arc::editor::sphere_mesh_parameters>(initial_component->parameters).segments == 32);

    REQUIRE(host
                ->execute({.request_id = 2,
                           .payload = arc::editor::host_set_entity_material_command{
                               .entity = initial.entity,
                               .path = "__arc_primitive_parameter__/segments/64"}})
                .succeeded);

    auto* edited_component = host->scene_state().scene.try_get<arc::editor::procedural_mesh_component>(entity);
    REQUIRE(edited_component != nullptr);
    CHECK(std::get<arc::editor::sphere_mesh_parameters>(edited_component->parameters).segments == 64);

    const auto query = host->query({.request_id = 3, .payload = arc::editor::host_selected_entity_query{}});
    REQUIRE(query.succeeded);
    CHECK(query.payload_json.find("\"proceduralMesh\"") != std::string::npos);
    CHECK(query.payload_json.find("\"segments\":64") != std::string::npos);

    const auto guid = arc::editor::entity_guid_of(host->scene_state(), entity);
    const auto persisted = std::find_if(
        host->scene_state().unknown_component_records.begin(), host->scene_state().unknown_component_records.end(),
        [guid](const auto& record) { return record.first == guid; });
    REQUIRE(persisted != host->scene_state().unknown_component_records.end());
    CHECK(persisted->second.find("ProceduralMesh") != std::string::npos);
    CHECK(persisted->second.find("\"segments\":64") != std::string::npos);

    REQUIRE(host->execute({.request_id = 4, .payload = arc::editor::host_history_undo_command{}}).succeeded);
    auto* restored_component = host->scene_state().scene.try_get<arc::editor::procedural_mesh_component>(entity);
    REQUIRE(restored_component != nullptr);
    CHECK(std::get<arc::editor::sphere_mesh_parameters>(restored_component->parameters).segments == 32);
}
