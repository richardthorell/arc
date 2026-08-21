#include <arc/editor/arc_host.h>
#include <arc/editor/editor_state.h>
#include <arc/render/render.h>

#include <catch2/catch_test_macros.hpp>

#include <memory>

TEST_CASE("native host exposes and reassigns procedural primitive meshes")
{
    auto renderer = std::make_unique<arc::render::renderer>();
    arc::editor::arc_host_manager manager;
    auto host = manager.acquire(std::move(renderer));

    REQUIRE(host->open_project({.name = "Procedural Mesh Picker"}, {}).succeeded);
    REQUIRE(host->execute(
                    {.request_id = 1,
                     .payload =
                         arc::editor::host_create_entity_command{.kind = arc::editor::host_create_entity_kind::sphere}})
                .succeeded);

    const auto initial = host->selected_entity_snapshot();
    REQUIRE(initial.mesh_renderer.has_value());
    REQUIRE(initial.mesh_renderer->has_mesh);
    CHECK_FALSE(initial.mesh_renderer->asset_backed_mesh);
    CHECK(initial.mesh_renderer->mesh_name == "Sphere");
    CHECK(initial.mesh_renderer->mesh_path == "arc://primitive/sphere");

    REQUIRE(host->execute({.request_id = 2,
                           .payload = arc::editor::host_set_entity_material_command{.entity = initial.entity,
                                                                                    .path = "__arc_primitive__/cube"}})
                .succeeded);

    const auto assigned = host->selected_entity_snapshot();
    REQUIRE(assigned.mesh_renderer.has_value());
    REQUIRE(assigned.mesh_renderer->has_mesh);
    CHECK_FALSE(assigned.mesh_renderer->asset_backed_mesh);
    CHECK(assigned.mesh_renderer->mesh_name == "Cube");
    CHECK(assigned.mesh_renderer->mesh_path == "arc://primitive/cube");
    CHECK(assigned.mesh_renderer->has_material);
}
