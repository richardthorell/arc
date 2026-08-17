#include <arc/editor/arc_host.h>
#include <arc/editor/editor_state.h>
#include <arc/editor/viewport_render_stats.h>
#include <arc/render/primitives.h>
#include <arc/render/renderer.h>
#include <arc/scene/render_scene.h>

#include <catch2/catch_test_macros.hpp>

#include <memory>
#include <string>

TEST_CASE("viewport render stats count conventional and instanced geometry")
{
    arc::render::renderer renderer;
    arc::editor::editor_scene_state state;
    arc::scene::prepare_render_scene_queries(state.scene);

    const auto source = arc::render::make_cube_mesh(1.0f);
    const auto expected_triangles = source.indices.size() / 3u;
    const auto expected_vertices = source.vertices.size();
    const auto mesh = renderer.create_mesh(source);
    REQUIRE(mesh.valid());

    arc::scene::mesh_renderer_component visible_mesh;
    visible_mesh.mesh.conventional = mesh;
    const auto mesh_entity = state.scene.create();
    state.scene.emplace<arc::scene::transform_component>(mesh_entity);
    state.scene.emplace<arc::scene::mesh_renderer_component>(mesh_entity, visible_mesh);

    arc::scene::instance_group_component instance_group;
    instance_group.mesh = mesh;
    instance_group.instance_count = 3u;
    const auto instances = state.scene.create();
    state.scene.emplace<arc::scene::transform_component>(instances);
    state.scene.emplace<arc::scene::instance_group_component>(instances, instance_group);

    auto hidden_mesh = visible_mesh;
    hidden_mesh.visible = false;
    const auto hidden = state.scene.create();
    state.scene.emplace<arc::scene::transform_component>(hidden);
    state.scene.emplace<arc::scene::mesh_renderer_component>(hidden, hidden_mesh);

    const auto stats = arc::editor::collect_viewport_render_stats(state, renderer);
    CHECK(stats.triangles == expected_triangles * 4u);
    CHECK(stats.vertices == expected_vertices * 4u);
    CHECK_FALSE(stats.gpu_memory_available);
    CHECK(stats.gpu_memory_used_bytes == 0u);
}

TEST_CASE("viewport render stats ignore inactive renderers")
{
    arc::render::renderer renderer;
    arc::editor::editor_scene_state state;
    arc::scene::prepare_render_scene_queries(state.scene);

    const auto source = arc::render::make_plane_mesh(2.0f);
    const auto mesh = renderer.create_mesh(source);
    REQUIRE(mesh.valid());

    arc::scene::mesh_renderer_component mesh_renderer;
    mesh_renderer.mesh.conventional = mesh;
    const auto entity = state.scene.create();
    state.scene.emplace<arc::scene::transform_component>(entity);
    state.scene.emplace<arc::scene::mesh_renderer_component>(entity, mesh_renderer);
    state.scene.emplace<arc::scene::active_component>(entity, false);

    const auto stats = arc::editor::collect_viewport_render_stats(state, renderer);
    CHECK(stats.triangles == 0u);
    CHECK(stats.vertices == 0u);
}

TEST_CASE("viewport state query transports render telemetry through the host response")
{
    auto renderer = std::make_unique<arc::render::renderer>();
    arc::editor::arc_host_manager manager;
    auto host = manager.acquire(std::move(renderer));

    const auto response = host->query(arc::editor::host_query_envelope{
        .request_id = 1,
        .payload = arc::editor::host_viewport_state_query{.viewport_id = "viewport-1"},
    });
    REQUIRE(response.succeeded);

    const auto& payload = response.payload_json;
    const auto version_fragment =
        "\"viewportTelemetryVersion\":" + std::to_string(arc::editor::viewport_render_stats_schema_version);
    CHECK(payload.find(version_fragment) != std::string::npos);
    CHECK(payload.find("\"viewportId\":\"viewport-1\"") != std::string::npos);
    CHECK(payload.find("\"triangles\":") != std::string::npos);
    CHECK(payload.find("\"verticesComplete\":") != std::string::npos);
    CHECK(payload.find("\"frameIntervalMs\":") != std::string::npos);
    CHECK(payload.find("\"cpuRenderTimeMs\":") != std::string::npos);
}
