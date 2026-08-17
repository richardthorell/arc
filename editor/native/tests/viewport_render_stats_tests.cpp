#include <arc/editor/editor_state.h>
#include <arc/editor/viewport_render_stats.h>
#include <arc/render/primitives.h>
#include <arc/render/renderer.h>
#include <arc/scene/render_scene.h>

#include <catch2/catch_test_macros.hpp>

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

    const auto mesh_entity = state.scene.create();
    state.scene.emplace<arc::scene::transform_component>(mesh_entity);
    state.scene.emplace<arc::scene::mesh_renderer_component>(mesh_entity, mesh, arc::render::material_handle{});

    const auto instances = state.scene.create();
    state.scene.emplace<arc::scene::transform_component>(instances);
    state.scene.emplace<arc::scene::instance_group_component>(instances, mesh, arc::render::material_handle{}, 3u, true);

    const auto hidden = state.scene.create();
    state.scene.emplace<arc::scene::transform_component>(hidden);
    state.scene.emplace<arc::scene::mesh_renderer_component>(hidden, mesh, arc::render::material_handle{},
                                                              arc::render::geometry_representation_policy::conventional,
                                                              false);

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

    const auto entity = state.scene.create();
    state.scene.emplace<arc::scene::transform_component>(entity);
    state.scene.emplace<arc::scene::mesh_renderer_component>(entity, mesh, arc::render::material_handle{});
    state.scene.emplace<arc::scene::active_component>(entity, false);

    const auto stats = arc::editor::collect_viewport_render_stats(state, renderer);
    CHECK(stats.triangles == 0u);
    CHECK(stats.vertices == 0u);
}
