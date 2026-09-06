#include <arc/editor/editor_state.h>

#include <catch2/catch_test_macros.hpp>

namespace
{
arc::render::scene_import_result make_skinned_scene(bool valid_skin = true, bool multiple_mesh_parts = false)
{
    arc::render::scene_import_result imported;
    arc::render::mesh_data mesh;
    mesh.name = "skinned triangle";
    mesh.vertices = {
        {.position = {-1.0f, 0.0f, 0.0f}},
        {.position = {1.0f, 0.0f, 0.0f}},
        {.position = {0.0f, 1.0f, 0.0f}},
    };
    mesh.skin_vertices.resize(mesh.vertices.size());
    mesh.indices = {0, 1, 2};
    imported.meshes.push_back(mesh);
    if (multiple_mesh_parts) imported.meshes.push_back(std::move(mesh));

    arc::render::skeleton_asset skeleton;
    skeleton.name = "CharacterRig";
    skeleton.root_joint = 0;
    skeleton.joints.resize(2);
    skeleton.joints[0].name = "root";
    skeleton.joints[1].name = "child";
    skeleton.joints[1].parent = 0;
    skeleton.joints[1].bind_position = {0.0f, 1.0f, 0.0f};
    auto child_inverse = arc::math::identity<float, 4>();
    child_inverse(1, 3) = -1.0f;
    skeleton.joints[1].inverse_bind_matrix = child_inverse;
    imported.skeletons.push_back(std::move(skeleton));
    imported.nodes.push_back({.name = "Body", .mesh_index = 0, .skin_index = valid_skin ? 0u : 7u});
    if (multiple_mesh_parts) imported.nodes.push_back({.name = "Clothes", .mesh_index = 1, .skin_index = 0u});
    imported.message = "imported";
    return imported;
}

arc::render::scene_import_result make_static_scene()
{
    arc::render::scene_import_result imported;
    arc::render::mesh_data mesh;
    mesh.vertices = {
        {.position = {-1.0f, 0.0f, 0.0f}},
        {.position = {1.0f, 0.0f, 0.0f}},
        {.position = {0.0f, 1.0f, 0.0f}},
    };
    mesh.indices = {0, 1, 2};
    imported.meshes.push_back(std::move(mesh));
    imported.nodes.push_back({.name = "Static", .mesh_index = 0});
    imported.message = "imported";
    return imported;
}
} // namespace

TEST_CASE("imported skinned nodes bind their skeleton palette", "[editor][skeleton][import]")
{
    arc::editor::editor_scene_state state;
    arc::render::renderer renderer;
    const auto result = arc::editor::apply_scene_import_result_to_editor(
        state, renderer, "assets/character.glb", make_skinned_scene(), arc::editor::editor_scene_open_mode::replace);

    REQUIRE(result.succeeded);
    REQUIRE(state.imported_scene_entities.size() == 1u);
    const auto entity = state.imported_scene_entities.front();
    REQUIRE(state.scene.has<arc::scene::skinned_mesh_renderer_component>(entity));
    CHECK_FALSE(state.scene.has<arc::scene::mesh_renderer_component>(entity));
    const auto& skinned = state.scene.get<arc::scene::skinned_mesh_renderer_component>(entity);
    CHECK(skinned.joint_count == 2u);
    REQUIRE(renderer.skin_palette_alive(skinned.skin_matrices));
    const auto* palette = renderer.skin_palette_data_for(skinned.skin_matrices);
    REQUIRE(palette != nullptr);
    REQUIRE(palette->current.size() == 2u);
    CHECK(palette->previous.size() == palette->current.size());
    CHECK(palette->current[0](0, 0) == 1.0f);
    CHECK(palette->current[1](1, 3) == 0.0f);
}

TEST_CASE("mesh parts sharing an imported skeleton share one palette", "[editor][skeleton][import]")
{
    arc::editor::editor_scene_state state;
    arc::render::renderer renderer;
    const auto result = arc::editor::apply_scene_import_result_to_editor(state, renderer, "assets/character.glb",
                                                                         make_skinned_scene(true, true),
                                                                         arc::editor::editor_scene_open_mode::replace);

    REQUIRE(result.succeeded);
    REQUIRE(state.imported_scene_entities.size() == 2u);
    const auto& first = state.scene.get<arc::scene::skinned_mesh_renderer_component>(state.imported_scene_entities[0]);
    const auto& second = state.scene.get<arc::scene::skinned_mesh_renderer_component>(state.imported_scene_entities[1]);
    CHECK(first.skin_matrices == second.skin_matrices);
}

TEST_CASE("invalid imported skin binding degrades to a static mesh", "[editor][skeleton][import]")
{
    arc::editor::editor_scene_state state;
    arc::render::renderer renderer;
    const auto result = arc::editor::apply_scene_import_result_to_editor(
        state, renderer, "assets/broken.glb", make_skinned_scene(false), arc::editor::editor_scene_open_mode::replace);

    REQUIRE(result.succeeded);
    const auto entity = state.imported_scene_entities.front();
    CHECK(state.scene.has<arc::scene::mesh_renderer_component>(entity));
    CHECK_FALSE(state.scene.has<arc::scene::skinned_mesh_renderer_component>(entity));
}

TEST_CASE("replacing an imported scene retires its skin palettes", "[editor][skeleton][import]")
{
    arc::editor::editor_scene_state state;
    arc::render::renderer renderer;
    REQUIRE(arc::editor::apply_scene_import_result_to_editor(state, renderer, "assets/character.glb",
                                                             make_skinned_scene(true, true),
                                                             arc::editor::editor_scene_open_mode::replace)
                .succeeded);
    const auto old_entity = state.imported_scene_entities.front();
    const auto old_palette = state.scene.get<arc::scene::skinned_mesh_renderer_component>(old_entity).skin_matrices;
    REQUIRE(renderer.skin_palette_alive(old_palette));

    REQUIRE(arc::editor::apply_scene_import_result_to_editor(state, renderer, "assets/prop.glb", make_static_scene(),
                                                             arc::editor::editor_scene_open_mode::replace)
                .succeeded);
    CHECK_FALSE(renderer.skin_palette_alive(old_palette));
    REQUIRE(state.imported_scene_entities.size() == 1u);
    CHECK(state.scene.has<arc::scene::mesh_renderer_component>(state.imported_scene_entities.front()));
}

TEST_CASE("imported skinned entities retain editor skeleton metadata", "[editor][skeleton][visualization]")
{
    arc::editor::editor_scene_state state;
    arc::render::renderer renderer;
    REQUIRE(arc::editor::apply_scene_import_result_to_editor(state, renderer, "assets/character.glb",
                                                             make_skinned_scene(),
                                                             arc::editor::editor_scene_open_mode::replace)
                .succeeded);
    REQUIRE(state.imported_scene_entities.size() == 1u);
    const auto entity = state.imported_scene_entities.front();
    const auto* skeleton = arc::editor::find_imported_skeleton(state, entity);
    REQUIRE(skeleton != nullptr);
    CHECK(skeleton->name == "CharacterRig");
    REQUIRE(skeleton->joints.size() == 2u);
    CHECK(skeleton->joints[1].parent == 0);

    REQUIRE(arc::editor::apply_scene_import_result_to_editor(state, renderer, "assets/prop.glb", make_static_scene(),
                                                             arc::editor::editor_scene_open_mode::replace)
                .succeeded);
    CHECK(state.imported_skeletons.empty());
}
