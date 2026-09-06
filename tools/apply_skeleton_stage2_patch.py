#!/usr/bin/env python3
from pathlib import Path

path = Path('editor/native/src/editor_state.cpp')
text = path.read_text()

old = '#include <arc/render/primitives.h>\n\n#include <algorithm>\n'
new = '#include <arc/render/primitives.h>\n#include <arc/scene/transforms.h>\n\n#include <algorithm>\n'
if old not in text: raise SystemExit('include marker not found')
text = text.replace(old, new, 1)

old = '#include <cstdio>\n#include <limits>\n'
new = '#include <cstdio>\n#include <functional>\n#include <limits>\n'
if old not in text: raise SystemExit('functional include marker not found')
text = text.replace(old, new, 1)

marker = '''render::material_handle material_from_import(editor_scene_state& state, render::renderer& renderer,
                                             const render::material_import& imported,
                                             const std::vector<render::texture_handle>& textures)
'''
helper = r'''render::skin_palette_data bind_pose_palette(const render::skeleton_asset& skeleton)
{
    render::skin_palette_data palette;
    palette.name = skeleton.name.empty() ? "Imported Skeleton" : skeleton.name;
    palette.current.resize(skeleton.joints.size(), math::identity<float, 4>());
    std::vector<math::matrix4f> joint_world(skeleton.joints.size(), math::identity<float, 4>());
    std::vector<std::uint8_t> state(skeleton.joints.size());

    std::function<bool(std::size_t)> evaluate = [&](std::size_t index)
    {
        if (index >= skeleton.joints.size()) return false;
        if (state[index] == 2u) return true;
        if (state[index] == 1u) return false;
        state[index] = 1u;

        const auto& joint = skeleton.joints[index];
        scene::transform_component local_transform;
        local_transform.position = joint.bind_position;
        local_transform.rotation = joint.bind_rotation;
        local_transform.scale = joint.bind_scale;
        auto world = scene::local_matrix(local_transform);
        if (joint.parent >= 0)
        {
            const auto parent = static_cast<std::size_t>(joint.parent);
            if (parent >= skeleton.joints.size() || !evaluate(parent)) return false;
            world = math::matmul(joint_world[parent], world);
        }
        joint_world[index] = world;
        palette.current[index] = math::matmul(world, joint.inverse_bind_matrix);
        state[index] = 2u;
        return true;
    };

    for (std::size_t joint = 0; joint < skeleton.joints.size(); ++joint)
        if (!evaluate(joint)) return {};
    palette.previous = palette.current;
    palette.content_revision = 1;
    return palette;
}

'''
if marker not in text: raise SystemExit('material helper marker not found')
text = text.replace(marker, helper + marker, 1)

old = '''void clear_imported_content(editor_scene_state& state)
{
    destroy_entity_if_alive(state, state.mesh_entity);
'''
new = '''void clear_imported_content(editor_scene_state& state, render::renderer* renderer = nullptr)
{
    if (renderer)
    {
        std::vector<render::buffer_handle> palettes;
        for (const auto entity : state.imported_scene_entities)
        {
            const auto* skinned = state.scene.try_get<scene::skinned_mesh_renderer_component>(entity);
            if (!skinned || !skinned->skin_matrices.valid()) continue;
            if (std::find(palettes.begin(), palettes.end(), skinned->skin_matrices) == palettes.end())
                palettes.push_back(skinned->skin_matrices);
        }
        for (const auto palette : palettes)
            if (renderer->skin_palette_alive(palette)) renderer->destroy_skin_palette(palette);
    }
    destroy_entity_if_alive(state, state.mesh_entity);
'''
if old not in text: raise SystemExit('clear imported marker not found')
text = text.replace(old, new, 1)

old = '    if (mode == editor_scene_open_mode::replace) clear_imported_content(scene);\n'
new = '    if (mode == editor_scene_open_mode::replace) clear_imported_content(scene, &renderer);\n'
if old not in text: raise SystemExit('clear call marker not found')
text = text.replace(old, new, 1)

old = '''    std::size_t created{};
    ecs::entity first_entity{};
    for (const auto& node : imported.nodes)
'''
new = '''    std::vector<render::buffer_handle> skin_palettes(imported.skeletons.size());
    std::vector<bool> skin_palette_attempted(imported.skeletons.size());
    const auto palette_for_skin = [&](std::size_t skin_index) -> render::buffer_handle
    {
        if (skin_index >= imported.skeletons.size()) return {};
        if (skin_palette_attempted[skin_index]) return skin_palettes[skin_index];
        skin_palette_attempted[skin_index] = true;
        const auto& skeleton = imported.skeletons[skin_index];
        if (!skeleton.valid()) return {};
        auto palette = bind_pose_palette(skeleton);
        if (!palette.valid()) return {};
        skin_palettes[skin_index] = renderer.create_skin_palette(std::move(palette));
        return skin_palettes[skin_index];
    };

    std::size_t created{};
    ecs::entity first_entity{};
    for (const auto& node : imported.nodes)
'''
if old not in text: raise SystemExit('node loop marker not found')
text = text.replace(old, new, 1)

old = '''        const auto& imported_mesh = imported.meshes[node.mesh_index];
        const bool has_skin_stream = !imported_mesh.skin_vertices.empty() &&
                                     imported_mesh.skin_vertices.size() == imported_mesh.vertices.size();
        const bool has_skeleton = node.skin_index < imported.skeletons.size() && imported.skeletons[node.skin_index].valid();
        bool skinned_bound{};
        if (has_skin_stream && has_skeleton)
        {
            auto palette = bind_pose_palette(imported.skeletons[node.skin_index]);
            if (palette.valid())
            {
                const auto palette_handle = renderer.create_skin_palette(std::move(palette));
                if (palette_handle.valid())
                {
                    scene::skinned_mesh_renderer_component renderer_component;
                    renderer_component.mesh = meshes[node.mesh_index];
                    renderer_component.material = materials[material_index];
                    renderer_component.skin_matrices = palette_handle;
                    renderer_component.joint_count =
                        static_cast<std::uint32_t>(imported.skeletons[node.skin_index].joints.size());
                    scene.scene.emplace<scene::skinned_mesh_renderer_component>(entity, renderer_component);
                    skinned_bound = true;
                }
            }
        }
'''
new = '''        const auto& imported_mesh = imported.meshes[node.mesh_index];
        const bool has_skin_stream = !imported_mesh.skin_vertices.empty() &&
                                     imported_mesh.skin_vertices.size() == imported_mesh.vertices.size();
        const bool has_skeleton =
            node.skin_index < imported.skeletons.size() && imported.skeletons[node.skin_index].valid();
        bool skinned_bound{};
        if (has_skin_stream && has_skeleton)
        {
            const auto palette_handle = palette_for_skin(node.skin_index);
            if (palette_handle.valid())
            {
                scene::skinned_mesh_renderer_component renderer_component;
                renderer_component.mesh = meshes[node.mesh_index];
                renderer_component.material = materials[material_index];
                renderer_component.skin_matrices = palette_handle;
                renderer_component.joint_count =
                    static_cast<std::uint32_t>(imported.skeletons[node.skin_index].joints.size());
                scene.scene.emplace<scene::skinned_mesh_renderer_component>(entity, renderer_component);
                skinned_bound = true;
            }
        }
'''
if old not in text: raise SystemExit('renderer component marker not found')
text = text.replace(old, new, 1)

path.write_text(text)

test = Path('editor/native/tests/skinned_scene_import_tests.cpp')
test.write_text(r'''#include <arc/editor/editor_state.h>

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
}

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
    const auto result = arc::editor::apply_scene_import_result_to_editor(
        state, renderer, "assets/character.glb", make_skinned_scene(true, true),
        arc::editor::editor_scene_open_mode::replace);

    REQUIRE(result.succeeded);
    REQUIRE(state.imported_scene_entities.size() == 2u);
    const auto& first =
        state.scene.get<arc::scene::skinned_mesh_renderer_component>(state.imported_scene_entities[0]);
    const auto& second =
        state.scene.get<arc::scene::skinned_mesh_renderer_component>(state.imported_scene_entities[1]);
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
    REQUIRE(arc::editor::apply_scene_import_result_to_editor(
                state, renderer, "assets/character.glb", make_skinned_scene(true, true),
                arc::editor::editor_scene_open_mode::replace)
                .succeeded);
    const auto old_entity = state.imported_scene_entities.front();
    const auto old_palette = state.scene.get<arc::scene::skinned_mesh_renderer_component>(old_entity).skin_matrices;
    REQUIRE(renderer.skin_palette_alive(old_palette));

    REQUIRE(arc::editor::apply_scene_import_result_to_editor(
                state, renderer, "assets/prop.glb", make_static_scene(), arc::editor::editor_scene_open_mode::replace)
                .succeeded);
    CHECK_FALSE(renderer.skin_palette_alive(old_palette));
    REQUIRE(state.imported_scene_entities.size() == 1u);
    CHECK(state.scene.has<arc::scene::mesh_renderer_component>(state.imported_scene_entities.front()));
}
''')
