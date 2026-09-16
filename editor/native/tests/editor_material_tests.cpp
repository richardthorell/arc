#include <arc/editor/arc_host.h>
#include <arc/editor/editor_console.h>
#include <arc/editor/editor_interaction.h>
#include <arc/editor/editor_gizmo.h>
#include <arc/editor/editor_state.h>
#include <arc/editor/editor_viewport.h>
#include <arc/editor/material_asset.h>
#include <arc/editor/material_library.h>
#include <arc/editor/material_preview.h>
#include <arc/editor/scene_document.h>
#include <arc/editor/world_environment_host.h>
#include <arc/project/project.h>
#include <arc/render/primitives.h>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <array>
#include <charconv>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <limits>
#include <string>
#include <string_view>
#include <thread>
#include <variant>

TEST_CASE("editor material assets load save and round trip")
{
    const auto root = std::filesystem::temp_directory_path() / "arc_editor_material_asset_tests";
    std::filesystem::create_directories(root / "materials");

    auto asset = arc::editor::make_default_material_asset("Bronze");
    asset.path = root / "materials" / "bronze.arcmat";
    asset.material.base_color = {0.8f, 0.42f, 0.18f, 1.0f};
    asset.material.metallic = 0.75f;
    asset.material.roughness = 0.32f;
    asset.material.normal_scale = 0.85f;
    asset.textures.base_color = "textures/bronze_base.png";
    asset.textures.normal = "textures/bronze_n.png";

    std::string message;
    REQUIRE(arc::editor::save_material_asset(asset, root, message));

    arc::editor::material_asset loaded;
    REQUIRE(arc::editor::load_material_asset(asset.path, root, loaded, message));
    REQUIRE(loaded.name == "Bronze");
    REQUIRE(loaded.shader == "arc/default_phong");
    REQUIRE(loaded.material.base_color[0] == Catch::Approx(0.8f));
    REQUIRE(loaded.material.metallic == Catch::Approx(0.75f));
    REQUIRE(loaded.material.roughness == Catch::Approx(0.32f));
    REQUIRE(loaded.material.normal_scale == Catch::Approx(0.85f));
    REQUIRE(loaded.textures.base_color == "textures/bronze_base.png");
    REQUIRE(loaded.textures.normal == "textures/bronze_n.png");
    REQUIRE(loaded.graph_reserved);

    const auto resolved = arc::editor::resolve_material_texture_path(root, loaded.textures.base_color);
    REQUIRE(resolved.lexically_normal() == (root / "textures" / "bronze_base.png").lexically_normal());
}

TEST_CASE("editor material assets tolerate missing and future fields")
{
    const auto root = std::filesystem::temp_directory_path() / "arc_editor_material_asset_defaults";
    std::filesystem::create_directories(root);
    const auto path = root / "future.arcmat";

    {
        std::ofstream stream(path, std::ios::binary);
        stream << R"({
  "version": 1,
  "name": "Future",
  "unknownFutureBlock": { "enabled": true },
  "surface": { "metallic": 0.2 },
  "textures": { "emissive": "textures/glow.png" },
  "graph": null
})";
    }

    arc::editor::material_asset loaded;
    std::string message;
    REQUIRE(arc::editor::load_material_asset(path, root, loaded, message));
    REQUIRE(loaded.name == "Future");
    REQUIRE(loaded.shader == "arc/default_phong");
    REQUIRE(loaded.material.metallic == Catch::Approx(0.2f));
    REQUIRE(loaded.material.roughness == Catch::Approx(0.62f));
    REQUIRE(loaded.textures.emissive == "textures/glow.png");
    REQUIRE(arc::editor::is_material_asset_path(path));
    REQUIRE_FALSE(arc::editor::is_material_asset_path(root / "mesh.glb"));
}

TEST_CASE("terrain material version two migrates to current version with fixed layer descriptors")
{
    const auto root = std::filesystem::temp_directory_path() / "arc_editor_terrain_material_tests";
    std::filesystem::create_directories(root / "materials");

    auto asset = arc::editor::make_default_material_asset("Layered Terrain");
    asset.version = 2;
    asset.path = root / "materials" / "layered.arcmat";
    asset.domain = "terrain";
    asset.material.domain = arc::render::material_domain::terrain;
    asset.material.terrain_layers[0].name = "Grass";
    asset.material.terrain_layers[0].tint = {0.65f, 0.8f, 0.5f, 1.0f};
    asset.material.terrain_layers[0].world_scale = 2.75f;
    asset.material.terrain_layers[0].roughness = 0.84f;
    asset.terrain_layers[0].base_color = "textures/terrain/grass/base.jpg";
    asset.terrain_layers[0].normal = "textures/terrain/grass/normal.png";
    asset.terrain_layers[0].roughness = "textures/terrain/grass/roughness.jpg";
    asset.terrain_layers[0].ao = "textures/terrain/grass/ao.jpg";
    asset.terrain_layers[0].height = "textures/terrain/grass/height.png";
    asset.terrain_layers[0].packed_aorh = "textures/terrain/grass/aorh.png";

    std::string message;
    REQUIRE(arc::editor::save_material_asset(asset, root, message));
    arc::editor::material_asset loaded;
    REQUIRE(arc::editor::load_material_asset(asset.path, root, loaded, message));
    REQUIRE(loaded.version == arc::editor::make_default_material_asset().version);
    REQUIRE(loaded.material.domain == arc::render::material_domain::terrain);
    REQUIRE(loaded.material.terrain_layers[0].name == "Grass");
    REQUIRE(loaded.material.terrain_layers[0].world_scale == Catch::Approx(2.75f));
    REQUIRE(loaded.material.terrain_layers[0].roughness == Catch::Approx(0.84f));
    REQUIRE(loaded.material.terrain_layers[0].tint[1] == Catch::Approx(0.8f));
    REQUIRE(loaded.terrain_layers[0].base_color == "textures/terrain/grass/base.jpg");
    REQUIRE(loaded.terrain_layers[0].normal == "textures/terrain/grass/normal.png");
    REQUIRE(loaded.terrain_layers[0].roughness == "textures/terrain/grass/roughness.jpg");
    REQUIRE(loaded.terrain_layers[0].ao == "textures/terrain/grass/ao.jpg");
    REQUIRE(loaded.terrain_layers[0].height == "textures/terrain/grass/height.png");
    REQUIRE(loaded.terrain_layers[0].packed_aorh == "textures/terrain/grass/aorh.png");
}

TEST_CASE("current material version round trips advanced PBR lobes and validates ranges")
{
    const auto root = std::filesystem::temp_directory_path() / "arc_editor_pbr_material_tests";
    std::filesystem::create_directories(root / "materials");
    auto asset = arc::editor::make_default_material_asset("Advanced PBR");
    asset.path = root / "materials" / "advanced.arcmat";
    asset.material.shading_model = arc::render::material_shading_model::transmission;
    asset.material.clear_coat_factor = 0.65f;
    asset.material.clear_coat_roughness = 0.18f;
    asset.material.anisotropy_factor = 0.72f;
    asset.material.anisotropy_rotation = 0.25f;
    asset.material.transmission_factor = 0.8f;
    asset.material.index_of_refraction = 1.46f;
    asset.material.thickness_factor = 0.35f;
    asset.material.attenuation_color = {0.7f, 0.9f, 1.0f};
    asset.material.attenuation_distance = 2.5f;
    asset.material.emissive_luminance_nits = 1200.0f;
    asset.textures.clear_coat = "textures/coat.png";
    asset.textures.anisotropy = "textures/brushed.png";
    asset.textures.transmission = "textures/transmission.png";

    std::string message;
    REQUIRE(arc::editor::save_material_asset(asset, root, message));
    arc::editor::material_asset loaded;
    REQUIRE(arc::editor::load_material_asset(asset.path, root, loaded, message));
    REQUIRE(loaded.version == arc::editor::make_default_material_asset().version);
    REQUIRE(loaded.material.shading_model == arc::render::material_shading_model::transmission);
    REQUIRE(loaded.material.clear_coat_factor == Catch::Approx(0.65f));
    REQUIRE(loaded.material.anisotropy_factor == Catch::Approx(0.72f));
    REQUIRE(loaded.material.transmission_factor == Catch::Approx(0.8f));
    REQUIRE(loaded.material.emissive_luminance_nits == Catch::Approx(1200.0f));
    REQUIRE(loaded.textures.clear_coat == "textures/coat.png");

    std::ifstream saved_input(asset.path, std::ios::binary);
    REQUIRE(saved_input.good());
    std::string invalid_text{std::istreambuf_iterator<char>{saved_input}, std::istreambuf_iterator<char>{}};
    const auto roughness = invalid_text.find("\"roughness\": 0.62");
    REQUIRE(roughness != std::string::npos);
    invalid_text.replace(roughness, std::string("\"roughness\": 0.62").size(), "\"roughness\": 2.0");
    const auto invalid_path = root / "materials" / "invalid.arcmat";
    {
        std::ofstream output(invalid_path, std::ios::binary | std::ios::trunc);
        output << invalid_text;
    }
    REQUIRE_FALSE(arc::editor::load_material_asset(invalid_path, root, loaded, message));
    std::error_code error;
    std::filesystem::remove_all(root, error);
}

TEST_CASE("editor material library applies materials to selected mesh renderer")
{
    arc::ecs::world scene;
    const auto selected = scene.create();
    scene.emplace<arc::scene::mesh_renderer_component>(selected);

    const arc::render::material_handle material{.index = 42, .generation = 7};
    REQUIRE(arc::editor::apply_material_to_selected(scene, selected, material));
    REQUIRE(scene.get<arc::scene::mesh_renderer_component>(selected).material == material);

    const auto empty = scene.create();
    REQUIRE_FALSE(arc::editor::apply_material_to_selected(scene, empty, material));
    REQUIRE_FALSE(arc::editor::apply_material_to_selected(scene, {}, material));
}

TEST_CASE("editor material texture slots accept texture assets and reject wrong types")
{
    const auto root = std::filesystem::temp_directory_path() / "arc_editor_material_slot_tests";
    arc::editor::material_editor_state editor;
    editor.open = true;
    editor.working = arc::editor::make_default_material_asset("Slot Test");

    std::string message;
    REQUIRE(arc::editor::assign_texture_to_material_slot(editor, arc::editor::material_texture_slot::base_color, root,
                                                         root / "textures" / "base.png", &message));
    REQUIRE(editor.dirty);
    REQUIRE(editor.working.textures.base_color == "textures/base.png");

    REQUIRE(arc::editor::assign_texture_to_material_slot(editor, arc::editor::material_texture_slot::normal, root,
                                                         std::filesystem::path{"textures/normal.dds"}, &message));
    REQUIRE(editor.working.textures.normal == "textures/normal.dds");

    REQUIRE_FALSE(arc::editor::assign_texture_to_material_slot(editor, arc::editor::material_texture_slot::ao, root,
                                                               std::filesystem::path{"materials/not_a_texture.arcmat"},
                                                               &message));
}

TEST_CASE("editor material library reuses material handles and saves live updates")
{
    const auto root = std::filesystem::temp_directory_path() / "arc_editor_material_library_reuse";
    std::filesystem::create_directories(root / "materials");
    const auto path = root / "materials" / "reused.arcmat";

    auto asset = arc::editor::make_default_material_asset("Reusable");
    asset.path = path;
    std::string message;
    REQUIRE(arc::editor::save_material_asset(asset, root, message));

    arc::render::renderer renderer;
    arc::editor::editor_material_library library;
    const auto first = arc::editor::load_material_for_editor(library, renderer, root, path);
    const auto second = arc::editor::load_material_for_editor(library, renderer, root, path);
    REQUIRE(first.valid());
    REQUIRE(first == second);
    renderer.frame_queue().commit(1);

    arc::editor::material_editor_state editor;
    REQUIRE(arc::editor::open_material_editor(editor, library, renderer, root, path, message));
    const auto opened = editor.material;
    renderer.frame_queue().commit(2);

    editor.working.material.roughness = 0.21f;
    editor.dirty = true;
    REQUIRE(arc::editor::save_material_editor(editor, library, renderer, root, message));
    REQUIRE(editor.material == opened);
    const auto packet = renderer.frame_queue().commit(3);
    REQUIRE_FALSE(packet.events.empty());
    const auto upload_event =
        std::ranges::find_if(packet.events, [](const auto& event)
                             { return event.type() == arc::render::render_event_type::material_upload; });
    REQUIRE(upload_event != packet.events.end());
    const auto& upload = std::get<arc::render::material_upload_event>(upload_event->payload);
    REQUIRE(upload.handle == opened);
    REQUIRE(upload.material->roughness == Catch::Approx(0.21f));
}

TEST_CASE("editor viewport material drop applies to hit entity and ignores misses")
{
    const auto root = std::filesystem::temp_directory_path() / "arc_editor_viewport_material_drop";
    std::filesystem::create_directories(root / "materials");
    const auto path = root / "materials" / "drop.arcmat";

    auto asset = arc::editor::make_default_material_asset("Drop Material");
    asset.path = path;
    std::string message;
    REQUIRE(arc::editor::save_material_asset(asset, root, message));

    arc::render::renderer renderer;
    arc::editor::editor_material_library library;
    arc::ecs::world scene;
    const auto entity = scene.create();
    scene.emplace<arc::scene::transform_component>(entity);
    scene.emplace<arc::scene::mesh_renderer_component>(entity);
    scene.emplace<arc::scene::bounds_component>(
        entity,
        arc::geometric::box3f{arc::geometric::point3f{-1.0f, -1.0f, -1.0f}, arc::geometric::point3f{1.0f, 1.0f, 1.0f}},
        arc::geometric::box3f{}, true);

    arc::ecs::entity selected{};
    const arc::editor::editor_ray hit_ray{.origin = arc::math::vector3f{0.0f, 0.0f, 5.0f},
                                          .direction = arc::math::vector3f{0.0f, 0.0f, -1.0f}};
    const auto hit = arc::editor::apply_material_asset_to_viewport_hit(
        library, renderer, root, std::filesystem::path{"materials/drop.arcmat"}, scene, hit_ray, selected, &message);
    REQUIRE(hit == entity);
    REQUIRE(selected == entity);
    REQUIRE(scene.get<arc::scene::mesh_renderer_component>(entity).material.valid());

    const auto assigned = scene.get<arc::scene::mesh_renderer_component>(entity).material;
    const arc::editor::editor_ray miss_ray{.origin = arc::math::vector3f{4.0f, 4.0f, 5.0f},
                                           .direction = arc::math::vector3f{0.0f, 0.0f, -1.0f}};
    const auto missed = arc::editor::apply_material_asset_to_viewport_hit(
        library, renderer, root, std::filesystem::path{"materials/drop.arcmat"}, scene, miss_ray, selected, &message);
    REQUIRE_FALSE(missed.valid());
    REQUIRE(selected == entity);
    REQUIRE(scene.get<arc::scene::mesh_renderer_component>(entity).material == assigned);
}
