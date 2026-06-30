#include <arc/editor/asset_drag_drop.h>
#include <arc/editor/editor_console.h>
#include <arc/editor/editor_interaction.h>
#include <arc/editor/editor_state.h>
#include <arc/editor/editor_viewport.h>
#include <arc/editor/material_asset.h>
#include <arc/editor/material_library.h>
#include <arc/editor/sdl_events.h>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <cmath>
#include <filesystem>
#include <fstream>

TEST_CASE("editor viewport tracks size and focus state")
{
    arc::editor::editor_viewport viewport;
    REQUIRE_FALSE(viewport.valid());

    viewport.set_size(640.8f, 480.2f);
    viewport.set_focused(true);
    viewport.set_hovered(true);

    REQUIRE(viewport.valid());
    REQUIRE(viewport.width() == 640);
    REQUIRE(viewport.height() == 480);
    REQUIRE(viewport.focused());
    REQUIRE(viewport.hovered());

    viewport.set_screen_rect(100.0f, 50.0f, 320.0f, 240.0f);
    REQUIRE(viewport.width() == 320);
    REQUIRE(viewport.height() == 240);
    REQUIRE(viewport.contains_screen_point(100.0f, 50.0f));
    REQUIRE(viewport.contains_screen_point(419.0f, 289.0f));
    REQUIRE_FALSE(viewport.contains_screen_point(420.0f, 290.0f));
    REQUIRE(viewport.local_x(124.0f) == Catch::Approx(24.0f));
    REQUIRE(viewport.local_y(74.0f) == Catch::Approx(24.0f));
}

TEST_CASE("editor console sink captures bounded log records")
{
    arc::editor::editor_console_sink sink(2);

    sink.write({ .level = arc::log_level::info, .category = "one", .message = "first" });
    sink.write({ .level = arc::log_level::warn, .category = "two", .message = "second" });
    sink.write({ .level = arc::log_level::error, .category = "three", .message = "third" });

    const auto entries = sink.entries();
    REQUIRE(entries.size() == 2);
    REQUIRE(entries[0].category == "two");
    REQUIRE(entries[1].level == arc::log_level::error);
    REQUIRE(entries[1].message == "third");
}

TEST_CASE("SDL events translate to ARC events")
{
    SDL_Event source{};
    arc::event destination{};

    source.type = SDL_EVENT_WINDOW_RESIZED;
    source.window.data1 = 1280;
    source.window.data2 = 720;
    REQUIRE(arc::editor::translate_sdl_event(source, destination));
    REQUIRE(destination.type == arc::event_type::resized);
    REQUIRE(destination.width == 1280);
    REQUIRE(destination.height == 720);

    source = {};
    source.type = SDL_EVENT_MOUSE_BUTTON_DOWN;
    source.button.button = SDL_BUTTON_LEFT;
    source.button.x = 10.0f;
    source.button.y = 20.0f;
    REQUIRE(arc::editor::translate_sdl_event(source, destination));
    REQUIRE(destination.type == arc::event_type::mouse_button_down);
    REQUIRE(destination.button == arc::mouse_button::left);
    REQUIRE(destination.x == 10);
    REQUIRE(destination.y == 20);

    source = {};
    source.type = SDL_EVENT_KEY_DOWN;
    source.key.key = SDLK_A;
    source.key.mod = SDL_KMOD_CTRL;
    source.key.repeat = true;
    REQUIRE(arc::editor::translate_sdl_event(source, destination));
    REQUIRE(destination.type == arc::event_type::key_down);
    REQUIRE(destination.key_code == SDLK_A);
    REQUIRE(destination.modifiers == SDL_KMOD_CTRL);
    REQUIRE(destination.repeat);
}

TEST_CASE("editor selection keeps one selected entity")
{
    arc::scene::registry scene;
    auto first = scene.create();
    auto second = scene.create();
    scene.emplace<arc::scene::selection_component>(first, true);

    arc::scene::entity selected = first;
    REQUIRE(arc::editor::select_entity(scene, second, selected));
    REQUIRE(selected == second);
    REQUIRE_FALSE(scene.get<arc::scene::selection_component>(first).selected);
    REQUIRE(scene.get<arc::scene::selection_component>(second).selected);

    arc::editor::clear_selection(scene, selected);
    REQUIRE_FALSE(selected.valid());
    REQUIRE_FALSE(scene.get<arc::scene::selection_component>(first).selected);
    REQUIRE_FALSE(scene.get<arc::scene::selection_component>(second).selected);
}

TEST_CASE("editor picking hits bounded entities")
{
    arc::scene::registry scene;
    const auto entity = scene.create();
    scene.emplace<arc::scene::transform_component>(entity);
    scene.emplace<arc::scene::bounds_component>(
        entity,
        arc::geometric::box3f{ arc::geometric::point3f{ -1.0f, -1.0f, -1.0f }, arc::geometric::point3f{ 1.0f, 1.0f, 1.0f } },
        arc::geometric::box3f{},
        true);

    const arc::editor::editor_ray ray{
        .origin = arc::math::vector3f{ 0.0f, 0.0f, 5.0f },
        .direction = arc::math::vector3f{ 0.0f, 0.0f, -1.0f }
    };
    REQUIRE(arc::editor::pick_bounded_entity(scene, ray) == entity);

    float distance{};
    REQUIRE(arc::editor::intersect_ray_box(
        ray,
        arc::geometric::box3f{ arc::geometric::point3f{ -1.0f, -1.0f, -1.0f }, arc::geometric::point3f{ 1.0f, 1.0f, 1.0f } },
        distance));
    REQUIRE(distance == Catch::Approx(4.0f));
}

TEST_CASE("editor camera controller orbits pans and zooms")
{
    arc::editor::editor_camera_controller camera;
    arc::scene::transform_component transform;

    camera.focus({ 0.0f, 0.0f, 0.0f }, 2.0f);
    const float focused_distance = camera.distance();
    camera.zoom(1.0f);
    REQUIRE(camera.distance() < focused_distance);

    camera.orbit(24.0f, -12.0f);
    camera.pan(10.0f, 5.0f);
    camera.apply_to(transform);
    REQUIRE(transform.dirty);
    REQUIRE(arc::math::length(transform.position) > 0.01f);
}

TEST_CASE("editor euler conversion keeps pure y rotation stable")
{
    const auto rotation = arc::editor::quaternion_from_euler_degrees({ 0.0f, 135.0f, 0.0f });
    const auto euler = arc::editor::euler_degrees_from_quaternion(rotation);

    REQUIRE(euler[0] == Catch::Approx(0.0f).margin(0.001f));
    REQUIRE(euler[1] == Catch::Approx(135.0f).margin(0.001f));
    REQUIRE(euler[2] == Catch::Approx(0.0f).margin(0.001f));
}

TEST_CASE("editor euler conversion round trips mixed rotation")
{
    const arc::math::vector3f input{ 20.0f, 135.0f, 10.0f };
    const auto rotation = arc::editor::quaternion_from_euler_degrees(input);
    const auto euler = arc::editor::euler_degrees_from_quaternion(rotation);

    REQUIRE(euler[0] == Catch::Approx(input[0]).margin(0.001f));
    REQUIRE(euler[1] == Catch::Approx(input[1]).margin(0.001f));
    REQUIRE(euler[2] == Catch::Approx(input[2]).margin(0.001f));
}

TEST_CASE("editor default sun rotation points downward at an angle")
{
    arc::scene::transform_component transform;
    transform.rotation = arc::editor::quaternion_from_euler_degrees({ -50.0f, -35.0f, 0.0f });

    const auto direction = arc::scene::forward_direction(transform);

    REQUIRE(direction[1] < -0.25f);
    REQUIRE(std::abs(direction[0]) > 0.10f);
    REQUIRE(direction[2] < -0.10f);
}

TEST_CASE("editor tool shortcuts update active tool")
{
    arc::input::input_manager input;
    input.bind_action("tool.select", { .device = arc::input::input_device_type::keyboard, .code = SDLK_Q });
    input.bind_action("tool.translate", { .device = arc::input::input_device_type::keyboard, .code = SDLK_W });
    input.bind_action("tool.rotate", { .device = arc::input::input_device_type::keyboard, .code = SDLK_E });
    input.bind_action("tool.scale", { .device = arc::input::input_device_type::keyboard, .code = SDLK_R });

    auto tool = arc::editor::editor_tool::select;
    input.begin_frame();
    input.process_event({ .type = arc::event_type::key_down, .key_code = SDLK_W });
    arc::editor::apply_tool_shortcuts(input, tool);
    REQUIRE(tool == arc::editor::editor_tool::translate);

    input.begin_frame();
    input.process_event({ .type = arc::event_type::key_down, .key_code = SDLK_R });
    arc::editor::apply_tool_shortcuts(input, tool);
    REQUIRE(tool == arc::editor::editor_tool::scale);
}

TEST_CASE("editor can add a selected primitive mesh entity")
{
    arc::editor::editor_scene_state scene;
    arc::render::renderer renderer;
    arc::render::material_desc imported_material;
    imported_material.name = "Imported Mesh Material";
    imported_material.base_color = { 1.0f, 0.72f, 0.05f, 1.0f };
    scene.default_material = renderer.create_material(imported_material);

    const auto entity = arc::editor::add_primitive_to_scene(scene, renderer, arc::editor::editor_primitive_type::plane);

    REQUIRE(scene.scene.alive(entity));
    REQUIRE(scene.selected_entity == entity);
    REQUIRE(scene.primitive_entities.size() == 1);
    REQUIRE(scene.scene.has<arc::scene::transform_component>(entity));
    REQUIRE(scene.scene.has<arc::scene::bounds_component>(entity));
    REQUIRE(scene.scene.has<arc::scene::mesh_renderer_component>(entity));
    REQUIRE(scene.scene.get<arc::scene::selection_component>(entity).selected);
    REQUIRE(scene.scene.get<arc::scene::mesh_renderer_component>(entity).mesh.valid());
    REQUIRE(scene.scene.get<arc::scene::mesh_renderer_component>(entity).material.valid());
    REQUIRE(scene.primitive_material.valid());
    REQUIRE(scene.primitive_material != scene.default_material);
    REQUIRE(scene.scene.get<arc::scene::mesh_renderer_component>(entity).material == scene.primitive_material);
}

TEST_CASE("editor material assets load save and round trip")
{
    const auto root = std::filesystem::temp_directory_path() / "arc_editor_material_asset_tests";
    std::filesystem::create_directories(root / "materials");

    auto asset = arc::editor::make_default_material_asset("Bronze");
    asset.path = root / "materials" / "bronze.arcmat";
    asset.material.base_color = { 0.8f, 0.42f, 0.18f, 1.0f };
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

TEST_CASE("editor material library applies materials to selected mesh renderer")
{
    arc::scene::registry scene;
    const auto selected = scene.create();
    scene.emplace<arc::scene::mesh_renderer_component>(selected);

    const arc::render::material_handle material{ .index = 42, .generation = 7 };
    REQUIRE(arc::editor::apply_material_to_selected(scene, selected, material));
    REQUIRE(scene.get<arc::scene::mesh_renderer_component>(selected).material == material);

    const auto empty = scene.create();
    REQUIRE_FALSE(arc::editor::apply_material_to_selected(scene, empty, material));
    REQUIRE_FALSE(arc::editor::apply_material_to_selected(scene, {}, material));
}

TEST_CASE("editor asset drag payloads classify and stay asset relative")
{
    const auto root = std::filesystem::temp_directory_path() / "arc_editor_payload_assets";

    REQUIRE(arc::editor::classify_asset_path(root / "materials" / "matte.arcmat") == arc::editor::editor_asset_kind::material);
    REQUIRE(arc::editor::classify_asset_path(root / "textures" / "albedo.png") == arc::editor::editor_asset_kind::texture);
    REQUIRE(arc::editor::classify_asset_path(root / "textures" / "normal.dds") == arc::editor::editor_asset_kind::texture);
    REQUIRE(arc::editor::classify_asset_path(root / "textures" / "studio.hdr") == arc::editor::editor_asset_kind::environment);
    REQUIRE(arc::editor::classify_asset_path(root / "models" / "import.fbx") == arc::editor::editor_asset_kind::scene);
    REQUIRE(arc::editor::classify_asset_path(root / "models" / "level.glb") == arc::editor::editor_asset_kind::scene);
    REQUIRE(arc::editor::is_texture_asset_kind(arc::editor::editor_asset_kind::texture));
    REQUIRE(arc::editor::is_texture_asset_kind(arc::editor::editor_asset_kind::environment));

    arc::editor::editor_asset_payload payload;
    REQUIRE(arc::editor::make_asset_payload(root, root / "materials" / "matte.arcmat", false, payload));
    REQUIRE(payload.kind == arc::editor::editor_asset_kind::material);
    REQUIRE(payload.relative_path.generic_string() == "materials/matte.arcmat");

    REQUIRE(arc::editor::asset_relative_path(root, root / "textures" / "albedo.jpg").generic_string() == "textures/albedo.jpg");
    REQUIRE(arc::editor::is_texture_asset_path(root / "textures" / "normal.dds"));
}

TEST_CASE("editor material texture slots accept texture assets and reject wrong types")
{
    const auto root = std::filesystem::temp_directory_path() / "arc_editor_material_slot_tests";
    arc::editor::material_editor_state editor;
    editor.open = true;
    editor.working = arc::editor::make_default_material_asset("Slot Test");

    std::string message;
    REQUIRE(arc::editor::assign_texture_to_material_slot(
        editor,
        arc::editor::material_texture_slot::base_color,
        root,
        root / "textures" / "base.png",
        &message));
    REQUIRE(editor.dirty);
    REQUIRE(editor.working.textures.base_color == "textures/base.png");

    REQUIRE(arc::editor::assign_texture_to_material_slot(
        editor,
        arc::editor::material_texture_slot::normal,
        root,
        std::filesystem::path{ "textures/normal.dds" },
        &message));
    REQUIRE(editor.working.textures.normal == "textures/normal.dds");

    REQUIRE_FALSE(arc::editor::assign_texture_to_material_slot(
        editor,
        arc::editor::material_texture_slot::ao,
        root,
        std::filesystem::path{ "materials/not_a_texture.arcmat" },
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
    REQUIRE(packet.events.back().type() == arc::render::render_event_type::material_upload);
    const auto& upload = std::get<arc::render::material_upload_event>(packet.events.back().payload);
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
    arc::scene::registry scene;
    const auto entity = scene.create();
    scene.emplace<arc::scene::transform_component>(entity);
    scene.emplace<arc::scene::mesh_renderer_component>(entity);
    scene.emplace<arc::scene::bounds_component>(
        entity,
        arc::geometric::box3f{ arc::geometric::point3f{ -1.0f, -1.0f, -1.0f }, arc::geometric::point3f{ 1.0f, 1.0f, 1.0f } },
        arc::geometric::box3f{},
        true);

    arc::scene::entity selected{};
    const arc::editor::editor_ray hit_ray{
        .origin = arc::math::vector3f{ 0.0f, 0.0f, 5.0f },
        .direction = arc::math::vector3f{ 0.0f, 0.0f, -1.0f }
    };
    const auto hit = arc::editor::apply_material_asset_to_viewport_hit(
        library,
        renderer,
        root,
        std::filesystem::path{ "materials/drop.arcmat" },
        scene,
        hit_ray,
        selected,
        &message);
    REQUIRE(hit == entity);
    REQUIRE(selected == entity);
    REQUIRE(scene.get<arc::scene::mesh_renderer_component>(entity).material.valid());

    const auto assigned = scene.get<arc::scene::mesh_renderer_component>(entity).material;
    const arc::editor::editor_ray miss_ray{
        .origin = arc::math::vector3f{ 4.0f, 4.0f, 5.0f },
        .direction = arc::math::vector3f{ 0.0f, 0.0f, -1.0f }
    };
    const auto missed = arc::editor::apply_material_asset_to_viewport_hit(
        library,
        renderer,
        root,
        std::filesystem::path{ "materials/drop.arcmat" },
        scene,
        miss_ray,
        selected,
        &message);
    REQUIRE_FALSE(missed.valid());
    REQUIRE(selected == entity);
    REQUIRE(scene.get<arc::scene::mesh_renderer_component>(entity).material == assigned);
}
