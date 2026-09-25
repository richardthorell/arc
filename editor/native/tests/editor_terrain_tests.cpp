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

TEST_CASE("terrain host snapshots validate brush settings and group a stroke into one history entry")
{
    auto renderer = std::make_unique<arc::render::renderer>();
    arc::editor::arc_host_manager manager;
    auto host = manager.acquire(std::move(renderer));
    arc::editor::editor_asset_state assets;
    REQUIRE(host->open_project({.name = "Terrain Authoring", .root = {}}, assets).succeeded);
    REQUIRE(
        host->execute(arc::editor::host_create_entity_command{.kind = arc::editor::host_create_entity_kind::terrain})
            .succeeded);
    const auto terrain_entity = host->scene_state().terrain_entity;
    REQUIRE(host->scene_state().terrain_material.valid());
    REQUIRE(host->execute(arc::editor::host_select_entity_command{
                              .entity = {terrain_entity.index, terrain_entity.generation}})
                .succeeded);
    const auto terrain_id = host->selected_entity_snapshot().entity;
    REQUIRE(host->selected_entity_snapshot().terrain.has_value());
    REQUIRE(host->selected_entity_snapshot().terrain->resolution == 257u);
    REQUIRE(host->selected_entity_snapshot().terrain->chunk_quads == 128u);

    REQUIRE(
        host->execute(arc::editor::host_set_terrain_brush_command{.entity = terrain_id,
                                                                  .tool = arc::editor::host_terrain_brush_tool::paint,
                                                                  .radius = 8.0f,
                                                                  .strength = 0.4f,
                                                                  .falloff = 0.75f,
                                                                  .active_layer = 2u})
            .succeeded);
    {
        const auto events = host->poll_events();
        REQUIRE(std::count_if(events.begin(), events.end(), [](const auto& event)
                              { return event.event_type == arc::editor::host_event_type::terrain_tool_changed; }) == 1);
        REQUIRE(std::none_of(events.begin(), events.end(),
                             [](const auto& event)
                             {
                                 return event.event_type == arc::editor::host_event_type::component_changed &&
                                        event.message.find("brush") != std::string::npos;
                             }));
    }
    const auto configured = *host->selected_entity_snapshot().terrain;
    REQUIRE(configured.brush_tool == arc::editor::host_terrain_brush_tool::paint);
    REQUIRE(configured.brush_radius == Catch::Approx(8.0f));
    REQUIRE(configured.active_layer == 2u);
    REQUIRE_FALSE(
        host->execute(arc::editor::host_set_terrain_brush_command{.entity = terrain_id,
                                                                  .radius = std::numeric_limits<float>::infinity(),
                                                                  .strength = 0.4f,
                                                                  .falloff = 0.75f})
            .succeeded);
    REQUIRE(host->selected_entity_snapshot().terrain->brush_radius == Catch::Approx(8.0f));

    REQUIRE(host->execute(arc::editor::host_viewport_set_tool_command{.tool = arc::editor::host_viewport_tool::terrain})
                .succeeded);
    host->request_viewport({.frame_index = 1u, .width = 800u, .height = 600u});
    REQUIRE(
        host->execute(arc::editor::host_terrain_hover_command{.entity = terrain_id, .x = 400u, .y = 300u}).succeeded);
    REQUIRE(host->terrain_tool_snapshot().active);
    REQUIRE(host->terrain_tool_snapshot().hover_visible);
    host->poll_events();
    auto& terrain = host->scene_state().scene.get<arc::scene::terrain_component>(terrain_entity);
    const auto before = terrain.layer_weights;
    const auto begin = host->execute(arc::editor::host_command_envelope{
        .payload = arc::editor::host_terrain_stroke_command{terrain_id, 400u, 300u, arc::editor::host_edit_phase::begin,
                                                            false},
        .edit = arc::editor::host_edit_transaction{901u, arc::editor::host_edit_phase::begin, "Terrain Stroke"}});
    REQUIRE(begin.succeeded);
    REQUIRE(begin.payload_json.find("\"hit\":true") != std::string::npos);
    REQUIRE(host->poll_events().empty());
    REQUIRE(host
                ->execute(arc::editor::host_command_envelope{
                    .payload = arc::editor::host_terrain_stroke_command{terrain_id, 405u, 300u,
                                                                        arc::editor::host_edit_phase::commit, false},
                    .edit = arc::editor::host_edit_transaction{901u, arc::editor::host_edit_phase::commit,
                                                               "Terrain Stroke"}})
                .succeeded);
    {
        const auto events = host->poll_events();
        REQUIRE(std::count_if(events.begin(), events.end(), [](const auto& event)
                              { return event.event_type == arc::editor::host_event_type::terrain_stroke_committed; }) ==
                1);
        REQUIRE(std::none_of(events.begin(), events.end(), [](const auto& event)
                             { return event.event_type == arc::editor::host_event_type::component_changed; }));
    }
    REQUIRE(host->scene_snapshot().undo_label == "Terrain Stroke");
    REQUIRE(terrain.layer_weights != before);
    REQUIRE(host->execute(arc::editor::host_history_undo_command{}).succeeded);
    REQUIRE(host->scene_state().scene.get<arc::scene::terrain_component>(terrain_entity).layer_weights == before);
}

TEST_CASE("M3.5 newly created terrain persists paint into its asset without leaving asset-owned rendering")
{
    const auto root =
        std::filesystem::temp_directory_path() /
        ("arc-editor-terrain-m3-5-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    std::filesystem::create_directories(root / "Content");
    auto renderer = std::make_unique<arc::render::renderer>();
    arc::editor::arc_host_manager manager;
    auto host = manager.acquire(std::move(renderer));
    arc::editor::editor_asset_state assets;
    assets.root = root / "Content";
    REQUIRE(host->open_project({.name = "Terrain M3.5", .root = root}, assets).succeeded);
    REQUIRE(host->execute(arc::editor::host_viewport_create_command{
                              .viewport_id = "viewport-1", .width = 800u, .height = 600u})
                .succeeded);

    const auto created = host->execute(
        arc::editor::host_create_terrain_command{.minimum_elevation = -12.0f, .maximum_elevation = 36.0f});
    REQUIRE(created.succeeded);
    const auto terrain_entity = host->scene_state().terrain_entity;
    auto& terrain = host->scene_state().scene.get<arc::scene::terrain_component>(terrain_entity);
    REQUIRE(terrain.asset.guid.valid());
    REQUIRE(terrain.asset.expected_type == arc::assets::asset_types::terrain);
    const auto terrain_path = root / terrain.asset.path_hint;
    REQUIRE(std::filesystem::is_regular_file(terrain_path));
    REQUIRE(std::filesystem::is_regular_file(root / "Content" / "Terrain" / "Terrain.height.png"));

    const auto guid = host->scene_state().scene.get<arc::scene::persistent_id_component>(terrain_entity).value;
    bool asset_owned{};
    for (std::uint64_t frame = 1u; frame <= 5000u && !asset_owned; ++frame)
    {
        host->request_viewport({.frame_index = frame, .width = 800u, .height = 600u});
        if (const auto* proxy = host->scene_state().terrain_render_proxies.find(guid)) asset_owned = proxy->asset_owned;
        if (!asset_owned) std::this_thread::sleep_for(std::chrono::milliseconds{2});
    }
    REQUIRE(asset_owned);
    const auto rebuild_status = host->execute(arc::editor::host_terrain_modifier_stack_command{
        .entity = {terrain_entity.index, terrain_entity.generation}, .operation = "inspect"});
    REQUIRE(rebuild_status.succeeded);
    CHECK(rebuild_status.payload_json.find("\"rebuild\"") != std::string::npos);
    CHECK(rebuild_status.payload_json.find("\"state\":\"idle\"") != std::string::npos);
    CHECK(terrain.heights.front() == Catch::Approx(-12.0f));
    const auto* initial_proxy = host->scene_state().terrain_render_proxies.find(guid);
    REQUIRE(initial_proxy != nullptr);
    std::vector<arc::render::geometry_resource_handle> geometry;
    for (const auto& region : initial_proxy->regions)
        geometry.push_back(region.geometry);

    const auto terrain_id = arc::editor::host_entity_id{terrain_entity.index, terrain_entity.generation};
    REQUIRE(
        host->execute(arc::editor::host_set_terrain_brush_command{.entity = terrain_id,
                                                                  .tool = arc::editor::host_terrain_brush_tool::paint,
                                                                  .radius = 8.0f,
                                                                  .strength = 0.4f,
                                                                  .falloff = 0.75f,
                                                                  .active_layer = 1u})
            .succeeded);
    REQUIRE(host->execute(arc::editor::host_viewport_set_tool_command{.tool = arc::editor::host_viewport_tool::terrain})
                .succeeded);
    host->request_viewport({.frame_index = 101u, .width = 800u, .height = 600u});
    REQUIRE(
        host->execute(arc::editor::host_terrain_hover_command{.entity = terrain_id, .x = 400u, .y = 300u}).succeeded);
    REQUIRE(host->terrain_tool_snapshot().hover_visible);
    const auto before = terrain.layer_weights;
    REQUIRE(
        host
            ->execute(arc::editor::host_command_envelope{
                .payload = arc::editor::host_terrain_stroke_command{terrain_id, 400u, 300u,
                                                                    arc::editor::host_edit_phase::begin, false},
                .edit = arc::editor::host_edit_transaction{935u, arc::editor::host_edit_phase::begin, "Terrain Paint"}})
            .succeeded);
    REQUIRE(terrain.layer_weights != before);
    host->request_viewport({.frame_index = 102u, .width = 800u, .height = 600u});
    const auto* preview_proxy = host->scene_state().terrain_render_proxies.find(guid);
    REQUIRE(preview_proxy != nullptr);
    CHECK(preview_proxy->asset_owned);
    REQUIRE(preview_proxy->regions.size() == geometry.size());
    for (std::size_t index = 0; index < geometry.size(); ++index)
        CHECK(preview_proxy->regions[index].geometry == geometry[index]);

    REQUIRE(host
                ->execute(arc::editor::host_command_envelope{
                    .payload = arc::editor::host_terrain_stroke_command{terrain_id, 400u, 300u,
                                                                        arc::editor::host_edit_phase::commit, false},
                    .edit = arc::editor::host_edit_transaction{935u, arc::editor::host_edit_phase::commit,
                                                               "Terrain Paint"}})
                .succeeded);
    std::ifstream terrain_document(terrain_path, std::ios::binary);
    const std::string terrain_json((std::istreambuf_iterator<char>(terrain_document)),
                                   std::istreambuf_iterator<char>());
    const auto decoded = arc::scene::read_terrain_asset_json(terrain_json);
    REQUIRE(decoded.has_value());
    const auto paint =
        std::ranges::find_if(decoded.value().modifiers, [](const auto& modifier)
                             { return modifier.type_id == arc::scene::terrain_builtin_modifier_types::paint_layer; });
    REQUIRE(paint != decoded.value().modifiers.end());
    CHECK_FALSE(paint->region_payloads.empty());

    const auto committed_weights = terrain.layer_weights;
    REQUIRE(
        host
            ->execute(arc::editor::host_command_envelope{
                .payload = arc::editor::host_terrain_stroke_command{terrain_id, 420u, 300u,
                                                                    arc::editor::host_edit_phase::begin, false},
                .edit = arc::editor::host_edit_transaction{936u, arc::editor::host_edit_phase::begin, "Terrain Paint"}})
            .succeeded);
    REQUIRE(terrain.layer_weights != committed_weights);
    host->request_viewport({.frame_index = 103u, .width = 800u, .height = 600u});
    REQUIRE(host
                ->execute(arc::editor::host_command_envelope{
                    .payload = arc::editor::host_terrain_stroke_command{terrain_id, 420u, 300u,
                                                                        arc::editor::host_edit_phase::cancel, false},
                    .edit = arc::editor::host_edit_transaction{936u, arc::editor::host_edit_phase::cancel,
                                                               "Terrain Paint"}})
                .succeeded);
    const auto& restored = host->scene_state().scene.get<arc::scene::terrain_component>(terrain_entity);
    CHECK(restored.layer_weights == committed_weights);
    const auto* restored_proxy = host->scene_state().terrain_render_proxies.find(guid);
    REQUIRE(restored_proxy != nullptr);
    CHECK(restored_proxy->asset_owned);
    REQUIRE(restored_proxy->regions.size() == geometry.size());
    for (std::size_t index = 0; index < geometry.size(); ++index)
        CHECK(restored_proxy->regions[index].geometry == geometry[index]);

    host.reset();
    std::error_code cleanup_error;
    std::filesystem::remove_all(root, cleanup_error);
}

TEST_CASE("terrain scene v4 bridge preserves legacy heightfields and terrain asset references")
{
    const auto root = std::filesystem::temp_directory_path() / "arc-terrain-scene-v2-test";
    std::error_code error;
    std::filesystem::remove_all(root, error);
    std::filesystem::create_directories(root, error);
    REQUIRE_FALSE(error);

    auto renderer = std::make_unique<arc::render::renderer>();
    arc::editor::arc_host_manager manager;
    auto host = manager.acquire(std::move(renderer));
    arc::editor::editor_asset_state assets;
    assets.root = root;
    REQUIRE(host->open_project({.name = "Terrain Persistence", .root = root}, assets).succeeded);
    REQUIRE(
        host->execute(arc::editor::host_create_entity_command{.kind = arc::editor::host_create_entity_kind::terrain})
            .succeeded);
    auto& terrain = host->scene_state().scene.get<arc::scene::terrain_component>(host->scene_state().terrain_entity);
    terrain.asset.expected_type = arc::assets::asset_types::terrain;
    terrain.asset.path_hint = "linked.terrain";
    arc::scene::terrain_brush_settings brush;
    brush.tool = arc::scene::terrain_brush_tool::sculpt;
    brush.radius = 9.0f;
    brush.strength = 0.65f;
    arc::scene::apply_terrain_brush(terrain, {4.0f, 0.0f, -7.0f}, brush, 0.5f);
    const auto expected_heights = terrain.heights;
    const auto expected_weights = terrain.layer_weights;
    const auto path = root / "terrain.arcscene";
    REQUIRE(host->execute(arc::editor::host_save_scene_as_command{.path = path}).succeeded);

    std::ifstream saved_stream(path, std::ios::binary);
    const std::string saved((std::istreambuf_iterator<char>(saved_stream)), std::istreambuf_iterator<char>());
    REQUIRE(saved.find("\"Terrain\"") != std::string::npos);
    REQUIRE(saved.find("\"version\": 4") != std::string::npos);
    REQUIRE(saved.find("\"heights\"") != std::string::npos);
    REQUIRE(saved.find("\"weights\"") != std::string::npos);

    terrain.heights.assign(terrain.heights.size(), -100.0f);
    REQUIRE(host->execute(arc::editor::host_open_scene_command{.path = path}).succeeded);
    const auto& loaded =
        host->scene_state().scene.get<arc::scene::terrain_component>(host->scene_state().terrain_entity);
    REQUIRE(loaded.layer_weights == expected_weights);
    REQUIRE(loaded.asset.expected_type == arc::assets::asset_types::terrain);
    REQUIRE(loaded.asset.path_hint == "linked.terrain");
    const auto [minimum, maximum] = std::minmax_element(expected_heights.begin(), expected_heights.end());
    const float tolerance = (*maximum - *minimum) / 65535.0f + 0.0001f;
    for (std::size_t index = 0; index < loaded.heights.size(); index += 997u)
        REQUIRE(loaded.heights[index] == Catch::Approx(expected_heights[index]).margin(tolerance));

    auto corrupt = saved;
    const auto payload = corrupt.find("\"heights\": \"");
    REQUIRE(payload != std::string::npos);
    const auto data_begin = payload + std::string("\"heights\": \"").size();
    const auto data_end = corrupt.find('"', data_begin);
    corrupt.replace(data_begin, data_end - data_begin, "AAAA");
    const auto corrupt_path = root / "corrupt.arcscene";
    {
        std::ofstream stream(corrupt_path, std::ios::binary | std::ios::trunc);
        stream << corrupt;
    }
    const auto revision_before = loaded.content_revision;
    REQUIRE_FALSE(host->execute(arc::editor::host_open_scene_command{.path = corrupt_path}).succeeded);
    REQUIRE(host->scene_state()
                .scene.get<arc::scene::terrain_component>(host->scene_state().terrain_entity)
                .content_revision == revision_before);
    std::filesystem::remove_all(root, error);
}
