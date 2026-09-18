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

TEST_CASE("arc host protocol serializes command and query envelopes")
{
    const arc::editor::host_entity_id entity{.index = 7, .generation = 3};
    const arc::editor::host_transform transform{
        .position = {1.0f, 2.0f, 3.0f}, .rotation = {0.0f, 0.0f, 0.707f, 0.707f}, .scale = {2.0f, 2.0f, 2.0f}};

    const arc::editor::host_command_envelope commands[]{
        {.request_id = 1, .payload = arc::editor::host_open_project_command{.name = "Protocol", .root = "D:/Protocol"}},
        {.request_id = 2,
         .payload = arc::editor::host_open_scene_command{.path = "D:/Protocol/assets/test.glb", .append = true}},
        {.request_id = 3,
         .payload = arc::editor::host_create_entity_command{.kind = arc::editor::host_create_entity_kind::empty,
                                                            .parent = entity}},
        {.request_id = 4, .payload = arc::editor::host_select_entity_command{.entity = entity}},
        {.request_id = 5, .payload = arc::editor::host_rename_entity_command{.entity = entity, .name = "Renamed"}},
        {.request_id = 6, .payload = arc::editor::host_delete_entity_command{.entity = entity}},
        {.request_id = 7, .payload = arc::editor::host_set_transform_command{.entity = entity, .transform = transform}},
        {.request_id = 8,
         .payload =
             arc::editor::host_viewport_attach_command{
                 .viewport_id = "scene-main", .native_handle = 1234, .x = 16, .y = 32, .width = 1280, .height = 720}},
        {.request_id = 9,
         .payload =
             arc::editor::host_viewport_resize_command{
                 .viewport_id = "scene-main", .x = 24, .y = 48, .width = 640, .height = 360}},
        {.request_id = 28,
         .payload =
             arc::editor::host_viewport_create_command{.viewport_id = "scene-shared",
                                                       .output = arc::editor::host_viewport_output_type::shared_texture,
                                                       .consumer_process_id = 4242,
                                                       .width = 1920,
                                                       .height = 1080}},
        {.request_id = 29,
         .payload =
             arc::editor::host_viewport_frame_released_command{
                 .viewport_id = "scene-shared", .generation = 4, .frame_id = 99, .consumer_handle = "0x1234"}},
        {.request_id = 30,
         .payload = arc::editor::host_viewport_pointer_command{.viewport_id = "scene-shared",
                                                               .phase = arc::editor::host_viewport_pointer_phase::move,
                                                               .x = 640,
                                                               .y = 360,
                                                               .alt = true}},
        {.request_id = 10,
         .payload =
             arc::editor::host_viewport_set_camera_mode_command{.projection =
                                                                    arc::editor::host_camera_projection::orthographic}},
        {.request_id = 11,
         .payload =
             arc::editor::host_viewport_set_render_options_command{
                 .render_mode = arc::editor::host_render_mode::wireframe,
                 .visualization = arc::editor::host_visualization_mode::world_normal,
                 .overlay = arc::editor::host_overlay_mode::all_wireframe,
                 .selection_outline = false,
                 .hover_outline = false,
                 .selection_bounds = true,
                 .component_gizmos = false,
                 .selection_hierarchy = true,
                 .shadows = false,
                 .grid = false,
                 .material_preview_mesh = "cube",
                 .material_preview_auto_rotate = false}},
        {.request_id = 12,
         .payload =
             arc::editor::host_viewport_camera_input_command{
                 .orbit_x = 4.0f, .orbit_y = -2.0f, .look_x = 3.0f, .look_y = -1.0f, .zoom = 1.0f}},
        {.request_id = 13,
         .payload =
             arc::editor::host_set_world_environment_command{
                 .environment = {.entity = entity, .sky_source = arc::editor::host_sky_source::solid_color}}},
        {.request_id = 14,
         .payload =
             arc::editor::host_apply_world_environment_preset_command{
                 .entity = entity, .preset = arc::editor::host_world_environment_preset::night}},
        {.request_id = 15,
         .payload =
             arc::editor::host_set_environment_hdri_command{.entity = entity, .path = "assets/environment/studio.hdr"}},
        {.request_id = 16, .payload = arc::editor::host_set_mesh_renderer_command{.entity = entity, .visible = false}},
        {.request_id = 17,
         .payload = arc::editor::host_set_entity_material_command{.entity = entity, .path = "materials/stone.arcmat"}},
        {.request_id = 18,
         .payload =
             arc::editor::host_create_prefab_command{.entity = entity, .path = "assets/prefabs/stone.arcprefab"}},
        {.request_id = 19,
         .payload =
             arc::editor::host_instantiate_prefab_command{.path = "assets/prefabs/stone.arcprefab", .parent = entity}},
        {.request_id = 20, .payload = arc::editor::host_apply_prefab_command{.entity = entity}},
        {.request_id = 21, .payload = arc::editor::host_revert_prefab_command{.entity = entity}},
        {.request_id = 22, .payload = arc::editor::host_unpack_prefab_command{.entity = entity}},
        {.request_id = 26,
         .payload =
             arc::editor::host_revert_prefab_override_command{.entity = entity,
                                                              .source_entity = "00112233445566778899aabbccddeeff",
                                                              .component_id = "102030405060708090a0b0c0d0e0f000",
                                                              .field_id = 7,
                                                              .kind = "field"}},
        {.request_id = 23,
         .payload =
             arc::editor::host_viewport_set_pose_command{.position = {1.0f, 2.0f, 3.0f}, .target = {0.0f, 0.0f, 0.0f}}},
        {.request_id = 24,
         .payload = arc::editor::host_viewport_capture_command{.capture_id = 99,
                                                               .color = true,
                                                               .depth = true,
                                                               .object_id = true,
                                                               .normals = true,
                                                               .scene_color = true,
                                                               .base_color = true,
                                                               .material_properties = true,
                                                               .emissive = true}},
        {.request_id = 25,
         .payload = arc::editor::host_autosave_scene_command{.path = "Saved/Recovery/protocol.arcscene"}},
        {.request_id = 27,
         .payload = arc::editor::host_viewport_pick_command{.viewport_id = "scene-main", .x = 24, .y = 36}}};

    for (const auto& command : commands)
    {
        const auto json = arc::editor::to_json(command);
        arc::editor::host_command_envelope parsed;
        std::string error;
        REQUIRE(arc::editor::from_json(json, parsed, error));
        REQUIRE(parsed.request_id == command.request_id);
        REQUIRE(parsed.command_type == arc::editor::command_type(command.payload));
        if (command.request_id == 3)
        {
            const auto& create = std::get<arc::editor::host_create_entity_command>(parsed.payload);
            REQUIRE(create.kind == arc::editor::host_create_entity_kind::empty);
            REQUIRE(create.parent == entity);
        }
        if (command.request_id == 12)
        {
            const auto& input = std::get<arc::editor::host_viewport_camera_input_command>(parsed.payload);
            REQUIRE(input.look_x == Catch::Approx(3.0f));
            REQUIRE(input.look_y == Catch::Approx(-1.0f));
        }
        if (command.request_id == 11)
        {
            const auto& options = std::get<arc::editor::host_viewport_set_render_options_command>(parsed.payload);
            REQUIRE_FALSE(options.grid);
            REQUIRE_FALSE(options.selection_outline);
            REQUIRE_FALSE(options.hover_outline);
            REQUIRE(options.selection_bounds);
            REQUIRE_FALSE(options.component_gizmos);
            REQUIRE(options.selection_hierarchy);
            REQUIRE(options.material_preview_mesh == "cube");
            REQUIRE_FALSE(options.material_preview_auto_rotate);
        }
        if (command.request_id == 26)
        {
            const auto& revert = std::get<arc::editor::host_revert_prefab_override_command>(parsed.payload);
            REQUIRE(revert.entity == entity);
            REQUIRE(revert.field_id == 7);
            REQUIRE(revert.kind == "field");
        }
        if (command.request_id == 27)
        {
            const auto& pick = std::get<arc::editor::host_viewport_pick_command>(parsed.payload);
            REQUIRE(pick.viewport_id == "scene-main");
        }
        if (command.request_id == 28)
        {
            const auto& create = std::get<arc::editor::host_viewport_create_command>(parsed.payload);
            REQUIRE(create.viewport_id == "scene-shared");
            REQUIRE(create.output == arc::editor::host_viewport_output_type::shared_texture);
            REQUIRE(create.consumer_process_id == 4242);
        }
        if (command.request_id == 29)
        {
            const auto& release = std::get<arc::editor::host_viewport_frame_released_command>(parsed.payload);
            REQUIRE(release.generation == 4);
            REQUIRE(release.consumer_handle == "0x1234");
        }
        if (command.request_id == 30)
        {
            const auto& pointer = std::get<arc::editor::host_viewport_pointer_command>(parsed.payload);
            REQUIRE(pointer.x == 640);
            REQUIRE(pointer.alt);
        }
    }

    const arc::editor::host_query_envelope queries[]{
        {.request_id = 16, .payload = arc::editor::host_scene_hierarchy_query{}},
        {.request_id = 17, .payload = arc::editor::host_selected_entity_query{}},
        {.request_id = 18, .payload = arc::editor::host_project_assets_query{}},
        {.request_id = 19,
         .payload = arc::editor::host_asset_thumbnail_query{.path = "textures/checker.png", .max_size = 128}},
        {.request_id = 20, .payload = arc::editor::host_viewport_state_query{}},
        {.request_id = 21, .payload = arc::editor::host_world_environment_query{.entity = entity}},
        {.request_id = 22,
         .payload = arc::editor::host_scene_spatial_query{.kind = arc::editor::host_spatial_query_kind::raycast,
                                                          .origin = {0.0f, 2.0f, 4.0f},
                                                          .direction = {0.0f, -0.2f, -1.0f}}},
        {.request_id = 23, .payload = arc::editor::host_viewport_capture_query{.capture_id = 99}},
        {.request_id = 24,
         .payload = arc::editor::host_scene_entities_query{.search = "rock", .offset = 10, .limit = 25}},
        {.request_id = 25,
         .payload = arc::editor::host_entity_by_guid_query{.guid = "00112233445566778899aabbccddeeff"}},
        {.request_id = 26, .payload = arc::editor::host_component_schema_query{}},
        {.request_id = 27,
         .payload = arc::editor::host_scene_spatial_query{.kind = arc::editor::host_spatial_query_kind::frustum}},
        {.request_id = 28, .payload = arc::editor::host_workspace_documents_query{}}};

    for (const auto& query : queries)
    {
        const auto json = arc::editor::to_json(query);
        arc::editor::host_query_envelope parsed;
        std::string error;
        REQUIRE(arc::editor::from_json(json, parsed, error));
        REQUIRE(parsed.request_id == query.request_id);
        REQUIRE(parsed.query_type == arc::editor::query_type(query.payload));
    }
}

TEST_CASE("profiler snapshots serialize scheduler and allocation telemetry")
{
    arc::editor::host_profiler_snapshot snapshot;
    snapshot.timestamp_nanoseconds = 42;
    snapshot.memory_bytes = 1024;
    snapshot.memory_soft_limit = 2048;
    snapshot.memory_hard_limit = 4096;
    snapshot.memory_pressure_events = 1;
    snapshot.jobs_submitted = 3;
    snapshot.jobs_completed = 2;
    snapshot.jobs_stolen = 1;
    snapshot.jobs_queued = 1;
    snapshot.memory_domains.push_back(
        {.domain = "components", .bytes_outstanding = 512, .peak_bytes = 1024, .soft_limit = 2048, .hard_limit = 4096});
    snapshot.allocation_groups.push_back({.domain = "components",
                                          .tag = "world.components",
                                          .world_id = 7,
                                          .thread_id = 9,
                                          .stack_id = 11,
                                          .allocation_count = 4,
                                          .bytes_outstanding = 512});
    snapshot.jobs.push_back({.sequence = 8,
                             .name = "render.frame",
                             .priority = "critical",
                             .affinity = "render",
                             .status = "succeeded",
                             .thread_id = 9,
                             .queued_nanoseconds = 10,
                             .started_nanoseconds = 20,
                             .completed_nanoseconds = 30});

    const auto json = arc::editor::to_json(snapshot);
    REQUIRE(json.find("\"timestampNanoseconds\":42") != std::string::npos);
    REQUIRE(json.find("\"world.components\"") != std::string::npos);
    REQUIRE(json.find("\"render.frame\"") != std::string::npos);
    REQUIRE(std::string(arc::editor::to_string(arc::editor::host_event_type::profiler_snapshot)) ==
            "profiler.snapshot");
}

TEST_CASE("texture asset diagnostics and streaming debug modes serialize through the host protocol")
{
    arc::editor::host_project_assets_snapshot snapshot;
    snapshot.assets.push_back({.guid = "texture-guid",
                               .path = "textures/terrain.png",
                               .kind = "texture",
                               .state = "importing",
                               .has_last_good = true,
                               .width = 8192,
                               .height = 4096,
                               .texture_format = "RGBA8 sRGB",
                               .mip_count = 14,
                               .tile_count = 2730,
                               .streaming_mode = "virtual_tiles",
                               .settings_version = 7,
                               .artifact_size = 64ull * 1024ull * 1024ull});
    const auto json = arc::editor::to_json(snapshot);
    CHECK(json.find("\"hasLastGood\":true") != std::string::npos);
    CHECK(json.find("\"streamingMode\":\"virtual_tiles\"") != std::string::npos);
    CHECK(json.find("\"artifactSize\":67108864") != std::string::npos);
    CHECK(std::string(arc::editor::to_string(arc::editor::host_visualization_mode::texture_desired_mip)) ==
          "textureDesiredMip");
    CHECK(std::string(arc::editor::to_string(arc::editor::host_visualization_mode::virtual_texture_recent_requests)) ==
          "virtualTextureRecentRequests");

    arc::editor::host_command_envelope envelope;
    std::string error;
    REQUIRE(arc::editor::from_json(
        R"({"requestId":9,"type":"viewport.setRenderOptions","payload":{"visualization":"textureResidentMip"}})",
        envelope, error));
    const auto& command = std::get<arc::editor::host_viewport_set_render_options_command>(envelope.payload);
    CHECK(command.visualization == arc::editor::host_visualization_mode::texture_resident_mip);
}

TEST_CASE("arc host resolves material thumbnails from secondary project asset roots")
{
    const auto root = std::filesystem::temp_directory_path() / "arc-editor-secondary-material-thumbnail";
    std::filesystem::remove_all(root);
    const auto primary_root = root / "Content";
    const auto secondary_root = root / "Assets";
    std::filesystem::create_directories(primary_root);
    std::filesystem::create_directories(secondary_root / "materials");

    auto material = arc::editor::make_default_material_asset("Secondary Material");
    material.path = secondary_root / "materials" / "secondary.arcmat";
    std::string message;
    REQUIRE(arc::editor::save_material_asset(material, secondary_root, message));

    auto renderer = std::make_unique<arc::render::renderer>();
    arc::editor::arc_host_manager manager;
    auto host = manager.acquire(std::move(renderer));
    arc::editor::editor_asset_state assets;
    assets.root = primary_root;
    REQUIRE(
        host->open_project(
                {.name = "Secondary Material Thumbnail", .root = root, .content_roots = {primary_root, secondary_root}},
                assets)
            .succeeded);

    const auto catalog = host->project_assets_snapshot();
    const auto found = std::find_if(catalog.assets.begin(), catalog.assets.end(), [](const auto& asset)
                                    { return asset.path == "Assets/materials/secondary.arcmat"; });
    REQUIRE(found != catalog.assets.end());
    REQUIRE(found->kind == "material");
    const auto preview = host->asset_thumbnail(found->path, 64);
    REQUIRE(preview.has_value());
    REQUIRE(preview->width == 64);
    REQUIRE(preview->height == 64);
    REQUIRE(preview->data_url.starts_with("data:image/bmp;base64,Qk"));

    host.reset();
    std::error_code cleanup_error;
    std::filesystem::remove_all(root, cleanup_error);
}

TEST_CASE("arc host catalogs textures and generates safe lazy thumbnails")
{
    const auto root = std::filesystem::temp_directory_path() / "arc-editor-thumbnail-test";
    std::filesystem::remove_all(root);
    std::filesystem::create_directories(root / "textures");
    const auto texture_path = root / "textures" / "preview.tga";
    std::array<unsigned char, 34> tga{};
    tga[2] = 2;
    tga[12] = 2;
    tga[14] = 2;
    tga[16] = 32;
    tga[17] = 0x20;
    const std::array<unsigned char, 16> pixels{0, 0, 255, 255, 0, 255, 0, 255, 255, 0, 0, 255, 255, 255, 255, 255};
    std::copy(pixels.begin(), pixels.end(), tga.begin() + 18);
    {
        std::ofstream output(texture_path, std::ios::binary);
        output.write(reinterpret_cast<const char*>(tga.data()), static_cast<std::streamsize>(tga.size()));
    }

    auto renderer = std::make_unique<arc::render::renderer>();
    arc::editor::arc_host_manager manager;
    auto host = manager.acquire(std::move(renderer));
    arc::editor::editor_asset_state assets;
    assets.root = root;
    REQUIRE(host->open_project({.name = "Thumbnail Test", .root = root}, assets).succeeded);

    const auto catalog = host->project_assets_snapshot();
    REQUIRE(std::any_of(catalog.assets.begin(), catalog.assets.end(), [](const auto& asset)
                        { return asset.path == "textures/preview.tga" && asset.kind == "texture"; }));

    const auto thumbnail = host->asset_thumbnail("textures/preview.tga", 64);
    REQUIRE(thumbnail.has_value());
    REQUIRE(thumbnail->width == 2);
    REQUIRE(thumbnail->height == 2);
    REQUIRE(thumbnail->data_url.starts_with("data:image/bmp;base64,Qk"));
    REQUIRE_FALSE(host->asset_thumbnail("../outside.tga", 64).has_value());

    const auto response = host->query(
        {.request_id = 91,
         .payload = arc::editor::host_asset_thumbnail_query{.path = "textures/preview.tga", .max_size = 64}});
    REQUIRE(response.succeeded);
    REQUIRE(response.payload_json.find("data:image/bmp;base64,") != std::string::npos);

    REQUIRE(host->execute(arc::editor::host_create_entity_command{
                              .kind = arc::editor::host_create_entity_kind::world_environment})
                .succeeded);

    const auto hierarchy = host->scene_snapshot();
    const auto environment = std::find_if(hierarchy.entities.begin(), hierarchy.entities.end(), [](const auto& entity)
                                          { return entity.kind == arc::editor::host_entity_kind::environment; });
    REQUIRE(environment != hierarchy.entities.end());
    REQUIRE(host->execute({.request_id = 92,
                           .payload = arc::editor::host_set_environment_hdri_command{.entity = environment->entity,
                                                                                     .path = "textures/preview.tga"}})
                .succeeded);
    const auto assigned_environment = host->world_environment_snapshot(environment->entity);
    REQUIRE(assigned_environment.has_value());
    REQUIRE(assigned_environment->hdri_path == "textures/preview.tga");
    REQUIRE(assigned_environment->sky_source == arc::editor::host_sky_source::physical_atmosphere);
    REQUIRE_FALSE(
        host->execute({.request_id = 93,
                       .payload = arc::editor::host_set_environment_hdri_command{.entity = environment->entity,
                                                                                 .path = "../outside.tga"}})
            .succeeded);
    REQUIRE(host->execute({.request_id = 94,
                           .payload = arc::editor::host_set_environment_hdri_command{.entity = environment->entity,
                                                                                     .path = {}}})
                .succeeded);
    REQUIRE(host->world_environment_snapshot(environment->entity)->hdri_path.empty());
    host.reset();
    std::filesystem::remove_all(root);
}

TEST_CASE("material preview renderer produces a deterministic PBR sphere")
{
    auto material = arc::editor::make_default_material_asset("Preview Bronze");
    material.material.base_color = {0.72f, 0.24f, 0.07f, 1.0f};
    material.material.metallic = 0.85f;
    material.material.roughness = 0.28f;
    material.material.emissive_factor = {0.01f, 0.0f, 0.0f};
    const auto first = arc::editor::render_material_preview(material, {}, 64);
    const auto second = arc::editor::render_material_preview(material, {}, 64);
    REQUIRE(first.succeeded());
    REQUIRE(first.texture.width == 64);
    REQUIRE(first.texture.height == 64);
    REQUIRE(first.texture.format == arc::render::texture_format::rgba8_srgb);
    REQUIRE(first.texture.pixels == second.texture.pixels);
    const auto center = (32u * 64u + 32u) * 4u;
    REQUIRE(first.texture.pixels[center] != first.texture.pixels[0]);
}

TEST_CASE("asset registry host commands reimport rename and validate GUIDs")
{
    const auto root = std::filesystem::temp_directory_path() / "arc-editor-asset-command-test";
    std::filesystem::remove_all(root);
    std::filesystem::create_directories(root / "materials");
    {
        std::ofstream output(root / "materials" / "source.arcmat", std::ios::binary);
        output << R"({"version":3,"name":"Source"})";
    }

    auto renderer = std::make_unique<arc::render::renderer>();
    arc::editor::arc_host_manager manager;
    auto host = manager.acquire(std::move(renderer));
    arc::editor::editor_asset_state assets;
    assets.root = root;
    REQUIRE(host->open_project({.name = "Asset Commands", .root = root}, assets).succeeded);

    const auto catalog = host->project_assets_snapshot();
    const auto source = std::find_if(catalog.assets.begin(), catalog.assets.end(),
                                     [](const auto& asset) { return asset.path == "materials/source.arcmat"; });
    REQUIRE(source != catalog.assets.end());
    const auto renamed = host->execute(
        {.request_id = 1,
         .payload = arc::editor::host_asset_rename_command{.guid = source->guid, .name = "renamed.arcmat"}});
    REQUIRE(renamed.succeeded);
    REQUIRE(std::filesystem::exists(root / "materials" / "renamed.arcmat"));
    REQUIRE(std::filesystem::exists(root / "materials" / "renamed.arcmat.arcmeta"));
    REQUIRE(host->execute({.request_id = 2, .payload = arc::editor::host_asset_reimport_command{.guid = source->guid}})
                .succeeded);
    REQUIRE_FALSE(host->execute({.request_id = 3,
                                 .payload = arc::editor::host_asset_rename_command{.guid = "not-a-guid",
                                                                                   .name = "invalid.arcmat"}})
                      .succeeded);

    arc::editor::host_command_envelope parsed;
    std::string error;
    REQUIRE(arc::editor::from_json(R"({"requestId":4,"type":"asset.move","payload":{"guid":")" + source->guid +
                                       R"(","path":"materials/moved.arcmat"}})",
                                   parsed, error));
    REQUIRE(std::holds_alternative<arc::editor::host_asset_move_command>(parsed.payload));

    host.reset();
    std::filesystem::remove_all(root);
}

TEST_CASE("mesh renderer host snapshot edits and material assignment round trip")
{
    const auto root = std::filesystem::temp_directory_path() / "arc-editor-mesh-material-host";
    std::filesystem::remove_all(root);
    std::filesystem::create_directories(root / "materials");
    auto material = arc::editor::make_default_material_asset("Inspector Stone");
    material.path = root / "materials" / "inspector_stone.arcmat";
    material.material.base_color = {0.3f, 0.34f, 0.38f, 1.0f};
    std::string message;
    REQUIRE(arc::editor::save_material_asset(material, root, message));

    auto renderer = std::make_unique<arc::render::renderer>();
    arc::editor::arc_host_manager manager;
    auto host = manager.acquire(std::move(renderer));
    arc::editor::editor_asset_state assets;
    assets.root = root;
    REQUIRE(host->open_project({.name = "Mesh Material Host", .root = root}, assets).succeeded);
    REQUIRE(host->execute(
                    {.request_id = 1,
                     .payload =
                         arc::editor::host_create_entity_command{.kind = arc::editor::host_create_entity_kind::sphere}})
                .succeeded);
    const auto entity = host->selected_entity_snapshot().entity;
    const auto initial = host->selected_entity_snapshot();
    REQUIRE(initial.mesh_renderer.has_value());
    REQUIRE(initial.mesh_renderer->representation == 0u);
    REQUIRE(initial.mesh_renderer->visible);
    REQUIRE(initial.mesh_renderer->has_material);

    REQUIRE(host->execute({.request_id = 2,
                           .payload = arc::editor::host_set_mesh_renderer_command{.entity = entity,
                                                                                  .representation = 2u,
                                                                                  .visible = false}})
                .succeeded);
    REQUIRE(host->selected_entity_snapshot().mesh_renderer->representation == 2u);
    REQUIRE_FALSE(host->selected_entity_snapshot().mesh_renderer->visible);
    REQUIRE(host->selected_entity_snapshot().mesh_renderer->visible == false);
    REQUIRE_FALSE(host->execute({.request_id = 31,
                                 .payload = arc::editor::host_set_mesh_renderer_command{.entity = entity,
                                                                                        .representation = 3u,
                                                                                        .visible = true}})
                      .succeeded);
    REQUIRE_FALSE(
        host->execute(
                {.request_id = 3,
                 .payload =
                     arc::editor::host_set_mesh_renderer_command{
                         .entity = entity, .visible = true, .bounds_scale = std::numeric_limits<float>::infinity()}})
            .succeeded);

    REQUIRE(host->execute({.request_id = 4,
                           .payload =
                               arc::editor::host_set_entity_material_command{
                                   .entity = entity, .path = "materials/inspector_stone.arcmat"}})
                .succeeded);
    const auto assigned = host->selected_entity_snapshot();
    REQUIRE(assigned.mesh_renderer->asset_backed_material);
    REQUIRE(assigned.mesh_renderer->material_name == "Inspector Stone");
    REQUIRE(assigned.mesh_renderer->material_path == "materials/inspector_stone.arcmat");
    REQUIRE_FALSE(host->execute({.request_id = 5,
                                 .payload = arc::editor::host_set_entity_material_command{.entity = entity,
                                                                                          .path = "../outside.arcmat"}})
                      .succeeded);

    const auto catalog = host->project_assets_snapshot();
    REQUIRE(std::any_of(catalog.assets.begin(), catalog.assets.end(), [](const auto& asset)
                        { return asset.path == "materials/inspector_stone.arcmat" && asset.kind == "material"; }));
    const auto preview = host->asset_thumbnail("materials/inspector_stone.arcmat", 96);
    REQUIRE(preview.has_value());
    REQUIRE(preview->width == 96);
    REQUIRE(preview->height == 96);
    REQUIRE(preview->data_url.starts_with("data:image/bmp;base64,Qk"));
    host.reset();
    std::filesystem::remove_all(root);
}

TEST_CASE("mesh renderer snapshot preserves secondary content root catalog path")
{
    const auto root = std::filesystem::temp_directory_path() / "arc-editor-secondary-material-roundtrip";
    std::filesystem::remove_all(root);
    const auto primary_root = root / "Content";
    const auto secondary_root = root / "Assets";
    std::filesystem::create_directories(primary_root);
    std::filesystem::create_directories(secondary_root / "materials");

    auto material = arc::editor::make_default_material_asset("Forest Ground");
    material.path = secondary_root / "materials" / "forest_ground.arcmat";
    std::string message;
    REQUIRE(arc::editor::save_material_asset(material, secondary_root, message));

    auto renderer = std::make_unique<arc::render::renderer>();
    arc::editor::arc_host_manager manager;
    auto host = manager.acquire(std::move(renderer));
    arc::editor::editor_asset_state assets;
    assets.root = primary_root;
    REQUIRE(
        host->open_project(
                {.name = "Secondary Material Roundtrip", .root = root, .content_roots = {primary_root, secondary_root}},
                assets)
            .succeeded);
    REQUIRE(host->execute(
                    {.request_id = 1,
                     .payload =
                         arc::editor::host_create_entity_command{.kind = arc::editor::host_create_entity_kind::sphere}})
                .succeeded);
    const auto entity = host->selected_entity_snapshot().entity;

    const auto catalog = host->project_assets_snapshot();
    const auto found = std::find_if(catalog.assets.begin(), catalog.assets.end(), [](const auto& asset)
                                    { return asset.path == "Assets/materials/forest_ground.arcmat"; });
    REQUIRE(found != catalog.assets.end());
    REQUIRE(found->kind == "material");

    REQUIRE(
        host->execute({.request_id = 2,
                       .payload = arc::editor::host_set_entity_material_command{.entity = entity, .path = found->path}})
            .succeeded);
    const auto assigned = host->selected_entity_snapshot();
    REQUIRE(assigned.mesh_renderer.has_value());
    REQUIRE(assigned.mesh_renderer->asset_backed_material);
    REQUIRE(assigned.mesh_renderer->material_name == "Forest Ground");
    REQUIRE(assigned.mesh_renderer->material_path == found->path);
    REQUIRE_FALSE(assigned.mesh_renderer->material_path.starts_with(".."));

    const auto preview = host->asset_thumbnail(assigned.mesh_renderer->material_path, 64);
    REQUIRE(preview.has_value());
    REQUIRE(preview->data_url.starts_with("data:image/bmp;base64,Qk"));

    host.reset();
    std::error_code cleanup_error;
    std::filesystem::remove_all(root, cleanup_error);
}

TEST_CASE("world environment JSON round trips every field and enum")
{
    arc::editor::host_world_environment_snapshot environment;
    environment.entity = {17, 4};
    environment.enabled = false;
    environment.sky_visible = false;
    environment.affect_lighting = false;
    environment.sky_source = arc::editor::host_sky_source::hdri;
    environment.solid_color = {0.11f, 0.22f, 0.33f};
    environment.hdri_path = "environments/night.hdr";
    environment.hdri_rotation_degrees = 37.0f;
    environment.radiance_intensity = 1.25f;
    environment.planet_radius = 6001.0f;
    environment.atmosphere_radius = 6102.0f;
    environment.rayleigh_strength = 1.1f;
    environment.mie_strength = 0.2f;
    environment.ozone_strength = 0.3f;
    environment.atmosphere_tint = {0.44f, 0.55f, 0.66f};
    environment.ground_albedo = {0.12f, 0.13f, 0.14f};
    environment.mie_anisotropy = 0.7f;
    environment.rayleigh_scale_height = 7.0f;
    environment.mie_scale_height = 2.0f;
    environment.multi_scattering_factor = 0.8f;
    environment.exposure = 1.4f;
    environment.sun_disk_size = 0.03f;
    environment.sun_disk_intensity = 2.0f;
    environment.sun_mode = arc::editor::host_sun_position_mode::geographic;
    environment.time_mode = arc::editor::host_celestial_time_mode::system_clock;
    environment.latitude_degrees = -12.5f;
    environment.longitude_degrees = 130.25f;
    environment.north_offset_degrees = 15.0f;
    environment.year = 2032;
    environment.month = 2;
    environment.day = 29;
    environment.local_time_hours = 21.25f;
    environment.utc_offset_hours = -7.0f;
    environment.playing = true;
    environment.loop_day = false;
    environment.time_scale = 120.0f;
    environment.automatic_sun_light = false;
    environment.sun_intensity_multiplier = 0.75f;
    environment.sun_temperature_multiplier = 1.2f;
    environment.moon_enabled = false;
    environment.automatic_moon_phase = false;
    environment.moon_phase = 0.45f;
    environment.moon_intensity = 0.4f;
    environment.moon_angular_radius_degrees = 0.31f;
    environment.stars_enabled = false;
    environment.star_density = 0.5f;
    environment.star_intensity = 1.5f;
    environment.star_twinkle = 0.2f;
    environment.clouds_enabled = false;
    environment.cloud_shadows = false;
    environment.cumulus = {false, 0.1f, 0.2f, 1000.0f, 200.0f, 0.3f, 0.4f, 0.5f, -1.0f, 0.25f, 3.0f, 0.6f, 0.7f};
    environment.cirrus = {true, 0.8f, 0.7f, 7000.0f, 300.0f, 0.6f, 0.5f, 0.4f, 0.5f, -0.5f, 9.0f, 0.3f, 0.2f};
    environment.fog_enabled = false;
    environment.fog_color = {0.15f, 0.25f, 0.35f};
    environment.fog_density = 0.02f;
    environment.fog_height_falloff = 0.3f;
    environment.fog_start_distance = 12.0f;
    environment.fog_max_opacity = 0.6f;
    environment.fog_sun_scattering = 0.4f;
    environment.lighting_enabled = false;
    environment.lighting_source = arc::editor::host_environment_lighting_source::constant_color;
    environment.lighting_color = {0.2f, 0.3f, 0.4f};
    environment.diffuse_intensity = 0.9f;
    environment.specular_intensity = 0.8f;

    const arc::editor::host_command_envelope command{
        .request_id = 42, .payload = arc::editor::host_set_world_environment_command{environment}};
    const auto json = arc::editor::to_json(command);
    arc::editor::host_command_envelope parsed;
    std::string error;
    REQUIRE(arc::editor::from_json(json, parsed, error));
    const auto& round_trip = std::get<arc::editor::host_set_world_environment_command>(parsed.payload);
    REQUIRE(round_trip.environment == environment);
}
