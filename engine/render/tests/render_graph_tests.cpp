#include <arc/render/render.h>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <atomic>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <thread>
#include <memory>
#include <vector>

#if !defined(ARC_RENDER_TEST_ASSET_ROOT)
#define ARC_RENDER_TEST_ASSET_ROOT "assets"
#endif

#include "render_test_support.h"

using arc::render::tests::count_recorded_pass;
using arc::render::tests::recording_command_encoder;

TEST_CASE("render graph orders passes by declared resources")
{
    arc::render::render_graph graph;
    const auto backbuffer = graph.add_resource({.name = "backbuffer",
                                                .kind = arc::render::render_resource_kind::color_texture,
                                                .format = arc::render::render_format::rgba8_unorm});
    graph.add_pass({.name = "clear",
                    .kind = arc::render::render_pass_kind::clear,
                    .writes = {{.handle = backbuffer,
                                .kind = arc::render::render_resource_kind::color_texture,
                                .usage = arc::render::render_resource_usage::color_attachment,
                                .write = true}}});
    graph.add_pass({.name = "present",
                    .kind = arc::render::render_pass_kind::present,
                    .reads = {{.handle = backbuffer,
                               .kind = arc::render::render_resource_kind::color_texture,
                               .usage = arc::render::render_resource_usage::sampled}}});

    const auto compiled = graph.compile().value();
    REQUIRE(compiled.passes.size() == 2);
    REQUIRE(compiled.passes[0].name == "clear");
    REQUIRE(compiled.passes[1].name == "present");
}

TEST_CASE("render graph compiles typed resources and transitions")
{
    arc::render::render_graph graph;
    graph.add_resource({.name = "viewport",
                        .kind = arc::render::render_resource_kind::color_texture,
                        .extent = {.width = 1280, .height = 720},
                        .format = arc::render::render_format::rgba8_unorm,
                        .persistent = true});

    graph.add_pass({.name = "viewport clear",
                    .kind = arc::render::render_pass_kind::clear,
                    .writes = {{.resource = "viewport",
                                .kind = arc::render::render_resource_kind::color_texture,
                                .usage = arc::render::render_resource_usage::color_attachment,
                                .write = true,
                                .load_op = arc::render::render_load_op::clear}}});
    graph.add_pass({.name = "imgui sample",
                    .kind = arc::render::render_pass_kind::imgui,
                    .reads = {{.resource = "viewport",
                               .kind = arc::render::render_resource_kind::color_texture,
                               .usage = arc::render::render_resource_usage::sampled}},
                    .side_effect = true});

    const auto compiled = graph.compile().value();
    REQUIRE(compiled.resources.size() == 1);
    REQUIRE(compiled.resources[0].name == "viewport");
    REQUIRE(compiled.resources[0].format == arc::render::render_format::rgba8_unorm);
    REQUIRE(compiled.passes.size() == 2);
    REQUIRE(compiled.passes[0].writes[0].usage == arc::render::render_resource_usage::color_attachment);
    REQUIRE(compiled.transitions.size() == 1);
    REQUIRE(compiled.transitions[0].resource == "viewport");
    REQUIRE(compiled.transitions[0].before == arc::render::render_resource_usage::color_attachment);
    REQUIRE(compiled.transitions[0].after == arc::render::render_resource_usage::sampled);
}

TEST_CASE("compiled render graph executes passes and barriers through a command encoder")
{
    arc::render::render_graph graph;
    const auto target = graph.add_resource({.name = "target",
                                            .kind = arc::render::render_resource_kind::color_texture,
                                            .format = arc::render::render_format::rgba8_unorm});
    std::uint32_t recorded{};
    graph.add_pass({.name = "produce",
                    .writes = {{.handle = target,
                                .kind = arc::render::render_resource_kind::color_texture,
                                .usage = arc::render::render_resource_usage::color_attachment,
                                .write = true}},
                    .record = count_recorded_pass,
                    .payload = arc::render::render_pass_payload::from(&recorded)});
    graph.add_pass({.name = "consume",
                    .reads = {{.handle = target,
                               .kind = arc::render::render_resource_kind::color_texture,
                               .usage = arc::render::render_resource_usage::sampled}},
                    .record = count_recorded_pass,
                    .payload = arc::render::render_pass_payload::from(&recorded),
                    .side_effect = true});

    recording_command_encoder encoder;
    arc::render::execute_render_graph(graph.compile().value(), encoder);

    REQUIRE(encoder.passes == std::vector<std::string>{"produce", "consume"});
    REQUIRE(encoder.barriers == std::vector<std::string>{"target"});
    REQUIRE(encoder.ended_passes == 2);
    REQUIRE(recorded == 2);
}

TEST_CASE("render graph schedules cross-queue waits and rotates persistent history")
{
    using namespace arc::render;
    render_graph graph;
    const auto seed = graph.add_resource({.name = "seed", .kind = render_resource_kind::buffer, .byte_size = 4096});
    const auto history =
        graph.add_resource({.name = "history",
                            .kind = render_resource_kind::color_texture,
                            .extent = {.width = 64, .height = 64},
                            .format = render_format::rgba16_float,
                            .persistent_key = "test.history",
                            .history_length = 2,
                            .history_reset = render_history_reset::camera_cut | render_history_reset::resize});
    graph.add_pass({.name = "graphics seed",
                    .queue = render_queue_type::graphics,
                    .writes = {{.handle = seed,
                                .kind = render_resource_kind::buffer,
                                .usage = render_resource_usage::storage_buffer,
                                .write = true}}});
    graph.add_pass({.name = "temporal compute",
                    .queue = render_queue_type::compute,
                    .reads = {{.handle = seed,
                               .kind = render_resource_kind::buffer,
                               .usage = render_resource_usage::storage_buffer},
                              {.handle = history,
                               .kind = render_resource_kind::color_texture,
                               .usage = render_resource_usage::sampled,
                               .history = render_history_access::previous}},
                    .writes = {{.handle = history,
                                .kind = render_resource_kind::color_texture,
                                .usage = render_resource_usage::storage,
                                .write = true}}});

    const auto compiled = graph.compile().value();
    REQUIRE(compiled.submissions.size() == 2);
    REQUIRE(compiled.submissions[0].queue == render_queue_type::graphics);
    REQUIRE(compiled.submissions[1].queue == render_queue_type::compute);
    REQUIRE(compiled.submissions[1].waits.size() == 1);
    REQUIRE(compiled.submissions[1].waits[0].queue == render_queue_type::graphics);
    REQUIRE(compiled.history_rotations.size() == 1);
    REQUIRE(compiled.history_rotations[0].persistent_key == "test.history");
    REQUIRE(compiled.history_rotations[0].history_length == 2);
    REQUIRE(compiled.lifetimes[history.index].physical_resource != compiled.lifetimes[seed.index].physical_resource);

    recording_command_encoder encoder;
    execute_render_graph(compiled, encoder);
    REQUIRE(encoder.submissions ==
            std::vector<render_queue_type>{render_queue_type::graphics, render_queue_type::compute});
}

TEST_CASE("render graph specializes view extents queues and temporal resets")
{
    using namespace arc::render;
    render_graph graph;
    const auto history = graph.add_resource({.name = "history",
                                             .kind = render_resource_kind::color_texture,
                                             .extent_mode = render_extent_mode::relative_to_view,
                                             .width_scale = 0.5f,
                                             .height_scale = 0.5f,
                                             .format = render_format::rgba16_float,
                                             .mip_levels = 3,
                                             .persistent_key = "view.history",
                                             .history_length = 2,
                                             .history_reset = render_history_reset::camera_cut});
    graph.add_pass({.name = "resolve",
                    .queue = render_queue_type::compute,
                    .writes = {{.handle = history,
                                .kind = render_resource_kind::color_texture,
                                .usage = render_resource_usage::storage,
                                .write = true}},
                    .side_effect = true});

    const auto compiled = graph
                              .compile({.view_id = 42,
                                        .output_extent = {1920, 1080, 1},
                                        .render_extent = {1280, 720, 1},
                                        .frame_index = 7,
                                        .world_epoch = 9,
                                        .temporal_reset = render_history_reset::camera_cut,
                                        .compute_queue_available = false})
                              .value();
    REQUIRE(compiled.view.view_id == 42);
    REQUIRE(compiled.resources[history.index].extent.width == 640);
    REQUIRE(compiled.resources[history.index].extent.height == 360);
    REQUIRE(compiled.passes[0].queue == render_queue_type::graphics);
    REQUIRE(compiled.submissions[0].queue == render_queue_type::graphics);
    REQUIRE(compiled.history_rotations[0].invalidated);
    REQUIRE(compiled.lifetimes[history.index].estimated_bytes ==
            (640ull * 360ull + 320ull * 180ull + 160ull * 90ull) * 8ull);
}

TEST_CASE("render graph culls dead work and aliases nonoverlapping transient resources")
{
    using namespace arc::render;
    render_graph graph;
    const auto dead = graph.add_resource({.name = "dead",
                                          .kind = render_resource_kind::color_texture,
                                          .extent = {64, 64, 1},
                                          .extent_mode = render_extent_mode::absolute,
                                          .format = render_format::rgba8_unorm});
    const auto first = graph.add_resource({.name = "first",
                                           .kind = render_resource_kind::color_texture,
                                           .extent = {64, 64, 1},
                                           .extent_mode = render_extent_mode::absolute,
                                           .format = render_format::rgba8_unorm});
    const auto gate = graph.add_resource({.name = "gate", .kind = render_resource_kind::buffer, .byte_size = 16});
    const auto second = graph.add_resource({.name = "second",
                                            .kind = render_resource_kind::color_texture,
                                            .extent = {64, 64, 1},
                                            .extent_mode = render_extent_mode::absolute,
                                            .format = render_format::rgba8_unorm});
    graph.add_pass({.name = "dead producer",
                    .writes = {{.handle = dead,
                                .kind = render_resource_kind::color_texture,
                                .usage = render_resource_usage::color_attachment,
                                .write = true}}});
    graph.add_pass({.name = "first producer",
                    .writes = {{.handle = first,
                                .kind = render_resource_kind::color_texture,
                                .usage = render_resource_usage::color_attachment,
                                .write = true}}});
    graph.add_pass({.name = "first consumer",
                    .reads = {{.handle = first,
                               .kind = render_resource_kind::color_texture,
                               .usage = render_resource_usage::sampled}},
                    .writes = {{.handle = gate,
                                .kind = render_resource_kind::buffer,
                                .usage = render_resource_usage::storage_buffer,
                                .write = true}}});
    graph.add_pass({.name = "second producer",
                    .reads = {{.handle = gate,
                               .kind = render_resource_kind::buffer,
                               .usage = render_resource_usage::storage_buffer}},
                    .writes = {{.handle = second,
                                .kind = render_resource_kind::color_texture,
                                .usage = render_resource_usage::color_attachment,
                                .write = true}},
                    .side_effect = true});

    const auto compiled = graph.compile().value();
    REQUIRE(compiled.culled_passes.size() == 1);
    REQUIRE(compiled.culled_passes[0].name == "dead producer");
    REQUIRE(compiled.lifetimes[dead.index].physical_resource == render_graph_resource_handle::invalid_index);
    REQUIRE(compiled.lifetimes[first.index].physical_resource == compiled.lifetimes[second.index].physical_resource);
    REQUIRE(compiled.lifetimes[second.index].aliased);
}

TEST_CASE("render graph rejects invalid resource declarations and internal reads")
{
    arc::render::render_graph undeclared;
    undeclared.add_pass(
        {.name = "bad read", .reads = {{.resource = "missing", .usage = arc::render::render_resource_usage::sampled}}});
    REQUIRE_FALSE(undeclared.compile());

    arc::render::render_graph read_before_write;
    const auto transient = read_before_write.add_resource({.name = "transient",
                                                           .kind = arc::render::render_resource_kind::color_texture,
                                                           .format = arc::render::render_format::rgba8_unorm});
    read_before_write.add_pass({.name = "bad read",
                                .reads = {{.handle = transient,
                                           .kind = arc::render::render_resource_kind::color_texture,
                                           .usage = arc::render::render_resource_usage::sampled}}});
    REQUIRE_FALSE(read_before_write.compile());

    arc::render::render_graph incompatible;
    const auto depth = incompatible.add_resource({.name = "depth",
                                                  .kind = arc::render::render_resource_kind::depth_texture,
                                                  .format = arc::render::render_format::d32_float});
    incompatible.add_pass({.name = "bad attachment",
                           .writes = {{.handle = depth,
                                       .kind = arc::render::render_resource_kind::depth_texture,
                                       .usage = arc::render::render_resource_usage::color_attachment,
                                       .write = true}}});
    REQUIRE_FALSE(incompatible.compile());
}

TEST_CASE("clear present graph declares the bring-up passes")
{
    const auto graph = arc::render::make_clear_present_graph("viewport");
    const auto compiled = graph.compile().value();

    REQUIRE(compiled.passes.size() == 2);
    REQUIRE(compiled.resources.size() == 1);
    REQUIRE(compiled.passes[0].kind == arc::render::render_pass_kind::clear);
    REQUIRE(compiled.passes[1].kind == arc::render::render_pass_kind::present);
    REQUIRE_FALSE(compiled.transitions.empty());
}

TEST_CASE("scene draw graph selects only implemented deferred passes")
{
    const auto graph = arc::render::make_scene_draw_graph("viewport", arc::render::render_path::deferred);
    const auto compiled = graph.compile().value();

    REQUIRE(compiled.passes.size() >= 19);
    const auto pass_index = [&](std::string_view name)
    {
        for (std::size_t index = 0; index < compiled.passes.size(); ++index)
        {
            if (compiled.passes[index].name == name) return index;
        }
        return compiled.passes.size();
    };

    const std::size_t static_shadow_index = pass_index("directional static shadows");
    const std::size_t dynamic_shadow_index = pass_index("directional dynamic shadows");
    const std::size_t sky_index = pass_index("sky composite");
    const std::size_t water_spectrum_index = pass_index("Water spectrum update");
    const std::size_t water_ifft_index = pass_index("Water inverse FFT");
    const std::size_t water_foam_index = pass_index("Water foam update");
    const std::size_t depth_index = pass_index("depth prepass");
    const std::size_t gbuffer_index = pass_index("gbuffer pass");
    const std::size_t deferred_index = pass_index("deferred lighting");
    const std::size_t transparent_index = pass_index("forward transparent");
    for (std::size_t index = 0; index < compiled.passes.size(); ++index)
        REQUIRE_FALSE(compiled.passes[index].name.empty());

    REQUIRE(static_shadow_index < gbuffer_index);
    REQUIRE(dynamic_shadow_index < gbuffer_index);
    REQUIRE(water_spectrum_index < water_ifft_index);
    REQUIRE(water_ifft_index < water_foam_index);
    REQUIRE(water_foam_index < depth_index);
    REQUIRE(depth_index < gbuffer_index);
    REQUIRE(gbuffer_index < deferred_index);
    REQUIRE(sky_index < deferred_index);
    REQUIRE(deferred_index < transparent_index);
    REQUIRE(compiled.passes[compiled.passes.size() - 5].builtin == arc::render::builtin_render_pass::debug_overlay);
    REQUIRE(compiled.passes[compiled.passes.size() - 4].builtin ==
            arc::render::builtin_render_pass::luminance_histogram);
    REQUIRE(compiled.passes[compiled.passes.size() - 3].builtin == arc::render::builtin_render_pass::exposure_resolve);
    REQUIRE(compiled.passes[compiled.passes.size() - 2].builtin == arc::render::builtin_render_pass::output_transform);
    REQUIRE(compiled.passes.back().builtin == arc::render::builtin_render_pass::editor_overlay);
    REQUIRE(compiled.resources.size() >= 18);
    REQUIRE(std::any_of(
        compiled.resources.begin(), compiled.resources.end(), [](const auto& resource)
        { return resource.name == "gbuffer_albedo" && resource.format == arc::render::render_format::rgba8_srgb; }));
    REQUIRE(std::any_of(compiled.resources.begin(), compiled.resources.end(),
                        [](const auto& resource) { return resource.name == "water_surface_fields"; }));
    REQUIRE(std::any_of(compiled.resources.begin(), compiled.resources.end(), [](const auto& resource)
                        { return resource.name == "water_foam_history" && resource.history_length == 2u; }));
    REQUIRE(compiled.lifetimes.size() == compiled.resources.size());
    REQUIRE_FALSE(compiled.transitions.empty());
}

TEST_CASE("scene draw graph provides a compact forward plus fallback")
{
    const auto compiled =
        arc::render::make_scene_draw_graph("viewport", arc::render::render_path::forward_plus, false).compile().value();

    REQUIRE(compiled.passes.size() >= 11);
    REQUIRE(compiled.resources.size() >= 8);
    REQUIRE(std::any_of(compiled.passes.begin(), compiled.passes.end(),
                        [](const auto& pass) { return pass.name == "forward opaque"; }));
    for (const auto& pass : compiled.passes)
        REQUIRE(pass.name != "gbuffer pass");
}

TEST_CASE("GPU-driven scene graph declares visibility indirect and temporal history work")
{
    using namespace arc::render;
    resolved_render_config config;
    config.quality = render_quality_tier::ultra;
    config.path = render_path::deferred;
    config.render_scale = 0.75f;
    config.features.gpu_driven_rendering = true;
    config.features.hzb_occlusion = true;
    config.features.temporal_antialiasing = true;
    config.features.temporal_upscaling = true;
    config.features.async_compute = true;
    config.features.virtual_geometry = true;
    config.features.virtual_geometry_path = virtual_geometry_raster_path::compute;
    config.features.submission = gpu_submission_path::indirect_count;

    const auto compiled = make_scene_draw_graph("viewport", config, true).compile().value();
    const auto contains = [&](builtin_render_pass expected)
    {
        return std::any_of(compiled.passes.begin(), compiled.passes.end(),
                           [expected](const auto& pass) { return pass.builtin == expected; });
    };
    REQUIRE(contains(builtin_render_pass::gpu_scene_upload));
    REQUIRE(contains(builtin_render_pass::gpu_frustum_distance_cull));
    REQUIRE(contains(builtin_render_pass::gpu_hzb_occlusion_cull));
    REQUIRE(contains(builtin_render_pass::gpu_lod_selection));
    REQUIRE(contains(builtin_render_pass::gpu_draw_bin_scatter));
    REQUIRE(contains(builtin_render_pass::gpu_indirect_command_generation));
    REQUIRE(contains(builtin_render_pass::gpu_visibility_overflow));
    REQUIRE(contains(builtin_render_pass::virtual_geometry_hierarchy_traversal));
    REQUIRE(contains(builtin_render_pass::virtual_geometry_page_requests));
    REQUIRE(contains(builtin_render_pass::virtual_geometry_cluster_binning));
    REQUIRE(contains(builtin_render_pass::virtual_geometry_software_depth));
    REQUIRE(contains(builtin_render_pass::virtual_geometry_visibility_resolve));
    REQUIRE(contains(builtin_render_pass::virtual_geometry_material_resolve));
    REQUIRE(contains(builtin_render_pass::virtual_geometry_shadow_traversal));
    REQUIRE_FALSE(contains(builtin_render_pass::virtual_geometry_mesh_shader_visibility));
    REQUIRE(contains(builtin_render_pass::depth_pyramid));
    REQUIRE(contains(builtin_render_pass::reactive_mask));
    REQUIRE(contains(builtin_render_pass::disocclusion_mask));
    REQUIRE(contains(builtin_render_pass::temporal_upscale));
    REQUIRE(contains(builtin_render_pass::spatial_sharpen));
    REQUIRE(std::any_of(compiled.submissions.begin(), compiled.submissions.end(),
                        [](const auto& submission) { return submission.queue == render_queue_type::compute; }));
    REQUIRE(std::any_of(compiled.history_rotations.begin(), compiled.history_rotations.end(),
                        [](const auto& history) { return history.persistent_key == "view.temporal_color"; }));
    REQUIRE(std::any_of(compiled.history_rotations.begin(), compiled.history_rotations.end(),
                        [](const auto& history) { return history.persistent_key == "view.depth_hzb"; }));
}

TEST_CASE("Ultra virtual shadow graph declares page feedback cache and lighting dependencies")
{
    using namespace arc::render;
    resolved_render_config config;
    config.quality = render_quality_tier::ultra;
    config.path = render_path::deferred;
    config.features.virtual_shadow_maps = true;
    config.features.screen_space_contact_shadows = true;
    config.screen_space_shadows = true;
    config.screen_space_shadow_scale = 1.0f;

    const auto compiled = make_scene_draw_graph("vsm", config, true).compile().value();
    const auto pass_index = [&](builtin_render_pass expected)
    {
        for (std::size_t index = 0; index < compiled.passes.size(); ++index)
            if (compiled.passes[index].builtin == expected) return index;
        return compiled.passes.size();
    };
    const auto marking = pass_index(builtin_render_pass::virtual_shadow_page_marking);
    const auto allocation = pass_index(builtin_render_pass::virtual_shadow_page_allocation);
    const auto culling = pass_index(builtin_render_pass::virtual_shadow_caster_culling);
    const auto static_render = pass_index(builtin_render_pass::virtual_shadow_static_render);
    const auto dynamic_render = pass_index(builtin_render_pass::virtual_shadow_dynamic_render);
    const auto publication = pass_index(builtin_render_pass::virtual_shadow_page_table_publication);
    const auto feedback = pass_index(builtin_render_pass::virtual_shadow_feedback_readback);
    const auto lighting = pass_index(builtin_render_pass::deferred_lighting);
    REQUIRE(marking < allocation);
    REQUIRE(allocation < culling);
    REQUIRE(culling < static_render);
    REQUIRE(culling < dynamic_render);
    REQUIRE(static_render < publication);
    REQUIRE(dynamic_render < publication);
    REQUIRE(publication < lighting);
    REQUIRE(feedback < compiled.passes.size());
    REQUIRE(pass_index(builtin_render_pass::screen_space_shadow) < lighting);
    REQUIRE(pass_index(builtin_render_pass::screen_space_shadow_filter) < lighting);

    const auto resource = [&](std::string_view name) -> const render_graph_resource*
    {
        const auto found = std::find_if(compiled.resources.begin(), compiled.resources.end(),
                                        [name](const auto& value) { return value.name == name; });
        return found == compiled.resources.end() ? nullptr : &*found;
    };
    const auto* static_pages = resource("virtual_shadow_static_pages");
    const auto* dynamic_pages = resource("virtual_shadow_dynamic_pages");
    const auto* page_table = resource("virtual_shadow_page_table");
    const auto* readback = resource("virtual_shadow_feedback_readback");
    REQUIRE(static_pages != nullptr);
    REQUIRE(dynamic_pages != nullptr);
    REQUIRE(page_table != nullptr);
    REQUIRE(readback != nullptr);
    REQUIRE(static_pages->format == render_format::d16_unorm);
    REQUIRE(static_pages->persistent);
    REQUIRE(dynamic_pages->persistent);
    REQUIRE(page_table->lifetime == render_resource_lifetime_class::per_world);
    REQUIRE(readback->memory == render_memory_class::readback);
    REQUIRE(readback->exported);
}

TEST_CASE("environment lighting graph schedules scalable IBL generation")
{
    arc::render::resolved_render_config config;
    config.quality = arc::render::render_quality_tier::medium;
    config.path = arc::render::render_path::deferred;
    arc::render::world_environment_data environment;
    environment.enabled = true;
    environment.sky_visible = true;
    environment.affect_lighting = true;
    environment.source = arc::render::sky_source_mode::hdri;
    environment.lighting.enabled = true;
    environment.lighting.source = arc::render::environment_lighting_source_mode::follow_sky;

    const auto compiled = arc::render::make_scene_draw_graph("viewport", config, true, environment).compile().value();
    const auto contains = [&](arc::render::builtin_render_pass expected)
    {
        return std::any_of(compiled.passes.begin(), compiled.passes.end(),
                           [expected](const auto& pass) { return pass.builtin == expected; });
    };
    REQUIRE(contains(arc::render::builtin_render_pass::environment_equirect_to_cube));
    REQUIRE(contains(arc::render::builtin_render_pass::environment_irradiance));
    REQUIRE(contains(arc::render::builtin_render_pass::environment_specular_prefilter));
    REQUIRE(contains(arc::render::builtin_render_pass::brdf_integration));
    REQUIRE(contains(arc::render::builtin_render_pass::luminance_histogram));
    REQUIRE(contains(arc::render::builtin_render_pass::exposure_resolve));
    REQUIRE(contains(arc::render::builtin_render_pass::output_transform));
    REQUIRE(std::any_of(compiled.resources.begin(), compiled.resources.end(),
                        [](const auto& resource)
                        {
                            return resource.name == "environment_specular" && resource.extent.width == 256 &&
                                   resource.array_layers == 6 && resource.mip_levels == 9;
                        }));
}

TEST_CASE("world environment graph selects scalable atmosphere and cloud passes")
{
    arc::render::resolved_render_config standard;
    standard.quality = arc::render::render_quality_tier::medium;
    standard.path = arc::render::render_path::deferred;
    arc::render::world_environment_data environment;
    environment.enabled = true;
    environment.sky_visible = true;
    environment.source = arc::render::sky_source_mode::physical_atmosphere;
    environment.atmosphere.enabled = true;
    environment.clouds.enabled = true;
    environment.clouds.cast_shadows = true;

    const auto compiled = arc::render::make_scene_draw_graph("viewport", standard, true, environment).compile().value();
    const auto contains = [&](arc::render::builtin_render_pass expected)
    {
        return std::any_of(compiled.passes.begin(), compiled.passes.end(),
                           [expected](const auto& pass) { return pass.builtin == expected; });
    };
    REQUIRE(contains(arc::render::builtin_render_pass::atmosphere_transmittance));
    REQUIRE(contains(arc::render::builtin_render_pass::atmosphere_multi_scattering));
    REQUIRE(contains(arc::render::builtin_render_pass::atmosphere_sky_view));
    REQUIRE(contains(arc::render::builtin_render_pass::cloud_shadow));
    REQUIRE(contains(arc::render::builtin_render_pass::sky_composite));
    REQUIRE(contains(arc::render::builtin_render_pass::debug_overlay));
    REQUIRE(contains(arc::render::builtin_render_pass::editor_overlay));

    standard.quality = arc::render::render_quality_tier::low;
    standard.path = arc::render::render_path::forward_plus;
    const auto low = arc::render::make_scene_draw_graph("viewport", standard, true, environment).compile().value();
    REQUIRE(std::none_of(low.passes.begin(), low.passes.end(),
                         [](const auto& pass)
                         {
                             return pass.builtin == arc::render::builtin_render_pass::atmosphere_transmittance ||
                                    pass.builtin == arc::render::builtin_render_pass::cloud_shadow;
                         }));
    REQUIRE(std::any_of(low.passes.begin(), low.passes.end(), [](const auto& pass)
                        { return pass.builtin == arc::render::builtin_render_pass::sky_composite; }));
    REQUIRE(std::any_of(low.passes.begin(), low.passes.end(), [](const auto& pass)
                        { return pass.builtin == arc::render::builtin_render_pass::debug_overlay; }));
    REQUIRE(std::any_of(low.passes.begin(), low.passes.end(), [](const auto& pass)
                        { return pass.builtin == arc::render::builtin_render_pass::editor_overlay; }));
}

TEST_CASE("world environment graph selects off solid and HDRI sky paths without atmosphere LUTs")
{
    arc::render::resolved_render_config config;
    config.quality = arc::render::render_quality_tier::medium;
    config.path = arc::render::render_path::deferred;
    const auto contains = [](const auto& graph, arc::render::builtin_render_pass expected)
    {
        return std::any_of(graph.passes.begin(), graph.passes.end(),
                           [expected](const auto& pass) { return pass.builtin == expected; });
    };

    arc::render::world_environment_data environment;
    environment.enabled = false;
    environment.sky_visible = false;
    environment.clouds.enabled = false;
    auto compiled = arc::render::make_scene_draw_graph("viewport", config, true, environment).compile().value();
    REQUIRE_FALSE(contains(compiled, arc::render::builtin_render_pass::sky_composite));
    REQUIRE_FALSE(contains(compiled, arc::render::builtin_render_pass::atmosphere_transmittance));

    environment.enabled = true;
    environment.sky_visible = true;
    environment.source = arc::render::sky_source_mode::solid_color;
    compiled = arc::render::make_scene_draw_graph("viewport", config, true, environment).compile().value();
    REQUIRE(contains(compiled, arc::render::builtin_render_pass::sky_composite));
    REQUIRE_FALSE(contains(compiled, arc::render::builtin_render_pass::atmosphere_transmittance));

    environment.source = arc::render::sky_source_mode::hdri;
    compiled = arc::render::make_scene_draw_graph("viewport", config, true, environment).compile().value();
    REQUIRE(contains(compiled, arc::render::builtin_render_pass::sky_composite));
    REQUIRE_FALSE(contains(compiled, arc::render::builtin_render_pass::atmosphere_transmittance));
    REQUIRE_FALSE(contains(compiled, arc::render::builtin_render_pass::environment_prefilter));
}
