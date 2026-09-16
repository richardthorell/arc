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

using arc::render::tests::recording_backend;

TEST_CASE("renderer submits committed packets to attached backend")
{
    auto backend = std::make_unique<recording_backend>();
    auto* backend_ptr = backend.get();
    arc::render::renderer renderer;
    renderer.set_backend(std::move(backend));

    arc::render::render_event_buffer buffer;
    arc::render::render_event_writer writer(buffer);
    writer.debug_marker("frame");
    renderer.frame_queue().submit(std::move(buffer));

    const auto result = renderer.render_frame(42, arc::render::make_clear_present_graph("viewport"));

    REQUIRE(result.has_value());
    REQUIRE(backend_ptr->last_frame == 42);
    REQUIRE(backend_ptr->last_event_count == 1);
    REQUIRE(backend_ptr->last_pass_count == 2);
}

TEST_CASE("renderer resolves low quality policy and optional feature overrides")
{
    arc::render::render_capabilities capabilities{};
    capabilities.dedicated_video_memory = 1024ull * 1024ull * 1024ull;
    capabilities.dynamic_rendering = true;
    capabilities.synchronization2 = true;
    capabilities.timeline_semaphores = true;
    capabilities.descriptor_indexing = true;
    capabilities.draw_indirect = true;
    capabilities.draw_indirect_count = true;
    capabilities.sampler_anisotropy = true;
    capabilities.texture_compression_bc = true;

    arc::render::renderer_config config{};
    config.force_disable_optional_features = true;
    const auto resolved = arc::render::resolve_render_config(config, capabilities);

    REQUIRE(resolved.quality == arc::render::render_quality_tier::low);
    REQUIRE(resolved.path == arc::render::render_path::forward_plus);
    REQUIRE(resolved.minimum_render_scale == Catch::Approx(0.5f));
    REQUIRE(resolved.max_point_lights == 32);
    REQUIRE(resolved.directional_shadow_cascades == 2);
    REQUIRE(resolved.directional_shadow_resolution == 1024);
    REQUIRE(resolved.features.draw_indirect);
    REQUIRE(resolved.features.texture_compression_bc);
    REQUIRE_FALSE(resolved.features.dynamic_rendering);
    REQUIRE_FALSE(resolved.features.timeline_semaphores);
    REQUIRE_FALSE(resolved.features.descriptor_indexing);
    REQUIRE_FALSE(resolved.fallback_reasons.empty());
}

TEST_CASE("render quality profiles expose immutable implemented tier policy")
{
    using namespace arc::render;

    STATIC_REQUIRE(default_target_frame_time_ms > 16.0f);
    STATIC_REQUIRE(default_target_frame_time_ms < 17.0f);
    STATIC_REQUIRE(dynamic_resolution_scale_step == 1.0f / 16.0f);
    STATIC_REQUIRE(low_render_quality_profile.default_path == render_path::forward_plus);
    STATIC_REQUIRE(standard_render_quality_profile.default_path == render_path::deferred);

    const auto& low = quality_profile(render_quality_tier::low);
    REQUIRE(low.minimum_render_scale == Catch::Approx(0.5f));
    REQUIRE(low.max_point_lights == 32);
    REQUIRE(low.directional_shadow_cascades == 2);

    const auto& high = quality_profile(render_quality_tier::high);
    REQUIRE(&high == &high_render_quality_profile);
    REQUIRE(high.minimum_render_scale == Catch::Approx(0.67f));
    REQUIRE(high.directional_shadow_resolution == 4096);
    REQUIRE(high.local_shadow_atlas_resolution == 8192);

    const auto& ultra = quality_profile(render_quality_tier::ultra);
    REQUIRE(&ultra == &ultra_render_quality_profile);
    REQUIRE(ultra.max_point_lights == 128);
    REQUIRE(ultra.gi_trace_budget == 4);
    REQUIRE(ultra.virtual_shadow_budget_bytes == 512ull * 1024ull * 1024ull);
    REQUIRE(ultra.virtual_shadow_page_render_budget == 2048);
    REQUIRE(ultra.target_frame_time_ms == Catch::Approx(1000.0f / 30.0f));
}

TEST_CASE("renderer resolves GPU-driven temporal features and their forced fallbacks")
{
    using namespace arc::render;
    render_capabilities capabilities{};
    capabilities.dedicated_video_memory = 16ull * 1024ull * 1024ull * 1024ull;
    capabilities.compute_queue = true;
    capabilities.dedicated_compute_queue = true;
    capabilities.compute_shaders = true;
    capabilities.storage_buffers = true;
    capabilities.storage_images = true;
    capabilities.shader_draw_parameters = true;
    capabilities.gpu_scene_indirect = true;
    capabilities.gpu_scene_indirect_count = true;
    capabilities.hzb_occlusion = true;
    capabilities.temporal_resolve = true;
    capabilities.temporal_upscale = true;
    capabilities.fxaa = true;
    capabilities.descriptor_indexing = true;
    capabilities.virtual_geometry_compute = true;
    capabilities.virtual_geometry_streaming = true;
    capabilities.virtual_shadow_allocation = true;
    capabilities.virtual_shadow_feedback = true;
    capabilities.virtual_shadow_rendering = true;
    capabilities.virtual_shadow_sampling = true;
    capabilities.virtual_shadow_virtual_geometry = true;
    capabilities.screen_space_contact_shadows = true;
    capabilities.screen_space_indirect_lighting = true;
    capabilities.surface_cache = true;
    capabilities.radiance_cache = true;
    capabilities.software_ray_tracing = true;
    capabilities.hardware_ray_query = true;
    capabilities.draw_indirect = true;
    capabilities.draw_indirect_count = true;
    capabilities.sparse_resources = true;
    capabilities.ray_tracing = true;

    renderer_config config{};
    config.quality = render_quality_tier::ultra;
    auto resolved = resolve_render_config(config, capabilities);
    REQUIRE(resolved.quality == render_quality_tier::ultra);
    REQUIRE(resolved.features.gpu_driven_rendering);
    REQUIRE(resolved.features.gpu_binding_model == gpu_resource_binding_model::classic);
    REQUIRE_FALSE(resolved.features.gpu_visibility_compaction);
    REQUIRE(resolved.features.hzb_occlusion);
    REQUIRE(resolved.features.temporal_antialiasing);
    REQUIRE_FALSE(resolved.features.temporal_upscaling);
    REQUIRE(resolved.anti_aliasing == anti_aliasing_method::taa);
    REQUIRE(resolved.features.async_compute);
    REQUIRE_FALSE(resolved.features.virtual_geometry);
    REQUIRE(resolved.features.virtual_geometry_path == virtual_geometry_raster_path::unavailable);
    REQUIRE(resolved.features.software_ray_tracing);
    REQUIRE(resolved.features.virtual_shadow_maps);
    REQUIRE_FALSE(resolved.features.virtual_shadow_virtual_geometry);
    REQUIRE(resolved.features.screen_space_contact_shadows);
    REQUIRE(resolved.virtual_shadow_budget_bytes == 512ull * 1024ull * 1024ull);
    REQUIRE(resolved.features.hardware_ray_tracing);
    REQUIRE(resolved.features.screen_space_gi);
    REQUIRE(resolved.features.screen_space_reflections);
    REQUIRE(resolved.features.surface_cache);
    REQUIRE(resolved.features.radiance_cache);
    REQUIRE(resolved.features.software_gi);
    REQUIRE(resolved.features.software_reflections);
    REQUIRE(resolved.features.hardware_gi);
    REQUIRE(resolved.features.hardware_reflections);
    REQUIRE(resolved.indirect_lighting_path == lighting_trace_path::hybrid_hardware);
    REQUIRE(resolved.lighting_scene_gpu_budget_bytes == 768ull * 1024ull * 1024ull);
    REQUIRE(resolved.features.submission == gpu_submission_path::indirect_count);

    config.quality = render_quality_tier::high;
    const auto high = resolve_render_config(config, capabilities);
    REQUIRE_FALSE(high.features.virtual_geometry);
    REQUIRE(high.features.virtual_geometry_path == virtual_geometry_raster_path::unavailable);
    REQUIRE_FALSE(high.features.virtual_shadow_maps);
    config.quality = render_quality_tier::ultra;

    capabilities.bindless_sampled_images = true;
    capabilities.bindless_samplers = true;
    capabilities.bindless_material_tables = true;
    resolved = resolve_render_config(config, capabilities);
    REQUIRE(resolved.features.gpu_binding_model == gpu_resource_binding_model::classic);
    REQUIRE(resolved.features.virtual_geometry);
    REQUIRE(resolved.features.virtual_geometry_path == virtual_geometry_raster_path::compute);

    capabilities.gpu_visibility_compaction = true;
    capabilities.bindless_geometry_tables = true;
    capabilities.gpu_transparent_sorting = true;
    capabilities.gpu_skinning = true;
    resolved = resolve_render_config(config, capabilities);
    REQUIRE(resolved.features.gpu_binding_model == gpu_resource_binding_model::bindless);
    REQUIRE(resolved.features.gpu_visibility_compaction);
    REQUIRE(resolved.features.gpu_transparent_sorting);
    REQUIRE(resolved.features.gpu_skinning);
    REQUIRE(resolved.features.virtual_geometry);
    REQUIRE(resolved.features.virtual_geometry_path == virtual_geometry_raster_path::compute);
    REQUIRE(resolved.features.virtual_shadow_virtual_geometry);

    config.force_disable_gpu_driven = true;
    config.force_disable_temporal = true;
    config.force_disable_async_compute = true;
    config.force_disable_dynamic_gi = true;
    config.force_disable_hardware_ray_tracing = true;
    config.force_cpu_submission = true;
    resolved = resolve_render_config(config, capabilities);
    REQUIRE_FALSE(resolved.features.gpu_driven_rendering);
    REQUIRE_FALSE(resolved.features.virtual_geometry);
    REQUIRE_FALSE(resolved.features.virtual_shadow_maps);
    REQUIRE_FALSE(resolved.features.temporal_antialiasing);
    REQUIRE_FALSE(resolved.features.async_compute);
    REQUIRE_FALSE(resolved.features.screen_space_gi);
    REQUIRE_FALSE(resolved.features.software_gi);
    REQUIRE_FALSE(resolved.features.hardware_gi);
    REQUIRE(resolved.features.submission == gpu_submission_path::cpu_direct);
}

TEST_CASE("renderer applies resolved configuration when attaching a backend")
{
    auto backend = std::make_unique<recording_backend>();
    backend->capabilities_.dedicated_video_memory = 4ull * 1024ull * 1024ull * 1024ull;
    backend->capabilities_.dynamic_rendering = true;
    auto* backend_ptr = backend.get();

    arc::render::renderer_config config{};
    config.quality = arc::render::render_quality_tier::high;
    arc::render::renderer renderer(config);
    renderer.set_backend(std::move(backend));

    REQUIRE(renderer.resolved_config().quality == arc::render::render_quality_tier::medium);
    REQUIRE(renderer.resolved_config().path == arc::render::render_path::deferred);
    REQUIRE(backend_ptr->configured.quality == arc::render::render_quality_tier::medium);
    REQUIRE_FALSE(backend_ptr->configured.fallback_reasons.empty());
}

TEST_CASE("frame budget controller scales expensive systems before resolution")
{
    arc::render::frame_budget_controller controller;
    controller.reset(arc::render::standard_render_quality_profile, arc::render::default_target_frame_time_ms);

    for (std::uint32_t index = 0; index < 12; ++index)
        controller.update(30.0f);
    const auto reduced = controller.settings();
    REQUIRE(reduced.radiance_probe_update_budget <
            arc::render::standard_render_quality_profile.radiance_probe_update_budget);
    REQUIRE(reduced.volumetric_resolution_scale == Catch::Approx(1.0f));
    REQUIRE(reduced.render_scale == Catch::Approx(1.0f));
    REQUIRE(controller.smoothed_frame_time_ms() > arc::render::default_target_frame_time_ms);

    for (std::uint32_t index = 0; index < 48; ++index)
        controller.update(5.0f);
    REQUIRE(controller.settings().radiance_probe_update_budget >= reduced.radiance_probe_update_budget);
    REQUIRE(controller.settings().render_scale <= 1.0f);
}

TEST_CASE("renderer exposes compiled render graph snapshots through frame profile")
{
    auto backend = std::make_unique<recording_backend>();
    auto* backend_ptr = backend.get();
    arc::render::renderer renderer;
    renderer.set_backend(std::move(backend));

    const auto result = renderer.render_frame(7, arc::render::make_scene_draw_graph("viewport"));

    REQUIRE(result.has_value());
    const auto profile = renderer.last_frame_profile();
    REQUIRE(profile.frame_index == 7);
    REQUIRE(profile.summary == "recorded");
    REQUIRE(profile.graph.passes.size() == backend_ptr->last_pass_count);
    REQUIRE_FALSE(profile.graph.resources.empty());
    REQUIRE(profile.graph.resources[2].name == "scene_color");
    REQUIRE(profile.graph.resources[2].format == arc::render::render_format::rgba16_float);
    REQUIRE(profile.clustered_lights.available);
    REQUIRE(profile.clustered_lights.cluster_count == 96);
    REQUIRE(profile.clustered_lights.overflow_count == 1);
}

TEST_CASE("renderer forwards ObjectID pick requests to backend")
{
    auto backend = std::make_unique<recording_backend>();
    auto* backend_ptr = backend.get();
    arc::render::renderer renderer;
    renderer.set_backend(std::move(backend));

    renderer.request_object_pick(7, 12, 34);

    REQUIRE(backend_ptr->pick_requested);
    REQUIRE(backend_ptr->pick_request.request_id == 7);
    REQUIRE(backend_ptr->pick_request.x == 12);
    REQUIRE(backend_ptr->pick_request.y == 34);
    REQUIRE_FALSE(renderer.last_object_pick().available);
}

TEST_CASE("renderer forwards coherent asynchronous frame capture requests")
{
    auto backend = std::make_unique<recording_backend>();
    auto* backend_ptr = backend.get();
    arc::render::renderer renderer;
    renderer.set_backend(std::move(backend));

    renderer.request_frame_capture({.capture_id = 31,
                                    .channels = {arc::render::render_capture_channel::output_color,
                                                 arc::render::render_capture_channel::object_id}});

    REQUIRE(backend_ptr->capture_requested);
    REQUIRE(backend_ptr->capture_request.capture_id == 31);
    REQUIRE(backend_ptr->capture_request.channels.size() == 2);
    REQUIRE_FALSE(renderer.last_frame_capture().available);
}

TEST_CASE("renderer forwards viewport resize events to backend")
{
    auto backend = std::make_unique<recording_backend>();
    auto* backend_ptr = backend.get();
    arc::render::renderer renderer;
    renderer.set_backend(std::move(backend));

    arc::render::render_event_buffer buffer;
    arc::render::render_event_writer writer(buffer);
    writer.viewport_resize(800, 450);
    renderer.frame_queue().submit(std::move(buffer));

    const auto result = renderer.render_frame(1, arc::render::make_clear_present_graph("viewport"));
    REQUIRE(result.has_value());
    REQUIRE(backend_ptr->viewport_width == 800);
    REQUIRE(backend_ptr->viewport_height == 450);
    REQUIRE(renderer.viewport_texture().valid());
}

TEST_CASE("renderer create mesh enqueues typed upload and tracks handle lifetime")
{
    arc::render::renderer renderer;
    arc::render::mesh_data mesh;
    mesh.name = "triangle";
    mesh.vertices.resize(3);
    mesh.vertices[0].position[0] = -1.0f;
    mesh.vertices[1].position[0] = 1.0f;
    mesh.vertices[2].position[1] = 1.0f;
    mesh.indices = {0, 1, 2};

    const auto handle = renderer.create_mesh(std::move(mesh));
    REQUIRE(renderer.mesh_alive(handle));

    const auto packet = renderer.frame_queue().commit(1);
    REQUIRE(packet.events.size() == 3);
    REQUIRE(packet.events[0].type() == arc::render::render_event_type::mesh_upload);
    const auto& upload = std::get<arc::render::mesh_upload_event>(packet.events[0].payload);
    REQUIRE(upload.handle == handle);
    REQUIRE(upload.mesh->vertices.size() == 3);
    REQUIRE(upload.mesh->indices.size() == 3);
    REQUIRE(packet.events[1].type() == arc::render::render_event_type::gpu_resource_table_update);
    REQUIRE(packet.events[2].type() == arc::render::render_event_type::lighting_geometry_upload);
    const auto& lighting_upload = std::get<arc::render::lighting_geometry_upload_event>(packet.events[2].payload);
    REQUIRE(lighting_upload.geometry->cards.size() == 6);
    REQUIRE(lighting_upload.geometry->distance_field.mode ==
            arc::render::distance_field_mode::two_sided_unsigned_distance);
}

TEST_CASE("renderer updates mesh vertices and retires stale handles")
{
    arc::render::renderer renderer;
    arc::render::mesh_data mesh;
    mesh.name = "dynamic terrain chunk";
    mesh.usage = arc::render::mesh_usage::dynamic_per_frame;
    mesh.vertices.resize(4);
    mesh.indices = {0, 1, 2, 0, 2, 3};
    const auto handle = renderer.create_mesh(std::move(mesh));
    renderer.frame_queue().commit(1);

    std::vector<arc::render::mesh_vertex> vertices(4);
    vertices[0].position[1] = 3.0f;
    REQUIRE(renderer.update_mesh_vertices(handle, vertices));
    auto update = renderer.frame_queue().commit(2);
    REQUIRE(update.events.size() == 2);
    REQUIRE(update.events[0].type() == arc::render::render_event_type::mesh_upload);
    REQUIRE(update.events[1].type() == arc::render::render_event_type::gpu_resource_table_update);
    REQUIRE(std::get<arc::render::mesh_upload_event>(update.events[0].payload).mesh->indices.size() == 6);
    REQUIRE(std::get<arc::render::mesh_upload_event>(update.events[0].payload).mesh->usage ==
            arc::render::mesh_usage::dynamic_per_frame);
    REQUIRE(std::get<arc::render::mesh_upload_event>(update.events[0].payload).mesh->vertices[0].position[1] == 3.0f);

    REQUIRE(renderer.destroy_mesh(handle));
    REQUIRE_FALSE(renderer.mesh_alive(handle));
    auto destroy = renderer.frame_queue().commit(3);
    REQUIRE(destroy.events.size() == 2);
    REQUIRE(destroy.events[0].type() == arc::render::render_event_type::mesh_destroy);
    REQUIRE(destroy.events[1].type() == arc::render::render_event_type::gpu_resource_table_update);
    REQUIRE_FALSE(renderer.destroy_mesh(handle));
}

TEST_CASE("renderer owns generation-safe skin palettes and publishes table updates")
{
    using namespace arc::render;
    renderer renderer;
    skin_palette_data palette;
    palette.name = "hero pose";
    palette.current.resize(2, arc::math::identity<float, 4>());
    palette.current[1](0, 3) = 2.0f;

    const auto handle = renderer.create_skin_palette(std::move(palette));
    REQUIRE(handle.valid());
    REQUIRE(renderer.skin_palette_alive(handle));
    REQUIRE(renderer.skin_palette_data_for(handle) != nullptr);
    REQUIRE(renderer.skin_palette_data_for(handle)->previous.size() == 2);
    auto created = renderer.frame_queue().commit(1);
    REQUIRE(created.events.size() == 2);
    REQUIRE(created.events[0].type() == render_event_type::skin_palette_upload);
    REQUIRE(created.events[1].type() == render_event_type::gpu_resource_table_update);
    const auto& upload = std::get<skin_palette_upload_event>(created.events[0].payload);
    REQUIRE(upload.handle == handle);
    REQUIRE(upload.palette->current[1](0, 3) == Catch::Approx(2.0f));

    skin_palette_data update;
    update.name = "hero pose 2";
    update.current.resize(2, arc::math::identity<float, 4>());
    update.current[0](1, 3) = 3.0f;
    REQUIRE(renderer.update_skin_palette(handle, std::move(update)));
    const auto* retained = renderer.skin_palette_data_for(handle);
    REQUIRE(retained != nullptr);
    REQUIRE(retained->previous[1](0, 3) == Catch::Approx(2.0f));
    auto updated = renderer.frame_queue().commit(2);
    REQUIRE(updated.events.size() == 2);
    REQUIRE(updated.events[0].type() == render_event_type::skin_palette_upload);

    REQUIRE(renderer.destroy_skin_palette(handle));
    REQUIRE_FALSE(renderer.skin_palette_alive(handle));
    auto destroyed = renderer.frame_queue().commit(3);
    REQUIRE(destroyed.events.size() == 2);
    REQUIRE(destroyed.events[0].type() == render_event_type::skin_palette_destroy);
    REQUIRE_FALSE(renderer.update_skin_palette(handle, skin_palette_data{}));

    skin_palette_data replacement;
    replacement.current.resize(1, arc::math::identity<float, 4>());
    const auto recycled = renderer.create_skin_palette(std::move(replacement));
    REQUIRE(recycled.index == handle.index);
    REQUIRE(recycled.generation != handle.generation);
}

TEST_CASE("renderer rejects mismatched optional mesh skin streams")
{
    arc::render::renderer renderer;
    arc::render::mesh_data invalid;
    invalid.vertices.resize(3);
    invalid.skin_vertices.resize(2);
    invalid.indices = {0, 1, 2};
    REQUIRE_FALSE(renderer.create_mesh(std::move(invalid)).valid());
    REQUIRE(renderer.frame_queue().commit(1).events.empty());
}

TEST_CASE("renderer create virtual mesh enqueues typed upload and keeps CPU cluster metadata")
{
    arc::render::renderer renderer;
    arc::render::virtual_mesh_data mesh;
    mesh.vertices.resize(3);
    mesh.indices = {0, 1, 2};
    mesh.clusters.push_back(
        {.first_index = 0, .index_count = 3, .triangle_count = 1, .vertex_count = 3, .material_index = 2});

    const auto handle = renderer.create_virtual_mesh(std::move(mesh));
    REQUIRE(renderer.virtual_mesh_alive(handle));
    REQUIRE(renderer.virtual_mesh_content_generation(handle) == 1);
    REQUIRE(renderer.virtual_mesh_data_for(handle) != nullptr);
    REQUIRE(renderer.virtual_mesh_data_for(handle)->clusters.size() == 1);

    const auto packet = renderer.frame_queue().commit(1);
    REQUIRE(packet.events.size() == 1);
    REQUIRE(packet.events[0].type() == arc::render::render_event_type::virtual_mesh_upload);
    const auto& upload = std::get<arc::render::virtual_mesh_upload_event>(packet.events[0].payload);
    REQUIRE(upload.handle == handle);
    REQUIRE(upload.mesh->vertices.size() == 3);
    REQUIRE(upload.mesh->indices.size() == 3);
    REQUIRE(upload.mesh->clusters.size() == 1);
    REQUIRE(upload.mesh->clusters[0].index_count == 3);

    auto updated = *upload.mesh;
    updated.clusters.push_back(updated.clusters.front());
    REQUIRE(renderer.update_virtual_mesh(handle, std::move(updated)));
    const auto update = renderer.frame_queue().commit(2);
    REQUIRE(update.events.size() == 1);
    REQUIRE(update.events[0].type() == arc::render::render_event_type::virtual_mesh_upload);
    REQUIRE(renderer.virtual_mesh_data_for(handle)->clusters.size() == 2);
    REQUIRE(renderer.virtual_mesh_content_generation(handle) == 2);

    REQUIRE(renderer.destroy_virtual_mesh(handle));
    REQUIRE_FALSE(renderer.virtual_mesh_alive(handle));
    REQUIRE(renderer.virtual_mesh_content_generation(handle) == 0);
    REQUIRE(renderer.virtual_mesh_data_for(handle) == nullptr);
    const auto destroy = renderer.frame_queue().commit(3);
    REQUIRE(destroy.events.size() == 1);
    REQUIRE(destroy.events[0].type() == arc::render::render_event_type::virtual_mesh_destroy);
    REQUIRE(std::get<arc::render::virtual_mesh_destroy_event>(destroy.events[0].payload).handle == handle);
    REQUIRE_FALSE(renderer.destroy_virtual_mesh(handle));
}

TEST_CASE("virtual geometry residency waits for backend page publication acknowledgement")
{
    using namespace arc::render;
    auto backend = std::make_unique<recording_backend>();
    auto* backend_pointer = backend.get();
    renderer renderer;
    renderer.set_backend(std::move(backend));

    virtual_mesh_data mesh;
    mesh.vertices.resize(3);
    mesh.indices = {0, 1, 2};
    mesh.clusters.push_back({.first_index = 0, .index_count = 3, .page_index = 0});
    mesh.pages.push_back({.uncompressed_size = 4, .compressed_size = 4});
    const auto handle = renderer.create_virtual_mesh(std::move(mesh));
    auto bytes = std::make_shared<const std::vector<std::byte>>(4, std::byte{0x2a});
    REQUIRE(renderer.publish_virtual_geometry_page({.resource = handle,
                                                    .resource_generation = 1,
                                                    .page_index = 0,
                                                    .decoded_bytes = bytes,
                                                    .compressed_cpu_bytes = 4}));
    REQUIRE(renderer.virtual_geometry_residency().snapshot().resident_pages == 0);

    backend_pointer->virtual_upload_results.push_back({.resource = handle,
                                                       .resource_generation = 1,
                                                       .page_index = 0,
                                                       .gpu_bytes = 4,
                                                       .compressed_cpu_bytes = 4,
                                                       .succeeded = true});
    REQUIRE(renderer.render_frame(1, render_graph{}));
    REQUIRE(renderer.virtual_geometry_residency().snapshot().resident_pages == 1);
}

TEST_CASE("renderer realizes and retires one unified cooked geometry resource")
{
    arc::render::mesh_data source;
    source.vertices.resize(3);
    source.vertices[1].position[0] = 1.0f;
    source.vertices[2].position[1] = 1.0f;
    source.indices = {0, 1, 2};

    arc::render::renderer renderer;
    const auto geometry = renderer.create_geometry_resource(arc::render::build_virtual_mesh(source), 9);
    REQUIRE(geometry.valid());
    REQUIRE(geometry.asset_generation == 9);
    REQUIRE(geometry.conventional_lod_count == 4);
    REQUIRE(renderer.mesh_alive(geometry.conventional));
    REQUIRE(renderer.virtual_mesh_alive(geometry.virtualized));
    const auto uploads = renderer.frame_queue().commit(1);
    REQUIRE(uploads.events.size() == 13);
    REQUIRE(uploads.events.back().type() == arc::render::render_event_type::virtual_mesh_upload);

    REQUIRE(renderer.destroy_geometry_resource(geometry));
    REQUIRE_FALSE(renderer.mesh_alive(geometry.conventional));
    REQUIRE_FALSE(renderer.virtual_mesh_alive(geometry.virtualized));
    const auto destroys = renderer.frame_queue().commit(2);
    REQUIRE(destroys.events.size() == 13);
    REQUIRE(destroys.events.back().type() == arc::render::render_event_type::virtual_mesh_destroy);
}

TEST_CASE("renderer creates texture and material resources")
{
    arc::render::renderer renderer;
    arc::render::texture_data texture;
    texture.name = "encoded";
    texture.mime_type = "image/png";
    texture.encoded = {std::byte{1}, std::byte{2}};

    const auto texture_handle = renderer.create_texture(std::move(texture));
    REQUIRE(renderer.texture_alive(texture_handle));

    arc::render::material_descriptor material;
    material.name = "pbr";
    material.base_color_texture = texture_handle;
    material.metallic = 0.25f;
    material.roughness = 0.8f;
    material.alpha_mode = arc::render::material_alpha_mode::masked;

    const auto material_handle = renderer.create_material(material);
    REQUIRE(renderer.material_alive(material_handle));

    const auto packet = renderer.frame_queue().commit(1);
    REQUIRE(packet.events.size() == 4);
    REQUIRE(packet.events[0].type() == arc::render::render_event_type::texture_upload);
    REQUIRE(packet.events[1].type() == arc::render::render_event_type::gpu_resource_table_update);
    REQUIRE(packet.events[2].type() == arc::render::render_event_type::material_upload);
    REQUIRE(packet.events[3].type() == arc::render::render_event_type::gpu_resource_table_update);
    const auto& uploaded = std::get<arc::render::material_upload_event>(packet.events[2].payload);
    REQUIRE(uploaded.handle == material_handle);
    REQUIRE(uploaded.material->handle == material_handle);
    REQUIRE(uploaded.material->base_color_texture == texture_handle);
    REQUIRE(uploaded.material->alpha_mode == arc::render::material_alpha_mode::masked);

    auto updated = material;
    updated.name = "pbr_updated";
    updated.roughness = 0.35f;
    updated.base_color = {0.25f, 0.5f, 0.75f, 1.0f};

    REQUIRE(renderer.update_material(material_handle, updated));
    const auto update_packet = renderer.frame_queue().commit(2);
    REQUIRE(update_packet.events.size() == 2);
    REQUIRE(update_packet.events[0].type() == arc::render::render_event_type::material_upload);
    REQUIRE(update_packet.events[1].type() == arc::render::render_event_type::gpu_resource_table_update);
    const auto& material_update = std::get<arc::render::material_upload_event>(update_packet.events[0].payload);
    REQUIRE(material_update.handle == material_handle);
    REQUIRE(material_update.material->handle == material_handle);
    REQUIRE(material_update.material->roughness == Catch::Approx(0.35f));
    REQUIRE(material_update.material->base_color[2] == Catch::Approx(0.75f));

    arc::render::texture_data replacement;
    replacement.name = "environment replacement";
    replacement.width = 2;
    replacement.height = 1;
    replacement.pixels.resize(8);
    REQUIRE(renderer.update_texture(texture_handle, replacement));
    const auto texture_update_packet = renderer.frame_queue().commit(3);
    REQUIRE(texture_update_packet.events.size() == 2);
    REQUIRE(texture_update_packet.events[1].type() == arc::render::render_event_type::gpu_resource_table_update);
    const auto& texture_update = std::get<arc::render::texture_upload_event>(texture_update_packet.events[0].payload);
    REQUIRE(texture_update.handle == texture_handle);
    REQUIRE(texture_update.texture->width == 2);

    REQUIRE_FALSE(renderer.update_material({.index = 999, .generation = 1}, updated));
    REQUIRE_FALSE(renderer.update_texture({.index = 999, .generation = 1}, replacement));
}

TEST_CASE("renderer creates environment resources")
{
    arc::render::renderer renderer;
    arc::render::environment_descriptor environment;
    environment.name = "studio";
    environment.fallback_color = {0.20f, 0.22f, 0.25f};
    environment.intensity = 1.5f;

    const auto handle = renderer.create_environment(environment);
    REQUIRE(handle.valid());
    REQUIRE(renderer.environment_alive(handle));

    const auto packet = renderer.frame_queue().commit(12);
    REQUIRE(packet.events.size() == 1);
    REQUIRE(packet.events[0].type() == arc::render::render_event_type::environment_upload);
    const auto& upload = std::get<arc::render::environment_upload_event>(packet.events[0].payload);
    REQUIRE(upload.handle == handle);
    REQUIRE(upload.environment);
    REQUIRE(upload.environment->handle == handle);
    REQUIRE(upload.environment->intensity == Catch::Approx(1.5f));

    environment.intensity = 0.75f;
    REQUIRE(renderer.update_environment(handle, environment));
    REQUIRE(renderer.destroy_environment(handle));
    REQUIRE_FALSE(renderer.environment_alive(handle));
    const auto lifecycle = renderer.frame_queue().commit(13);
    REQUIRE(lifecycle.events.size() == 2);
    REQUIRE(lifecycle.events[0].type() == arc::render::render_event_type::environment_upload);
    REQUIRE(lifecycle.events[1].type() == arc::render::render_event_type::environment_destroy);
}
