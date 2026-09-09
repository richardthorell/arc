#include "vulkan_backend_internal.h"

namespace arc::render::vulkan::backend_detail
{
render_submit_result vulkan_render_backend::submit(const render_frame_packet& packet,
                                                   const compiled_render_graph& graph)
{
    last_profile_.frame_index = packet.frame_index;
    last_profile_.gpu_scene = {};
    apply_gpu_visibility_statistics(completed_gpu_visibility_statistics_);
    last_profile_.temporal = {};
    temporal_output_view_ = VK_NULL_HANDLE;
    last_profile_.terrain = {};
    last_profile_.terrain.gpu_traversal = resolved_config_.features.gpu_terrain_traversal;
    last_profile_.terrain.gpu_selected_patches = completed_gpu_terrain_statistics_.selected_count;
    last_profile_.terrain.gpu_culled_nodes = completed_gpu_terrain_statistics_.culled_count;
    last_profile_.terrain.gpu_indirect_commands = completed_gpu_terrain_statistics_.draw_count;
    last_profile_.terrain.gpu_overflow_instances = completed_gpu_terrain_statistics_.overflow_count;
    upload_frame_ = packet.frame_index;
    upload_batch_failed_ = false;
    frame_draws_.clear();
    frame_virtual_draws_.clear();
    frame_terrain_draws_.clear();
    frame_gpu_terrain_draws_.clear();
    gpu_terrain_active_instances_.clear();
    frame_shadow_draws_.clear();
    frame_virtual_shadow_draws_.clear();
    frame_directional_lights_.clear();
    frame_point_lights_.clear();
    frame_spot_lights_.clear();
    frame_area_lights_.clear();
    frame_debug_overlay_lines_.clear();
    frame_debug_overlay_triangles_.clear();
    frame_environment_ = {};
    pending_debug_markers_.clear();
    for (const auto& event : packet.events)
    {
        if (const auto* upload = std::get_if<mesh_upload_event>(&event.payload))
        {
            upload_mesh(*upload);
            ++shadow_resource_revision_;
        }
        else if (const auto* destroy = std::get_if<mesh_destroy_event>(&event.payload))
        {
            retire_mesh(destroy->handle);
            ++shadow_resource_revision_;
        }
        else if (const auto* palette = std::get_if<skin_palette_upload_event>(&event.payload))
            upload_skin_palette(*palette);
        else if (const auto* destroyed_palette = std::get_if<skin_palette_destroy_event>(&event.payload))
            retire_skin_palette(destroyed_palette->handle);
        else if (const auto* virtual_upload = std::get_if<virtual_mesh_upload_event>(&event.payload))
        {
            upload_virtual_mesh(*virtual_upload);
            ++shadow_resource_revision_;
        }
        else if (const auto* virtual_destroy = std::get_if<virtual_mesh_destroy_event>(&event.payload))
        {
            retire_virtual_mesh(virtual_destroy->handle);
            ++shadow_resource_revision_;
        }
        else if (const auto* virtual_page = std::get_if<virtual_geometry_page_upload_event>(&event.payload))
            upload_virtual_geometry_page(*virtual_page);
        else if (const auto* terrain = std::get_if<terrain_upload_event>(&event.payload))
        {
            upload_terrain(*terrain);
            ++shadow_resource_revision_;
        }
        else if (const auto* height_update = std::get_if<terrain_height_update_event>(&event.payload))
        {
            update_terrain_heights(*height_update);
            ++shadow_resource_revision_;
        }
        else if (const auto* weight_update = std::get_if<terrain_weight_update_event>(&event.payload))
            update_terrain_weights(*weight_update);
        else if (const auto* terrain_destroy = std::get_if<terrain_destroy_event>(&event.payload))
        {
            retire_terrain(terrain_destroy->handle);
            ++shadow_resource_revision_;
        }
        else if (const auto* texture = std::get_if<texture_upload_event>(&event.payload))
            upload_texture(*texture);
        else if (const auto* streamed_registration = std::get_if<texture_stream_register_event>(&event.payload))
            register_streamed_texture(*streamed_registration);
        else if (const auto* streamed_upload = std::get_if<texture_stream_upload_event>(&event.payload))
            upload_streamed_texture(*streamed_upload);
        else if (const auto* streamed_eviction = std::get_if<texture_stream_evict_event>(&event.payload))
            evict_streamed_texture(*streamed_eviction);
        else if (const auto* destroyed_texture = std::get_if<texture_destroy_event>(&event.payload))
            retire_texture(destroyed_texture->handle);
        else if (const auto* material = std::get_if<material_upload_event>(&event.payload))
        {
            upload_material(*material);
            ++shadow_resource_revision_;
        }
        else if (const auto* environment = std::get_if<environment_upload_event>(&event.payload))
            upload_environment(*environment);
        else if (const auto* destroyed_environment = std::get_if<environment_destroy_event>(&event.payload))
        {
            environments_.erase(resource_key(destroyed_environment->handle));
            if (active_environment_ == destroyed_environment->handle) active_environment_ = {};
        }
        else if (const auto* draw = std::get_if<draw_mesh_event>(&event.payload))
        {
            frame_draws_.push_back(*draw);
            frame_shadow_draws_.push_back(*draw);
        }
        else if (const auto* light = std::get_if<directional_light_event>(&event.payload))
            frame_directional_lights_.push_back(*light);
        else if (const auto* point_light = std::get_if<point_light_event>(&event.payload))
            frame_point_lights_.push_back(*point_light);
        else if (const auto* spot_light = std::get_if<spot_light_event>(&event.payload))
            frame_spot_lights_.push_back(*spot_light);
        else if (const auto* area_light = std::get_if<area_light_event>(&event.payload))
            frame_area_lights_.push_back(*area_light);
        else if (const auto* table_update = std::get_if<gpu_resource_table_update_event>(&event.payload))
            apply_gpu_resource_table_update(*table_update);
        else if (const auto* gpu_scene_update = std::get_if<gpu_scene_update_event>(&event.payload))
            apply_gpu_scene_update(*gpu_scene_update);
        else if (const auto* world = std::get_if<render_world_event>(&event.payload))
            append_render_world(*world);
        else if (const auto* marker = std::get_if<debug_marker_event>(&event.payload))
            pending_debug_markers_.push_back(marker->label);
    }
    if (virtual_geometry_tables_dirty_ && !rebuild_virtual_geometry_tables()) upload_batch_failed_ = true;
    if (!flush_gpu_resource_tables()) upload_batch_failed_ = true;
    if (!flush_upload_batch()) upload_batch_failed_ = true;
    if (upload_batch_failed_)
    {
        for (auto& result : frame_texture_upload_results_)
            result.succeeded = false;
        for (auto& result : frame_virtual_geometry_upload_results_)
            result.succeeded = false;
    }
    completed_texture_upload_results_.insert(completed_texture_upload_results_.end(),
                                             frame_texture_upload_results_.begin(),
                                             frame_texture_upload_results_.end());
    frame_texture_upload_results_.clear();
    completed_virtual_geometry_upload_results_.insert(completed_virtual_geometry_upload_results_.end(),
                                                      frame_virtual_geometry_upload_results_.begin(),
                                                      frame_virtual_geometry_upload_results_.end());
    frame_virtual_geometry_upload_results_.clear();

    last_profile_.graph = graph;
    last_profile_.summary.clear();
    last_profile_.summary.reserve(64);
    last_profile_.summary += std::to_string(graph.passes.size());
    last_profile_.summary += " graph pass(es), ";
    last_profile_.summary += std::to_string(packet.events.size());
    last_profile_.summary += " render event(s)";

    const environment_descriptor* lighting_environment = active_environment();
    if (frame_environment_.lighting.environment.valid())
    {
        const auto found = environments_.find(resource_key(frame_environment_.lighting.environment));
        if (found != environments_.end()) lighting_environment = &found->second.data;
    }
    auto point_lights_for_tier = frame_point_lights_;
    const bool low_area_fallback = resolved_config_.quality == render_quality_tier::low;
    if (low_area_fallback)
    {
        for (const auto& area : frame_area_lights_)
        {
            const float width = std::max(area.width, 0.001f);
            const float height = area.shape == area_light_shape::disk ? width : std::max(area.height, 0.001f);
            const float surface_area =
                area.shape == area_light_shape::disk ? math::pi<float> * 0.25f * width * width : width * height;
            const float lumens = area.intensity_unit == light_intensity_unit::nit
                                     ? area.intensity * math::pi<float> * surface_area * (area.two_sided ? 2.0f : 1.0f)
                                     : area.intensity;
            point_lights_for_tier.push_back({.position = area.position,
                                             .color = area.color,
                                             .intensity = lumens,
                                             .range = std::clamp(std::sqrt(std::max(lumens, 0.0f)) * 0.5f, 5.0f, 50.0f),
                                             .enabled = area.enabled,
                                             .use_color_temperature = area.use_color_temperature,
                                             .temperature_kelvin = area.temperature_kelvin,
                                             .intensity_unit = area.intensity_unit == light_intensity_unit::unitless
                                                                   ? light_intensity_unit::unitless
                                                                   : light_intensity_unit::lumen,
                                             .label = area.label + " (low-tier point fallback)"});
        }
    }
    frame_lighting_ = pack_scene_lighting(
        frame_directional_lights_, point_lights_for_tier, frame_spot_lights_,
        frame_environment_.affect_lighting && frame_environment_.lighting.enabled ? lighting_environment : nullptr,
        resolved_config_.max_point_lights, resolved_config_.max_spot_lights,
        low_area_fallback ? empty_area_lights_ : frame_area_lights_);
    if (frame_environment_.affect_lighting && frame_environment_.lighting.enabled)
    {
        math::vector3f ambient = frame_environment_.lighting.constant_color;
        if (frame_environment_.lighting.source == environment_lighting_source_mode::follow_sky)
        {
            ambient = frame_environment_.source == sky_source_mode::solid_color
                          ? frame_environment_.solid_color
                          : math::vector3f{frame_environment_.atmosphere.tint[0] * 0.28f,
                                           frame_environment_.atmosphere.tint[1] * 0.28f,
                                           frame_environment_.atmosphere.tint[2] * 0.28f};
        }
        if (frame_environment_.lighting.source != environment_lighting_source_mode::hdri || !lighting_environment)
        {
            frame_lighting_.ambient_color_intensity = {ambient[0], ambient[1], ambient[2],
                                                       frame_environment_.lighting.diffuse_intensity};
        }
    }
    update_environment_profile(lighting_environment);
    last_profile_.clustered_lights = make_clustered_light_profile();
    update_shadow_profile(packet.frame_index);
    last_profile_.temporal = {.enabled = resolved_config_.features.temporal_antialiasing,
                              .upscaling = resolved_config_.features.temporal_upscaling,
                              .history_valid = frame_camera_.history_valid,
                              .camera_cut = frame_camera_.camera_cut,
                              .jitter = frame_camera_.jitter,
                              .reset_reason = frame_camera_.camera_cut ? "camera cut, resize, or world change" : ""};
    update_light_buffer();
    warn_about_skipped_lights(frame_lighting_);

    std::ostringstream message;
    message << "vulkan accepted frame " << packet.frame_index << " with " << packet.events.size() << " event(s) and "
            << graph.passes.size() << " pass(es)";
    if (upload_batch_failed_)
    {
        message << "; one or more resource upload batches failed";
        return render_submit_result::failure({render_submit_error_code::backend_failure, message.str()});
    }
    return render_submit_result::success();
}

void vulkan_render_backend::append_render_world(const render_world_event& event)
{
    if (!event.packet) return;

    const auto& packet = *event.packet;
    const auto make_draw = [&](const render_item& item, bool selected_for_overlay)
    {
        return draw_mesh_event{.gpu_scene_instance = item.gpu_scene_instance,
                               .mesh = item.mesh,
                               .material = item.material,
                               .model = item.model,
                               .previous_model = item.previous_model,
                               .view_projection = packet.camera.view_projection,
                               .previous_view_projection = packet.camera.previous_view_projection,
                               .world_bounds = item.world_bounds,
                               .mode = packet.mode,
                               .visualization = packet.visualization,
                               .object_id = item.object_id,
                               .skin_palette = item.skin_matrices,
                               .skin_joint_count = item.skin_joint_count,
                               .selected = selected_for_overlay,
                               .casts_shadows = item.casts_shadows,
                               .receives_shadows = item.receives_shadows,
                               .mobility = item.mobility,
                               .shadow_lod_bias = item.shadow_lod_bias,
                               .maximum_shadow_distance = item.maximum_shadow_distance,
                               .base_color_tint = item.base_color_tint,
                               .wire_color = math::vector4f{1.0f, 0.48f, 0.04f, 1.0f},
                               .label = item.label};
    };
    const auto make_virtual_draw = [&](const virtual_render_item& item, bool selected_for_overlay)
    {
        auto tint = item.base_color_tint;
        auto material = item.material;
        if (packet.visualization == mesh_visualization_mode::cluster_debug)
        {
            tint = cluster_debug_color(item.root_node);
            material = {};
        }
        const auto visualization = packet.visualization == mesh_visualization_mode::cluster_debug
                                       ? mesh_visualization_mode::albedo
                                       : packet.visualization;
        return virtual_cluster_draw{
            .draw = draw_mesh_event{.gpu_scene_instance = item.gpu_scene_instance,
                                    .mesh = item.mesh,
                                    .material = material,
                                    .model = item.model,
                                    .previous_model = item.previous_model,
                                    .view_projection = packet.camera.view_projection,
                                    .previous_view_projection = packet.camera.previous_view_projection,
                                    .world_bounds = item.world_bounds,
                                    .mode = packet.mode,
                                    .visualization = visualization,
                                    .object_id = item.object_id,
                                    .selected = selected_for_overlay,
                                    .casts_shadows = item.casts_shadows,
                                    .receives_shadows = item.receives_shadows,
                                    .mobility = item.mobility,
                                    .shadow_lod_bias = item.shadow_lod_bias,
                                    .maximum_shadow_distance = item.maximum_shadow_distance,
                                    .base_color_tint = tint,
                                    .wire_color = math::vector4f{1.0f, 0.48f, 0.04f, 1.0f},
                                    .label = item.label},
            .mesh = item.mesh,
            .cluster_index = item.root_node};
    };

    frame_directional_lights_.insert(frame_directional_lights_.end(), packet.directional_lights.begin(),
                                     packet.directional_lights.end());
    frame_point_lights_.insert(frame_point_lights_.end(), packet.point_lights.begin(), packet.point_lights.end());
    frame_spot_lights_.insert(frame_spot_lights_.end(), packet.spot_lights.begin(), packet.spot_lights.end());
    frame_area_lights_.insert(frame_area_lights_.end(), packet.area_lights.begin(), packet.area_lights.end());

    if (resolved_config_.features.gpu_driven_rendering)
    {
        for (const auto& item : packet.items)
        {
            if (!item.visible || !item.mesh.valid()) continue;
            frame_draws_.push_back(
                make_draw(item, packet.overlay == editor_overlay_mode::all_wireframe ||
                                    (packet.overlay == editor_overlay_mode::selected_wireframe && item.selected)));
        }
    }
    else
        for (const auto index : packet.visible_items)
        {
            if (index >= packet.items.size()) continue;
            const auto& item = packet.items[index];
            frame_draws_.push_back(
                make_draw(item, packet.overlay == editor_overlay_mode::all_wireframe ||
                                    (packet.overlay == editor_overlay_mode::selected_wireframe && item.selected)));
        }

    if (resolved_config_.features.gpu_driven_rendering)
    {
        for (const auto& item : packet.virtual_items)
        {
            if (!item.visible || !item.mesh.valid()) continue;
            frame_virtual_draws_.push_back(make_virtual_draw(
                item, packet.overlay == editor_overlay_mode::all_wireframe ||
                          (packet.overlay == editor_overlay_mode::selected_wireframe && item.selected)));
        }
    }
    else
        for (const auto index : packet.visible_virtual_items)
        {
            if (index >= packet.virtual_items.size()) continue;
            const auto& item = packet.virtual_items[index];
            frame_virtual_draws_.push_back(make_virtual_draw(
                item, packet.overlay == editor_overlay_mode::all_wireframe ||
                          (packet.overlay == editor_overlay_mode::selected_wireframe && item.selected)));
        }

    for (const auto& item : packet.items)
    {
        if (!item.visible || !item.casts_shadows || !item.mesh.valid()) continue;
        frame_shadow_draws_.push_back(make_draw(item, item.selected));
    }
    for (const auto& item : packet.virtual_items)
    {
        if (!item.visible || !item.casts_shadows || !item.mesh.valid()) continue;
        frame_virtual_shadow_draws_.push_back(make_virtual_draw(item, item.selected));
    }

    for (const auto& patch : packet.visible_terrain_patches)
    {
        if (patch.terrain_index >= packet.terrains.size()) continue;
        const auto& terrain = packet.terrains[patch.terrain_index];
        if (!terrain.terrain.valid()) continue;
        frame_terrain_draws_.push_back({terrain, patch, packet.camera.view_projection,
                                        packet.camera.previous_view_projection, packet.mode, packet.visualization});
    }
    if (resolved_config_.features.gpu_terrain_traversal)
        for (const auto& terrain : packet.terrains)
            if (terrain.terrain.valid() && terrain.gpu_scene_instance.valid())
                frame_gpu_terrain_draws_.push_back({terrain, packet.camera.view_projection,
                                                    packet.camera.previous_view_projection, packet.mode,
                                                    packet.visualization});

    if (!frame_camera_valid_ ||
        math::length_squared(math::sub(packet.camera.position, frame_camera_.position)) > 100.0f)
        exposure_needs_reset_ = true;
    frame_camera_ = packet.camera;
    frame_camera_valid_ = true;
    frame_environment_ = packet.environment;
    frame_shadows_enabled_ = packet.shadows_enabled;
    last_profile_.virtual_geometry.enabled = resolved_config_.features.virtual_geometry;
    last_profile_.virtual_geometry.raster_path = resolved_config_.features.virtual_geometry_path;
    if (!resolved_config_.features.virtual_geometry)
        last_profile_.virtual_geometry.fallback_reason =
            "Vulkan virtual-geometry traversal, streaming, and visibility rasterization are unavailable; "
            "using conventional LOD geometry";
    frame_debug_overlay_lines_.insert(frame_debug_overlay_lines_.end(), packet.debug_overlay.lines.begin(),
                                      packet.debug_overlay.lines.end());
    frame_debug_overlay_triangles_.insert(frame_debug_overlay_triangles_.end(), packet.debug_overlay.triangles.begin(),
                                          packet.debug_overlay.triangles.end());
    last_profile_.terrain.hierarchy_nodes += packet.terrain_statistics.hierarchy_nodes;
    last_profile_.terrain.selected_patches += packet.terrain_statistics.selected_patches;
    last_profile_.terrain.culled_nodes += packet.terrain_statistics.culled_nodes;
    last_profile_.terrain.rendered_triangles += packet.terrain_statistics.rendered_triangles;
    for (std::size_t lod = 0; lod < last_profile_.terrain.patches_per_lod.size(); ++lod)
        last_profile_.terrain.patches_per_lod[lod] += packet.terrain_statistics.patches_per_lod[lod];
    if (resolved_config_.features.gpu_driven_rendering)
    {
        auto& profile = last_profile_.gpu_scene;
        profile.enabled = true;
        profile.hzb_occlusion = resolved_config_.features.hzb_occlusion;
        profile.history_valid = packet.camera.history_valid;
        profile.visible_instances =
            static_cast<std::uint32_t>(packet.visible_items.size() + packet.visible_virtual_items.size());
        profile.frustum_rejected =
            static_cast<std::uint32_t>(packet.culled_item_count + packet.culled_virtual_cluster_count);
        profile.indirect_commands = static_cast<std::uint32_t>(packet.items.size() + packet.virtual_items.size());
        if (resolved_config_.features.gpu_binding_model == gpu_resource_binding_model::classic)
            profile.cpu_submissions += static_cast<std::uint32_t>(packet.items.size() + packet.virtual_items.size() +
                                                                  packet.visible_terrain_patches.size());
    }
}

clustered_light_grid_profile vulkan_render_backend::make_clustered_light_profile() const noexcept
{
    clustered_light_grid_profile profile{};
    const std::uint32_t width = std::max(1u, viewport_width_);
    const std::uint32_t height = std::max(1u, viewport_height_);
    profile.tiles_x = (width + profile.tile_size_pixels - 1u) / profile.tile_size_pixels;
    profile.tiles_y = (height + profile.tile_size_pixels - 1u) / profile.tile_size_pixels;
    profile.cluster_count = profile.tiles_x * profile.tiles_y * profile.depth_slices;
    profile.point_light_references = frame_lighting_.point_count * profile.depth_slices;
    profile.spot_light_references = frame_lighting_.spot_count * profile.depth_slices;
    profile.overflow_count = frame_lighting_.skipped_point_count + frame_lighting_.skipped_spot_count;
    profile.available = true;
    return profile;
}

std::uint64_t vulkan_render_backend::light_shadow_key(render_object_id object) noexcept
{
    return (static_cast<std::uint64_t>(object.generation) << 32u) | static_cast<std::uint64_t>(object.index);
}

void vulkan_render_backend::update_shadow_profile(std::uint64_t frame_index)
{
    auto& profile = last_profile_.shadows;
    profile = {};
    active_local_shadows_.clear();
    frame_lighting_.local_shadow_face_count = 0u;
    profile.directional_cascade_count = resolved_config_.directional_shadow_cascades;
    profile.directional_resolution = resolved_config_.directional_shadow_resolution;
    profile.local_atlas_resolution = resolved_config_.local_shadow_atlas_resolution;
    profile.static_cache_hit = last_static_shadow_cache_hit_;
    if (!frame_shadows_enabled_)
    {
        profile.directional_cascade_count = 0;
        profile.local_atlas_resolution = 0;
        profile.fallback_reason = "Shadows disabled by the viewport";
        return;
    }
    const auto detect_moved_static = [&](draw_mesh_event& draw)
    {
        if (draw.mobility != render_mobility::static_object || !draw.object_id.valid()) return;
        std::uint64_t transform_hash = 1469598103934665603ull;
        for (std::size_t index = 0; index < 16u; ++index)
        {
            const float value = draw.model.data()[index];
            transform_hash ^= std::bit_cast<std::uint32_t>(value);
            transform_hash *= 1099511628211ull;
        }
        const auto key = light_shadow_key(draw.object_id);
        const auto previous = static_shadow_transform_hashes_.find(key);
        if (previous != static_shadow_transform_hashes_.end() && previous->second != transform_hash)
        {
            draw.mobility = render_mobility::movable;
            if (reported_moved_static_objects_.insert(key).second)
                arc::diagnostics::warn(
                    "render.vulkan",
                    "A static shadow caster moved at runtime; treating it as movable while rebuilding caches");
        }
        static_shadow_transform_hashes_[key] = transform_hash;
    };
    for (auto& draw : frame_shadow_draws_)
    {
        detect_moved_static(draw);
        if (!draw.casts_shadows) continue;
        if (draw.mobility == render_mobility::static_object)
            ++profile.static_caster_count;
        else
            ++profile.dynamic_caster_count;
    }
    for (auto& draw : frame_virtual_shadow_draws_)
    {
        detect_moved_static(draw.draw);
        if (!draw.draw.casts_shadows) continue;
        if (draw.draw.mobility == render_mobility::static_object)
            ++profile.static_caster_count;
        else
            ++profile.dynamic_caster_count;
    }
    if (!local_shadow_allocator_) return;

    struct candidate
    {
        shadow_light_kind kind{};
        std::uint64_t key{};
        std::uint32_t resolution{};
        std::uint16_t priority{};
        float score{};
        math::vector3f position{};
        math::vector3f direction{0.0f, -1.0f, 0.0f};
        float range{1.0f};
        float outer_angle{math::pi<float> * 0.25f};
        shadow_settings settings{};
        render_mobility mobility{render_mobility::movable};
        render_object_id object_id{};
    };
    std::vector<candidate> candidates;
    candidates.reserve(frame_point_lights_.size() + frame_spot_lights_.size());
    const auto influence = [&](const math::vector3f& position, float intensity, float range)
    {
        const auto delta = math::sub(position, frame_camera_.position);
        const float distance_squared = std::max(math::length_squared(delta), 1.0f);
        return std::max(intensity, 0.0f) * std::max(range, 0.0f) / distance_squared;
    };
    for (const auto& light : frame_point_lights_)
    {
        if (!light.enabled || !light.casts_shadows || !light.shadow.enabled ||
            resolved_config_.max_shadowed_point_lights == 0u)
            continue;
        candidates.push_back({.kind = shadow_light_kind::point,
                              .key = light_shadow_key(light.object_id),
                              .resolution = light.shadow.resolution,
                              .priority = light.shadow.priority,
                              .score = static_cast<float>(light.shadow.priority) * 100000.0f +
                                       influence(light.position, light.intensity, light.range),
                              .position = light.position,
                              .range = light.range,
                              .settings = light.shadow,
                              .mobility = light.mobility,
                              .object_id = light.object_id});
    }
    for (const auto& light : frame_spot_lights_)
    {
        if (!light.enabled || !light.casts_shadows || !light.shadow.enabled ||
            resolved_config_.max_shadowed_spot_lights == 0u)
            continue;
        candidates.push_back({.kind = shadow_light_kind::spot,
                              .key = light_shadow_key(light.object_id),
                              .resolution = light.shadow.resolution,
                              .priority = light.shadow.priority,
                              .score = static_cast<float>(light.shadow.priority) * 100000.0f +
                                       influence(light.position, light.intensity, light.range),
                              .position = light.position,
                              .direction = light.direction,
                              .range = light.range,
                              .outer_angle = light.outer_angle,
                              .settings = light.shadow,
                              .mobility = light.mobility,
                              .object_id = light.object_id});
    }
    std::stable_sort(candidates.begin(), candidates.end(),
                     [](const candidate& lhs, const candidate& rhs)
                     {
                         if (lhs.score != rhs.score) return lhs.score > rhs.score;
                         if (lhs.kind != rhs.kind) return lhs.kind < rhs.kind;
                         return lhs.key < rhs.key;
                     });

    std::uint32_t point_count{};
    std::uint32_t spot_count{};
    for (const auto& candidate : candidates)
    {
        auto& count = candidate.kind == shadow_light_kind::point ? point_count : spot_count;
        const std::uint32_t budget = candidate.kind == shadow_light_kind::point
                                         ? resolved_config_.max_shadowed_point_lights
                                         : resolved_config_.max_shadowed_spot_lights;
        if (count >= budget) continue;
        const auto allocation = local_shadow_allocator_->allocate(
            {.kind = candidate.kind,
             .light_key = candidate.key,
             .requested_resolution = std::min(candidate.resolution, resolved_config_.max_local_shadow_resolution),
             .minimum_resolution = 128u,
             .priority = candidate.priority,
             .frame_index = frame_index});
        if (allocation)
        {
            ++count;
            bool redraw = true;
            if (candidate.mobility == render_mobility::static_object &&
                candidate.settings.cache_mode != shadow_cache_mode::always_update)
            {
                std::uint64_t signature = 1469598103934665603ull;
                const auto hash = [&](const void* bytes, std::size_t count)
                {
                    const auto* data = static_cast<const std::byte*>(bytes);
                    for (std::size_t index = 0; index < count; ++index)
                    {
                        signature ^= std::to_integer<unsigned char>(data[index]);
                        signature *= 1099511628211ull;
                    }
                };
                hash(&candidate.kind, sizeof(candidate.kind));
                hash(&candidate.position, sizeof(candidate.position));
                hash(&candidate.direction, sizeof(candidate.direction));
                hash(&candidate.range, sizeof(candidate.range));
                hash(&candidate.outer_angle, sizeof(candidate.outer_angle));
                hash(&candidate.settings.resolution, sizeof(candidate.settings.resolution));
                hash(&candidate.settings.priority, sizeof(candidate.settings.priority));
                hash(&candidate.settings.strength, sizeof(candidate.settings.strength));
                hash(&candidate.settings.bias, sizeof(candidate.settings.bias));
                hash(&candidate.settings.normal_bias, sizeof(candidate.settings.normal_bias));
                hash(&candidate.settings.filter, sizeof(candidate.settings.filter));
                hash(&candidate.settings.cache_mode, sizeof(candidate.settings.cache_mode));
                hash(&allocation->handle, sizeof(allocation->handle));
                hash(&shadow_resource_revision_, sizeof(shadow_resource_revision_));
                for (const auto& draw : frame_shadow_draws_)
                {
                    if (!draw.casts_shadows || draw.mobility != render_mobility::static_object) continue;
                    hash(&draw.object_id, sizeof(draw.object_id));
                    hash(draw.model.data(), sizeof(float) * 16u);
                    hash(&draw.mesh, sizeof(draw.mesh));
                    hash(&draw.material, sizeof(draw.material));
                }
                for (const auto& draw : frame_virtual_shadow_draws_)
                {
                    if (!draw.draw.casts_shadows || draw.draw.mobility != render_mobility::static_object) continue;
                    hash(&draw.draw.object_id, sizeof(draw.draw.object_id));
                    hash(draw.draw.model.data(), sizeof(float) * 16u);
                    hash(&draw.mesh, sizeof(draw.mesh));
                    hash(&draw.cluster_index, sizeof(draw.cluster_index));
                }
                const auto cached = local_shadow_static_signatures_.find(candidate.key);
                redraw = cached == local_shadow_static_signatures_.end() || cached->second != signature;
                local_shadow_static_signatures_[candidate.key] = signature;
                if (redraw)
                    ++profile.local_cache_misses;
                else
                    ++profile.local_cache_hits;
            }
            active_local_shadows_.push_back({.kind = candidate.kind,
                                             .allocation = *allocation,
                                             .position = candidate.position,
                                             .direction = candidate.direction,
                                             .range = candidate.range,
                                             .outer_angle = candidate.outer_angle,
                                             .settings = candidate.settings,
                                             .mobility = candidate.mobility,
                                             .redraw = redraw});
            const std::uint32_t first_face = frame_lighting_.local_shadow_face_count;
            const std::uint32_t available = max_local_shadow_faces - first_face;
            const std::uint32_t face_count = std::min(allocation->face_count, available);
            const float inverse_atlas =
                1.0f / static_cast<float>(std::max(resolved_config_.local_shadow_atlas_resolution, 1u));
            static constexpr std::array<math::vector3f, point_shadow_face_count> point_directions{
                math::vector3f{1.0f, 0.0f, 0.0f}, math::vector3f{-1.0f, 0.0f, 0.0f},
                math::vector3f{0.0f, 1.0f, 0.0f}, math::vector3f{0.0f, -1.0f, 0.0f},
                math::vector3f{0.0f, 0.0f, 1.0f}, math::vector3f{0.0f, 0.0f, -1.0f}};
            static constexpr std::array<math::vector3f, point_shadow_face_count> point_ups{
                math::vector3f{0.0f, -1.0f, 0.0f}, math::vector3f{0.0f, -1.0f, 0.0f},
                math::vector3f{0.0f, 0.0f, 1.0f},  math::vector3f{0.0f, 0.0f, -1.0f},
                math::vector3f{0.0f, -1.0f, 0.0f}, math::vector3f{0.0f, -1.0f, 0.0f}};
            const float near_plane = std::clamp(candidate.range * 0.002f, 0.02f, 0.25f);
            const auto projection = perspective_rh_zo(candidate.kind == shadow_light_kind::point
                                                          ? math::pi<float> * 0.5f
                                                          : std::max(candidate.outer_angle * 2.0f, 0.02f),
                                                      near_plane, std::max(candidate.range, near_plane + 0.01f));
            for (std::uint32_t face = 0; face < face_count; ++face)
            {
                const auto direction = candidate.kind == shadow_light_kind::point
                                           ? point_directions[face]
                                           : math::normalize(candidate.direction, 0.0f);
                const auto up = candidate.kind == shadow_light_kind::point
                                    ? point_ups[face]
                                    : (std::abs(direction[1]) > 0.98f ? math::vector3f{0.0f, 0.0f, 1.0f}
                                                                      : math::vector3f{0.0f, 1.0f, 0.0f});
                auto& packed_face = frame_lighting_.local_shadow_faces[first_face + face];
                packed_face.light_view_projection = math::matmul(
                    projection, look_at_rh(candidate.position, math::add(candidate.position, direction), up));
                const auto& rect = allocation->faces[face];
                packed_face.atlas_rect = {static_cast<float>(rect.content_x()) * inverse_atlas,
                                          static_cast<float>(rect.content_y()) * inverse_atlas,
                                          static_cast<float>(rect.content_size()) * inverse_atlas,
                                          static_cast<float>(rect.content_size()) * inverse_atlas};
                packed_face.parameters = {inverse_atlas, std::max(candidate.settings.bias, 0.0f), near_plane,
                                          candidate.range};
            }
            frame_lighting_.local_shadow_face_count += face_count;
            const auto patch_light = [&](auto& packed_lights, std::uint32_t packed_count)
            {
                for (std::uint32_t index = 0; index < packed_count; ++index)
                {
                    auto& packed = packed_lights[index];
                    if (static_cast<std::uint32_t>(packed.object_id_shadow[0]) != candidate.object_id.index ||
                        static_cast<std::uint32_t>(packed.object_id_shadow[1]) != candidate.object_id.generation)
                        continue;
                    packed.shadow_parameters = {static_cast<float>(first_face), static_cast<float>(face_count),
                                                std::clamp(candidate.settings.strength, 0.0f, 1.0f),
                                                std::max(candidate.settings.normal_bias, 0.0f)};
                    break;
                }
            };
            if (candidate.kind == shadow_light_kind::point)
                patch_light(frame_lighting_.point_lights, frame_lighting_.point_count);
            else
                patch_light(frame_lighting_.spot_lights, frame_lighting_.spot_count);
        }
        else
            profile.fallback_reason = "local shadow atlas exhausted; affected lights render unshadowed";
    }
    profile.shadowed_point_lights = point_count;
    profile.shadowed_spot_lights = spot_count;
    const auto statistics = local_shadow_allocator_->statistics();
    profile.local_allocation_count = statistics.allocation_count;
    profile.local_occupied_texels = statistics.occupied_texels;
    profile.local_eviction_count = statistics.eviction_count;
    profile.local_resolution_reductions = statistics.resolution_reduction_count;
    profile.screen_space_shadows = false;
    if (resolved_config_.screen_space_shadows && profile.fallback_reason.empty())
        profile.fallback_reason =
            "screen-space shadow passes selected but unavailable in the Vulkan compatibility path";
}

} // namespace arc::render::vulkan::backend_detail
