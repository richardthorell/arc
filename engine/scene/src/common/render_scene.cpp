#include <arc/scene/render_scene.h>
#include <arc/scene/environment.h>

#include <arc/diagnostics/log.h>
#include <arc/render/lighting.h>
#include <arc/render/render_world.h>
#include <arc/scene/transforms.h>

#include <algorithm>
#include <cmath>
#include <memory>

namespace arc::scene
{
namespace
{

bool entity_is_active(const registry& scene, entity value)
{
    const auto* active = scene.try_get<active_component>(value);
    return !active || active->active;
}

std::string entity_label(const registry& scene, entity value)
{
    const auto* name = scene.try_get<name_component>(value);
    return name ? name->value : std::string{};
}

std::uint32_t render_layer_mask(const registry& scene, entity value)
{
    const auto* layer = scene.try_get<render_layer_component>(value);
    return layer ? layer->mask : 1u;
}

geometric::box3f transform_bounds(const geometric::box3f& local_bounds, const math::matrix4f& world)
{
    const auto corner_at = [&](std::uint32_t index) {
        return math::vector3f{
            (index & 1u) != 0u ? local_bounds.max[0] : local_bounds.min[0],
            (index & 2u) != 0u ? local_bounds.max[1] : local_bounds.min[1],
            (index & 4u) != 0u ? local_bounds.max[2] : local_bounds.min[2]
        };
    };

    auto transformed = math::transform_point(world, corner_at(0));
    geometric::box3f result{
        geometric::point3f{ transformed },
        geometric::point3f{ transformed }
    };
    for (std::uint32_t index = 1; index < 8; ++index)
    {
        transformed = math::transform_point(world, corner_at(index));
        result = geometric::expand(result, geometric::point3f{ transformed });
    }
    return result;
}

geometric::box3f cluster_bounds(const render::virtual_mesh_cluster& cluster, const math::matrix4f& world)
{
    return transform_bounds(
        geometric::box3f{
            geometric::point3f{ cluster.bounds_min },
            geometric::point3f{ cluster.bounds_max } },
        world);
}

geometric::box3f world_bounds_for(const registry& scene, entity value, const transform_component& transform)
{
    const auto* bounds = scene.try_get<bounds_component>(value);
    if (bounds)
        return bounds->dirty ? transform_bounds(bounds->local_bounds, transform.dirty ? local_matrix(transform) : transform.world) : bounds->world_bounds;
    return geometric::box3f{
        geometric::point3f{ transform.position - math::vector3f{ 0.5f, 0.5f, 0.5f } },
        geometric::point3f{ transform.position + math::vector3f{ 0.5f, 0.5f, 0.5f } }
    };
}

bool entity_selected(const registry& scene, entity value)
{
    const auto* selection = scene.try_get<selection_component>(value);
    return selection && selection->selected;
}

math::vector3f effective_light_color(const math::vector3f& color, bool use_temperature, float kelvin)
{
    if (!use_temperature)
        return color;
    return math::mul(color, render::color_temperature_rgb(kelvin));
}

void append_mesh_item(
    registry& scene,
    render::render_world_packet& packet,
    render_scene_result& result,
    entity value,
    const transform_component& transform,
    render::mesh_handle mesh,
    render::material_handle material,
    bool visible,
    bool transparent = false,
    render::buffer_handle skin = {},
    std::uint32_t joint_count = 0,
    std::uint32_t instance_count = 1,
    const math::vector4f& base_color_tint = math::vector4f{ 1.0f, 1.0f, 1.0f, 1.0f })
{
    if (!entity_is_active(scene, value))
        return;

    ++result.renderable_count;
    const bool selected = entity_selected(scene, value);
    if (selected)
        ++result.selected_count;

    const math::matrix4f world = transform.dirty ? local_matrix(transform) : transform.world;
    packet.items.push_back({
        .mesh = mesh,
        .material = material,
        .model = world,
        .previous_model = world,
        .world_bounds = world_bounds_for(scene, value, transform),
        .render_layer_mask = render_layer_mask(scene, value),
        .object_id = render::make_render_object_id(value.index, value.generation),
        .skin_matrices = skin,
        .skin_joint_count = joint_count,
        .instance_count = instance_count,
        .visible = visible,
        .selected = selected,
        .transparent = transparent,
        .base_color_tint = base_color_tint,
        .label = entity_label(scene, value)
    });
}

void append_virtual_mesh_items(
    registry& scene,
    render::renderer& renderer,
    render::render_world_packet& packet,
    render_scene_result& result,
    entity value,
    const transform_component& transform,
    const virtual_mesh_renderer_component& mesh_renderer)
{
    if (!entity_is_active(scene, value))
        return;

    ++result.renderable_count;
    const bool selected = entity_selected(scene, value);
    if (selected)
        ++result.selected_count;

    if (!mesh_renderer.visible || !renderer.virtual_mesh_alive(mesh_renderer.mesh))
        return;

    const auto* mesh = renderer.virtual_mesh_data_for(mesh_renderer.mesh);
    if (!mesh)
        return;

    const math::matrix4f world = transform.dirty ? local_matrix(transform) : transform.world;
    const auto object = render::make_render_object_id(value.index, value.generation);
    const auto layer_mask = render_layer_mask(scene, value);
    const auto label = entity_label(scene, value);
    for (std::uint32_t cluster_index = 0; cluster_index < mesh->clusters.size(); ++cluster_index)
    {
        const auto& cluster = mesh->clusters[cluster_index];
        packet.virtual_items.push_back({
            .mesh = mesh_renderer.mesh,
            .material = mesh_renderer.material,
            .cluster_index = cluster_index,
            .model = world,
            .previous_model = world,
            .world_bounds = cluster_bounds(cluster, world),
            .render_layer_mask = layer_mask,
            .object_id = object,
            .visible = mesh_renderer.visible,
            .selected = selected,
            .base_color_tint = mesh_renderer.base_color_tint,
            .label = label
        });
    }
}

bool environment_mesh_visible(
    const registry& scene,
    entity value,
    const render_environment_visibility& visibility,
    bool& transparent)
{
    transparent = false;
    if (const auto* terrain = scene.try_get<terrain_component>(value))
        return visibility.terrain && terrain->enabled;
    if (const auto* water = scene.try_get<water_component>(value))
    {
        transparent = true;
        return visibility.water && water->enabled;
    }
    if (const auto* vegetation = scene.try_get<vegetation_component>(value))
        return visibility.vegetation && vegetation->enabled;
    return true;
}

void apply_lod(
    const registry& scene,
    entity value,
    render::mesh_handle& mesh,
    render::material_handle& material)
{
    const auto* lod = scene.try_get<lod_component>(value);
    if (!lod || !lod->enabled || lod->levels.empty())
        return;

    for (const auto& level : lod->levels)
    {
        if (level.mesh.valid())
        {
            mesh = level.mesh;
            material = level.material.valid() ? level.material : material;
            return;
        }
    }
}

} // namespace

render_scene_result render_scene(
    registry& scene,
    render::renderer& renderer,
    std::uint32_t viewport_width,
    std::uint32_t viewport_height,
    render::render_mode mode,
    render::mesh_visualization_mode visualization,
    render::editor_overlay_mode overlay,
    bool shadows_enabled,
    render_environment_visibility environment_visibility,
    float delta_seconds)
{
    render_scene_result result{};
    update_world_environments(scene, delta_seconds);

    const transform_component* camera_transform{};
    const camera_component* camera{};
    scene.view<transform_component, camera_component>().each(
        [&](entity value, const transform_component& transform, const camera_component& candidate) {
            if (!camera && candidate.active && entity_is_active(scene, value))
            {
                camera_transform = &transform;
                camera = &candidate;
            }
        });

    if (!camera || !camera_transform)
    {
        static bool warned{};
        if (!warned)
        {
            arc::warn("scene", "No active camera found for render extraction");
            warned = true;
        }
        return result;
    }

    result.camera_found = true;
    const float aspect = viewport_height == 0 ? 1.0f : static_cast<float>(viewport_width) / static_cast<float>(viewport_height);
    const math::matrix4f view = view_matrix(*camera_transform);
    const math::matrix4f projection = camera->projection == camera_projection::orthographic
        ? orthographic_rh_zo(camera->orthographic_height, aspect, camera->near_plane, camera->far_plane)
        : perspective_rh_zo(camera->fov_y_radians, aspect, camera->near_plane, camera->far_plane);
    const math::matrix4f vp = math::matmul(projection, view);

    render::render_world_packet world_packet;
    world_packet.camera.view = view;
    world_packet.camera.projection = projection;
    world_packet.camera.view_projection = vp;
    world_packet.camera.previous_view_projection = vp;
    world_packet.camera.position = camera_transform->position;
    world_packet.camera.forward = forward_direction(*camera_transform);
    world_packet.camera.up = up_direction(*camera_transform);
    world_packet.camera.clear_color = camera->clear_color;
    world_packet.camera.near_plane = camera->near_plane;
    world_packet.camera.far_plane = camera->far_plane;
    world_packet.camera.render_width = viewport_width;
    world_packet.camera.render_height = viewport_height;
    world_packet.camera.output_width = viewport_width;
    world_packet.camera.output_height = viewport_height;
    world_packet.mode = mode;
    world_packet.visualization = visualization;
    world_packet.overlay = overlay;
    world_packet.shadows_enabled = shadows_enabled;
    world_packet.viewport_width = viewport_width;
    world_packet.viewport_height = viewport_height;

    scene.view<world_environment_component>().each(
        [&](entity value, const world_environment_component& world) {
            if (!entity_is_active(scene, value))
                return;
            ++result.world_environment_count;
            if (result.world_environment_count > 1)
            {
                if (world_packet.environment.fallback_reason.empty())
                    world_packet.environment.fallback_reason =
                        "Multiple active world environments found; using the first active entity";
                return;
            }
            auto& output = world_packet.environment;
            output.enabled = world.enabled;
            output.sky_visible = world.enabled && world.sky_visible && environment_visibility.sky;
            output.affect_lighting = world.enabled && world.affect_lighting;
            output.source = static_cast<render::sky_source_mode>(world.source);
            output.solid_color = world.solid_color;
            output.hdri_texture = world.hdri_texture;
            output.hdri_rotation_degrees = world.hdri_rotation_degrees;
            output.radiance_intensity = world.radiance_intensity;
            output.label = entity_label(scene, value);

            if (const auto* sky = scene.try_get<sky_atmosphere_component>(value))
            {
                output.atmosphere = {
                    .enabled = sky->enabled,
                    .planet_radius = sky->planet_radius,
                    .atmosphere_radius = sky->atmosphere_radius,
                    .rayleigh_strength = sky->rayleigh_strength,
                    .mie_strength = sky->mie_strength,
                    .ozone_strength = sky->ozone_strength,
                    .tint = sky->tint,
                    .ground_albedo = sky->ground_albedo,
                    .mie_anisotropy = sky->mie_anisotropy,
                    .rayleigh_scale_height = sky->rayleigh_scale_height,
                    .mie_scale_height = sky->mie_scale_height,
                    .multi_scattering_factor = sky->multi_scattering_factor,
                    .exposure = sky->exposure,
                    .sun_disk_size = sky->sun_disk_size,
                    .sun_disk_intensity = sky->sun_disk_intensity,
                    .label = output.label
                };
                ++result.sky_atmosphere_count;
            }

            if (const auto* celestial = scene.try_get<celestial_sky_component>(value))
            {
                auto sun_direction = math::vector3f{ 0.35f, -0.85f, -0.4f };
                if (scene.alive(celestial->sun_light))
                {
                    if (const auto* transform = scene.try_get<transform_component>(celestial->sun_light))
                        sun_direction = forward_direction(*transform);
                }
                const float phase = celestial->automatic_moon_phase
                    ? calculate_moon_phase(celestial->year, celestial->month, celestial->day,
                        celestial->local_time_hours, celestial->utc_offset_hours)
                    : std::clamp(celestial->moon_phase, 0.0f, 1.0f);
                auto moon_direction = math::normalize(math::vector3f{
                    -sun_direction[0] + 0.18f * std::sin(phase * 6.28318530718f),
                    -sun_direction[1],
                    -sun_direction[2] + 0.12f * std::cos(phase * 6.28318530718f) });
                const auto* authored_atmosphere = scene.try_get<sky_atmosphere_component>(value);
                const float sun_angular_radius = authored_atmosphere
                    ? std::max(0.01f, authored_atmosphere->sun_disk_size * 10.664f)
                    : 0.2666f;
                output.celestial = {
                    .enabled = celestial->enabled,
                    .sun_direction = sun_direction,
                    .moon_direction = moon_direction,
                    .sun_angular_radius_degrees = sun_angular_radius,
                    .sun_intensity = celestial->sun_intensity_multiplier,
                    .moon_enabled = celestial->moon_enabled,
                    .moon_phase = phase,
                    .moon_intensity = celestial->moon_intensity,
                    .moon_angular_radius_degrees = celestial->moon_angular_radius_degrees,
                    .stars_enabled = celestial->stars_enabled,
                    .star_density = celestial->star_density,
                    .star_intensity = celestial->star_intensity,
                    .star_twinkle = celestial->star_twinkle,
                    .time_seconds = celestial->animation_time_seconds
                };
            }

            const auto cloud_data = [](const cloud_layer_settings& layer) {
                return render::cloud_layer_data{
                    .enabled = layer.enabled,
                    .coverage = layer.coverage,
                    .density = layer.density,
                    .altitude = layer.altitude,
                    .thickness = layer.thickness,
                    .scale = layer.scale,
                    .detail = layer.detail,
                    .softness = layer.softness,
                    .wind_direction = layer.wind_direction,
                    .wind_speed = layer.wind_speed,
                    .lighting_strength = layer.lighting_strength,
                    .silver_lining = layer.silver_lining
                };
            };
            if (const auto* clouds = scene.try_get<cloud_layers_component>(value))
            {
                output.clouds = {
                    .enabled = clouds->enabled,
                    .cast_shadows = clouds->cast_shadows,
                    .cumulus = cloud_data(clouds->cumulus),
                    .cirrus = cloud_data(clouds->cirrus)
                };
            }
            if (const auto* lighting = scene.try_get<environment_lighting_component>(value))
            {
                output.lighting = {
                    .enabled = lighting->enabled,
                    .source = static_cast<render::environment_lighting_source_mode>(lighting->source),
                    .environment = lighting->environment,
                    .hdri_texture = lighting->hdri_texture,
                    .constant_color = lighting->constant_color,
                    .diffuse_intensity = lighting->diffuse_intensity,
                    .specular_intensity = lighting->specular_intensity
                };
            }
            if (const auto* fog = scene.try_get<height_fog_component>(value);
                fog && world.enabled && environment_visibility.fog && fog->enabled)
            {
                output.fog = {
                    .enabled = true,
                    .color = fog->color,
                    .density = fog->density,
                    .height_falloff = fog->height_falloff,
                    .start_distance = fog->start_distance,
                    .max_opacity = fog->max_opacity,
                    .sun_scattering_strength = fog->sun_scattering_strength,
                    .label = output.label
                };
                ++result.height_fog_count;
            }
        });

    // Compatibility for scenes authored before world_environment_component.
    scene.view<sky_atmosphere_component>().each(
        [&](entity value, const sky_atmosphere_component& sky) {
            if (result.world_environment_count != 0 || !environment_visibility.sky ||
                !entity_is_active(scene, value) || !sky.enabled)
                return;
            world_packet.environment.enabled = true;
            world_packet.environment.sky_visible = true;
            world_packet.environment.affect_lighting = true;
            world_packet.environment.atmosphere.enabled = true;
            world_packet.environment.atmosphere.planet_radius = sky.planet_radius;
            world_packet.environment.atmosphere.atmosphere_radius = sky.atmosphere_radius;
            world_packet.environment.atmosphere.rayleigh_strength = sky.rayleigh_strength;
            world_packet.environment.atmosphere.mie_strength = sky.mie_strength;
            world_packet.environment.atmosphere.ozone_strength = sky.ozone_strength;
            world_packet.environment.atmosphere.tint = sky.tint;
            world_packet.environment.atmosphere.ground_albedo = sky.ground_albedo;
            world_packet.environment.atmosphere.mie_anisotropy = sky.mie_anisotropy;
            world_packet.environment.atmosphere.exposure = sky.exposure;
            world_packet.environment.atmosphere.sun_disk_size = sky.sun_disk_size;
            world_packet.environment.atmosphere.sun_disk_intensity = sky.sun_disk_intensity;
            world_packet.environment.label = entity_label(scene, value);
            ++result.sky_atmosphere_count;
        });

    scene.view<height_fog_component>().each(
        [&](entity value, const height_fog_component& fog) {
            if (result.world_environment_count != 0 || world_packet.environment.fog.enabled ||
                !environment_visibility.fog || !entity_is_active(scene, value) || !fog.enabled)
                return;
            world_packet.environment.fog = {
                .enabled = true,
                .color = fog.color,
                .density = fog.density,
                .height_falloff = fog.height_falloff,
                .start_distance = fog.start_distance,
                .max_opacity = fog.max_opacity,
                .sun_scattering_strength = fog.sun_scattering_strength,
                .label = entity_label(scene, value)
            };
            ++result.height_fog_count;
        });

    scene.view<transform_component, mesh_renderer_component>().each(
        [&](entity value, const transform_component& transform, const mesh_renderer_component& mesh_renderer) {
            bool transparent{};
            if (!environment_mesh_visible(scene, value, environment_visibility, transparent))
                return;
            auto mesh = mesh_renderer.mesh;
            auto material = mesh_renderer.material;
            apply_lod(scene, value, mesh, material);
            append_mesh_item(scene, world_packet, result, value, transform, mesh, material, mesh_renderer.visible, transparent, {}, 0, 1, mesh_renderer.base_color_tint);
        });

    scene.view<transform_component, virtual_mesh_renderer_component>().each(
        [&](entity value, const transform_component& transform, const virtual_mesh_renderer_component& mesh_renderer) {
            append_virtual_mesh_items(scene, renderer, world_packet, result, value, transform, mesh_renderer);
        });

    scene.view<transform_component, terrain_component>().each(
        [&](entity value, const transform_component& transform, const terrain_component& terrain) {
            if (!environment_visibility.terrain || !entity_is_active(scene, value) || !terrain.enabled)
                return;
            world_packet.terrains.push_back({
                .object_id = render::make_render_object_id(value.index, value.generation),
                .position = transform.position,
                .size = terrain.size,
                .subdivisions = terrain.subdivisions,
                .height_scale = terrain.height_scale,
                .receive_shadows = terrain.receive_shadows,
                .label = entity_label(scene, value)
            });
            ++result.terrain_count;
        });

    scene.view<transform_component, water_component>().each(
        [&](entity value, const transform_component& transform, const water_component& water) {
            if (!environment_visibility.water || !entity_is_active(scene, value) || !water.enabled)
                return;
            world_packet.waters.push_back({
                .object_id = render::make_render_object_id(value.index, value.generation),
                .position = transform.position,
                .size = water.size,
                .color = water.color,
                .roughness = water.roughness,
                .wave_scale = water.wave_scale,
                .wave_speed = water.wave_speed,
                .transparency = water.transparency,
                .label = entity_label(scene, value)
            });
            ++result.water_count;
        });

    scene.view<transform_component, vegetation_component>().each(
        [&](entity value, const transform_component& transform, const vegetation_component& vegetation) {
            if (!environment_visibility.vegetation || !entity_is_active(scene, value) || !vegetation.enabled)
                return;
            world_packet.vegetation.push_back({
                .object_id = render::make_render_object_id(value.index, value.generation),
                .position = transform.position,
                .density = vegetation.density,
                .patch_size = vegetation.patch_size,
                .color = vegetation.color,
                .wind_strength = vegetation.wind_strength,
                .wind_speed = vegetation.wind_speed,
                .label = entity_label(scene, value)
            });
            ++result.vegetation_count;
            result.vegetation_instance_count += vegetation.density;
        });

    scene.view<transform_component, decal_component>().each(
        [&](entity value, const transform_component& transform, const decal_component& decal) {
            if (!environment_visibility.decals || !entity_is_active(scene, value) || !decal.enabled)
                return;
            const math::matrix4f world = transform.dirty ? local_matrix(transform) : transform.world;
            world_packet.decals.push_back({
                .object_id = render::make_render_object_id(value.index, value.generation),
                .model = world,
                .world_bounds = world_bounds_for(scene, value, transform),
                .color = decal.color,
                .texture = decal.texture,
                .opacity = decal.opacity,
                .label = entity_label(scene, value)
            });
            ++result.decal_count;
        });

    scene.view<transform_component, skinned_mesh_renderer_component>().each(
        [&](entity value, const transform_component& transform, const skinned_mesh_renderer_component& mesh_renderer) {
            append_mesh_item(
                scene,
                world_packet,
                result,
                value,
                transform,
                mesh_renderer.mesh,
                mesh_renderer.material,
                mesh_renderer.visible,
                false,
                mesh_renderer.skin_matrices,
                mesh_renderer.joint_count);
        });

    scene.view<transform_component, instance_group_component>().each(
        [&](entity value, const transform_component& transform, const instance_group_component& instances) {
            append_mesh_item(
                scene,
                world_packet,
                result,
                value,
                transform,
                instances.mesh,
                instances.material,
                instances.visible,
                false,
                {},
                0,
                instances.instance_count);
        });

    scene.view<transform_component, directional_light_component>().each(
        [&](entity value, const transform_component& transform, const directional_light_component& light) {
            if (!entity_is_active(scene, value) || !light.enabled)
                return;
            world_packet.directional_lights.push_back({
                .direction = forward_direction(transform),
                .color = effective_light_color(light.color, light.use_color_temperature, light.temperature_kelvin),
                .intensity = light.intensity,
                .casts_shadows = light.casts_shadows,
                .enabled = light.enabled,
                .use_color_temperature = light.use_color_temperature,
                .temperature_kelvin = light.temperature_kelvin,
                .intensity_unit = light.intensity_unit,
                .cookie_texture = light.cookie_texture,
                .shadow = light.shadow,
                .label = entity_label(scene, value)
            });
            ++result.directional_light_count;
        });

    scene.view<transform_component, point_light_component>().each(
        [&](entity value, const transform_component& transform, const point_light_component& light) {
            if (!entity_is_active(scene, value) || !light.enabled)
                return;
            world_packet.point_lights.push_back({
                .position = transform.position,
                .color = effective_light_color(light.color, light.use_color_temperature, light.temperature_kelvin),
                .intensity = light.intensity,
                .range = light.range,
                .casts_shadows = light.casts_shadows,
                .enabled = light.enabled,
                .use_color_temperature = light.use_color_temperature,
                .temperature_kelvin = light.temperature_kelvin,
                .intensity_unit = light.intensity_unit,
                .cookie_texture = light.cookie_texture,
                .shadow = light.shadow,
                .label = entity_label(scene, value)
            });
            ++result.point_light_count;
        });

    scene.view<transform_component, spot_light_component>().each(
        [&](entity value, const transform_component& transform, const spot_light_component& light) {
            if (!entity_is_active(scene, value) || !light.enabled)
                return;
            world_packet.spot_lights.push_back({
                .position = transform.position,
                .direction = forward_direction(transform),
                .color = effective_light_color(light.color, light.use_color_temperature, light.temperature_kelvin),
                .intensity = light.intensity,
                .range = light.range,
                .inner_angle = light.inner_angle,
                .outer_angle = light.outer_angle,
                .casts_shadows = light.casts_shadows,
                .enabled = light.enabled,
                .use_color_temperature = light.use_color_temperature,
                .temperature_kelvin = light.temperature_kelvin,
                .intensity_unit = light.intensity_unit,
                .cookie_texture = light.cookie_texture,
                .shadow = light.shadow,
                .label = entity_label(scene, value)
            });
            ++result.spot_light_count;
        });

    scene.view<transform_component, reflection_probe_component>().each(
        [&](entity value, const transform_component& transform, const reflection_probe_component& probe) {
            if (!entity_is_active(scene, value) || !probe.enabled)
                return;
            world_packet.reflection_probes.push_back({
                .position = transform.position,
                .radius = probe.radius,
                .intensity = probe.intensity,
                .label = entity_label(scene, value)
            });
            ++result.reflection_probe_count;
        });

    scene.view<transform_component, irradiance_probe_component>().each(
        [&](entity value, const transform_component& transform, const irradiance_probe_component& probe) {
            if (!entity_is_active(scene, value) || !probe.enabled)
                return;
            world_packet.irradiance_probes.push_back({
                .position = transform.position,
                .radius = probe.radius,
                .intensity = probe.intensity,
                .label = entity_label(scene, value)
            });
            ++result.irradiance_probe_count;
        });

    const auto lighting = render::pack_scene_lighting(
        world_packet.directional_lights,
        world_packet.point_lights,
        world_packet.spot_lights);
    result.skipped_directional_light_count = lighting.skipped_directional_count;
    result.skipped_point_light_count = lighting.skipped_point_count;
    result.skipped_spot_light_count = lighting.skipped_spot_count;

    result.environment = world_packet.environment;
    render::prepare_render_world(world_packet);
    result.submitted_draw_count = world_packet.visible_items.size() + world_packet.visible_virtual_items.size();
    result.culled_count = world_packet.culled_item_count;
    result.culled_virtual_cluster_count = world_packet.culled_virtual_cluster_count;
    result.instance_batch_count = world_packet.instance_batches.size();
    result.indirect_draw_count = world_packet.indirect_draws.size();

    render::render_event_buffer buffer;
    render::render_event_writer writer(buffer);
    writer.render_world(std::make_shared<render::render_world_packet>(std::move(world_packet)), "scene");
    renderer.frame_queue().submit(std::move(buffer));
    return result;
}

} // namespace arc::scene
