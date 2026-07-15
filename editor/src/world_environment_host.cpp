#include <arc/editor/world_environment_host.h>

namespace arc::editor
{
namespace
{

host_vec3 to_host(const math::vector3f& value) noexcept { return { value[0], value[1], value[2] }; }
math::vector3f to_scene(host_vec3 value) noexcept { return { value.x, value.y, value.z }; }

host_sky_source to_host(scene::sky_source value) noexcept
{
    switch (value)
    {
    case scene::sky_source::hdri: return host_sky_source::hdri;
    case scene::sky_source::solid_color: return host_sky_source::solid_color;
    case scene::sky_source::physical_atmosphere: break;
    }
    return host_sky_source::physical_atmosphere;
}

scene::sky_source to_scene(host_sky_source value) noexcept
{
    switch (value)
    {
    case host_sky_source::hdri: return scene::sky_source::hdri;
    case host_sky_source::solid_color: return scene::sky_source::solid_color;
    case host_sky_source::physical_atmosphere: break;
    }
    return scene::sky_source::physical_atmosphere;
}

host_sun_position_mode to_host(scene::sun_position_mode value) noexcept
{
    return value == scene::sun_position_mode::geographic
        ? host_sun_position_mode::geographic
        : host_sun_position_mode::manual_light;
}

scene::sun_position_mode to_scene(host_sun_position_mode value) noexcept
{
    return value == host_sun_position_mode::geographic
        ? scene::sun_position_mode::geographic
        : scene::sun_position_mode::manual_light;
}

host_celestial_time_mode to_host(scene::celestial_time_mode value) noexcept
{
    switch (value)
    {
    case scene::celestial_time_mode::simulated: return host_celestial_time_mode::simulated;
    case scene::celestial_time_mode::system_clock: return host_celestial_time_mode::system_clock;
    case scene::celestial_time_mode::fixed: break;
    }
    return host_celestial_time_mode::fixed;
}

scene::celestial_time_mode to_scene(host_celestial_time_mode value) noexcept
{
    switch (value)
    {
    case host_celestial_time_mode::simulated: return scene::celestial_time_mode::simulated;
    case host_celestial_time_mode::system_clock: return scene::celestial_time_mode::system_clock;
    case host_celestial_time_mode::fixed: break;
    }
    return scene::celestial_time_mode::fixed;
}

host_environment_lighting_source to_host(scene::environment_lighting_source value) noexcept
{
    switch (value)
    {
    case scene::environment_lighting_source::hdri: return host_environment_lighting_source::hdri;
    case scene::environment_lighting_source::constant_color: return host_environment_lighting_source::constant_color;
    case scene::environment_lighting_source::follow_sky: break;
    }
    return host_environment_lighting_source::follow_sky;
}

scene::environment_lighting_source to_scene(host_environment_lighting_source value) noexcept
{
    switch (value)
    {
    case host_environment_lighting_source::hdri: return scene::environment_lighting_source::hdri;
    case host_environment_lighting_source::constant_color: return scene::environment_lighting_source::constant_color;
    case host_environment_lighting_source::follow_sky: break;
    }
    return scene::environment_lighting_source::follow_sky;
}

host_cloud_layer to_host(const scene::cloud_layer_settings& layer) noexcept
{
    return { layer.enabled, layer.coverage, layer.density, layer.altitude, layer.thickness,
        layer.scale, layer.detail, layer.softness, layer.wind_direction[0], layer.wind_direction[1],
        layer.wind_speed, layer.lighting_strength, layer.silver_lining };
}

scene::cloud_layer_settings to_scene(const host_cloud_layer& layer) noexcept
{
    return { layer.enabled, layer.coverage, layer.density, layer.altitude, layer.thickness,
        layer.scale, layer.detail, layer.softness, { layer.wind_x, layer.wind_y },
        layer.wind_speed, layer.lighting_strength, layer.silver_lining };
}

} // namespace

host_world_environment_snapshot to_host_world_environment_snapshot(
    host_entity_id entity,
    const scene::world_environment_settings& settings,
    const std::filesystem::path& hdri_path)
{
    const auto& world = settings.world;
    const auto& atmosphere = settings.atmosphere;
    const auto& celestial = settings.celestial;
    const auto& clouds = settings.clouds;
    const auto& fog = settings.fog;
    const auto& lighting = settings.lighting;
    host_world_environment_snapshot result;
    result.entity = entity;
    result.enabled = world.enabled;
    result.sky_visible = world.sky_visible;
    result.affect_lighting = world.affect_lighting;
    result.sky_source = to_host(world.source);
    result.solid_color = to_host(world.solid_color);
    result.hdri_path = hdri_path.generic_string();
    result.hdri_rotation_degrees = world.hdri_rotation_degrees;
    result.radiance_intensity = world.radiance_intensity;
    result.planet_radius = atmosphere.planet_radius;
    result.atmosphere_radius = atmosphere.atmosphere_radius;
    result.rayleigh_strength = atmosphere.rayleigh_strength;
    result.mie_strength = atmosphere.mie_strength;
    result.ozone_strength = atmosphere.ozone_strength;
    result.atmosphere_tint = to_host(atmosphere.tint);
    result.ground_albedo = to_host(atmosphere.ground_albedo);
    result.mie_anisotropy = atmosphere.mie_anisotropy;
    result.rayleigh_scale_height = atmosphere.rayleigh_scale_height;
    result.mie_scale_height = atmosphere.mie_scale_height;
    result.multi_scattering_factor = atmosphere.multi_scattering_factor;
    result.exposure = atmosphere.exposure;
    result.sun_disk_size = atmosphere.sun_disk_size;
    result.sun_disk_intensity = atmosphere.sun_disk_intensity;
    result.sun_mode = to_host(celestial.sun_mode);
    result.time_mode = to_host(celestial.time_mode);
    result.latitude_degrees = celestial.latitude_degrees;
    result.longitude_degrees = celestial.longitude_degrees;
    result.north_offset_degrees = celestial.north_offset_degrees;
    result.year = celestial.year;
    result.month = celestial.month;
    result.day = celestial.day;
    result.local_time_hours = celestial.local_time_hours;
    result.utc_offset_hours = celestial.utc_offset_hours;
    result.playing = celestial.playing;
    result.loop_day = celestial.loop_day;
    result.time_scale = celestial.time_scale;
    result.automatic_sun_light = celestial.automatic_sun_light;
    result.sun_intensity_multiplier = celestial.sun_intensity_multiplier;
    result.sun_temperature_multiplier = celestial.sun_temperature_multiplier;
    result.moon_enabled = celestial.moon_enabled;
    result.automatic_moon_phase = celestial.automatic_moon_phase;
    result.moon_phase = celestial.moon_phase;
    result.moon_intensity = celestial.moon_intensity;
    result.moon_angular_radius_degrees = celestial.moon_angular_radius_degrees;
    result.stars_enabled = celestial.stars_enabled;
    result.star_density = celestial.star_density;
    result.star_intensity = celestial.star_intensity;
    result.star_twinkle = celestial.star_twinkle;
    result.clouds_enabled = clouds.enabled;
    result.cloud_shadows = clouds.cast_shadows;
    result.cumulus = to_host(clouds.cumulus);
    result.cirrus = to_host(clouds.cirrus);
    result.fog_enabled = fog.enabled;
    result.fog_color = to_host(fog.color);
    result.fog_density = fog.density;
    result.fog_height_falloff = fog.height_falloff;
    result.fog_start_distance = fog.start_distance;
    result.fog_max_opacity = fog.max_opacity;
    result.fog_sun_scattering = fog.sun_scattering_strength;
    result.lighting_enabled = lighting.enabled;
    result.lighting_source = to_host(lighting.source);
    result.lighting_color = to_host(lighting.constant_color);
    result.diffuse_intensity = lighting.diffuse_intensity;
    result.specular_intensity = lighting.specular_intensity;
    return result;
}

scene::world_environment_settings apply_host_world_environment_snapshot(
    const host_world_environment_snapshot& snapshot,
    const scene::world_environment_settings& current)
{
    auto result = current;
    auto& world = result.world;
    world.enabled = snapshot.enabled;
    world.sky_visible = snapshot.sky_visible;
    world.affect_lighting = snapshot.affect_lighting;
    world.source = to_scene(snapshot.sky_source);
    world.solid_color = to_scene(snapshot.solid_color);
    world.hdri_rotation_degrees = snapshot.hdri_rotation_degrees;
    world.radiance_intensity = snapshot.radiance_intensity;
    auto& atmosphere = result.atmosphere;
    atmosphere.planet_radius = snapshot.planet_radius;
    atmosphere.atmosphere_radius = snapshot.atmosphere_radius;
    atmosphere.rayleigh_strength = snapshot.rayleigh_strength;
    atmosphere.mie_strength = snapshot.mie_strength;
    atmosphere.ozone_strength = snapshot.ozone_strength;
    atmosphere.tint = to_scene(snapshot.atmosphere_tint);
    atmosphere.ground_albedo = to_scene(snapshot.ground_albedo);
    atmosphere.mie_anisotropy = snapshot.mie_anisotropy;
    atmosphere.rayleigh_scale_height = snapshot.rayleigh_scale_height;
    atmosphere.mie_scale_height = snapshot.mie_scale_height;
    atmosphere.multi_scattering_factor = snapshot.multi_scattering_factor;
    atmosphere.exposure = snapshot.exposure;
    atmosphere.sun_disk_size = snapshot.sun_disk_size;
    atmosphere.sun_disk_intensity = snapshot.sun_disk_intensity;
    auto& celestial = result.celestial;
    celestial.sun_mode = to_scene(snapshot.sun_mode);
    celestial.time_mode = to_scene(snapshot.time_mode);
    celestial.latitude_degrees = snapshot.latitude_degrees;
    celestial.longitude_degrees = snapshot.longitude_degrees;
    celestial.north_offset_degrees = snapshot.north_offset_degrees;
    celestial.year = snapshot.year;
    celestial.month = snapshot.month;
    celestial.day = snapshot.day;
    celestial.local_time_hours = snapshot.local_time_hours;
    celestial.utc_offset_hours = snapshot.utc_offset_hours;
    celestial.playing = snapshot.playing;
    celestial.loop_day = snapshot.loop_day;
    celestial.time_scale = snapshot.time_scale;
    celestial.automatic_sun_light = snapshot.automatic_sun_light;
    celestial.sun_intensity_multiplier = snapshot.sun_intensity_multiplier;
    celestial.sun_temperature_multiplier = snapshot.sun_temperature_multiplier;
    celestial.moon_enabled = snapshot.moon_enabled;
    celestial.automatic_moon_phase = snapshot.automatic_moon_phase;
    celestial.moon_phase = snapshot.moon_phase;
    celestial.moon_intensity = snapshot.moon_intensity;
    celestial.moon_angular_radius_degrees = snapshot.moon_angular_radius_degrees;
    celestial.stars_enabled = snapshot.stars_enabled;
    celestial.star_density = snapshot.star_density;
    celestial.star_intensity = snapshot.star_intensity;
    celestial.star_twinkle = snapshot.star_twinkle;
    result.clouds.enabled = snapshot.clouds_enabled;
    result.clouds.cast_shadows = snapshot.cloud_shadows;
    result.clouds.cumulus = to_scene(snapshot.cumulus);
    result.clouds.cirrus = to_scene(snapshot.cirrus);
    result.fog.enabled = snapshot.fog_enabled;
    result.fog.color = to_scene(snapshot.fog_color);
    result.fog.density = snapshot.fog_density;
    result.fog.height_falloff = snapshot.fog_height_falloff;
    result.fog.start_distance = snapshot.fog_start_distance;
    result.fog.max_opacity = snapshot.fog_max_opacity;
    result.fog.sun_scattering_strength = snapshot.fog_sun_scattering;
    result.lighting.enabled = snapshot.lighting_enabled;
    result.lighting.source = to_scene(snapshot.lighting_source);
    result.lighting.constant_color = to_scene(snapshot.lighting_color);
    result.lighting.diffuse_intensity = snapshot.diffuse_intensity;
    result.lighting.specular_intensity = snapshot.specular_intensity;
    return result;
}

} // namespace arc::editor
