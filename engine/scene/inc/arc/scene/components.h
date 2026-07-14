#pragma once

#include <arc/render/handles.h>
#include <arc/render/lighting.h>
#include <arc/render/virtual_mesh.h>
#include <arc/scene/entity.h>
#include <arc/geometric/box.h>
#include <arc/math/math.h>

#include <string>
#include <vector>

namespace arc::scene
{

/**
 * @brief Human-readable entity name for editor and diagnostics.
 */
struct name_component
{
    std::string value;
};

/**
 * @brief Local/world transform for a scene entity.
 */
struct transform_component
{
    math::vector3f position{};
    math::quatf rotation{};
    math::vector3f scale{ 1.0f, 1.0f, 1.0f };
    math::matrix4f world{ math::identity<float, 4>() };
    bool dirty{ true };

    /**
     * @brief Mark the cached world matrix as stale.
     */
    void mark_dirty() noexcept
    {
        dirty = true;
    }

    /**
     * @brief Set local position and invalidate cached world data.
     */
    void set_position(const math::vector3f& value) noexcept
    {
        position = value;
        mark_dirty();
    }

    /**
     * @brief Set local rotation and invalidate cached world data.
     */
    void set_rotation(const math::quatf& value) noexcept
    {
        rotation = value;
        mark_dirty();
    }

    /**
     * @brief Set local scale and invalidate cached world data.
     */
    void set_scale(const math::vector3f& value) noexcept
    {
        scale = value;
        mark_dirty();
    }
};

/**
 * @brief Editor/game tag used for grouping or lightweight labels.
 */
struct tag_component
{
    std::string value;
};

/**
 * @brief Runtime active flag used to filter scene extraction and updates.
 */
struct active_component
{
    bool active{ true };
};

/**
 * @brief Editor selection state for an entity.
 */
struct selection_component
{
    bool selected{ true };
};

/**
 * @brief Local and world axis-aligned bounds for culling, picking, and editor display.
 */
struct bounds_component
{
    geometric::box3f local_bounds{};
    geometric::box3f world_bounds{};
    bool dirty{ true };
};

/**
 * @brief Camera projection type.
 */
enum class camera_projection
{
    perspective,
    orthographic
};

/**
 * @brief Camera component using right-handed +Y-up, -Z-forward scene space.
 */
struct camera_component
{
    camera_projection projection{ camera_projection::perspective };
    float fov_y_radians{ 1.0471975512f };
    float near_plane{ 0.01f };
    float far_plane{ 1000.0f };
    float orthographic_height{ 10.0f };
    bool active{ true };
    math::vector4f clear_color{ 0.10f, 0.22f, 0.34f, 1.0f };
};

/**
 * @brief Renderable static mesh component.
 */
struct mesh_renderer_component
{
    render::mesh_handle mesh{};
    render::material_handle material{};
    bool visible{ true };
    math::vector4f base_color_tint{ 1.0f, 1.0f, 1.0f, 1.0f };
};

/**
 * @brief Renderable virtual mesh component backed by CPU-visible clusters.
 */
struct virtual_mesh_renderer_component
{
    render::virtual_mesh_handle mesh{};
    render::material_handle material{};
    bool visible{ true };
    math::vector4f base_color_tint{ 1.0f, 1.0f, 1.0f, 1.0f };
};

/**
 * @brief Renderable skinned mesh component. Animation data can fill the skin palette later.
 */
struct skinned_mesh_renderer_component
{
    render::mesh_handle mesh{};
    render::material_handle material{};
    render::buffer_handle skin_matrices{};
    std::uint32_t joint_count{};
    bool visible{ true };
};

/**
 * @brief One mesh/material choice in a screen-size LOD chain.
 */
struct lod_level
{
    float screen_coverage{ 1.0f };
    render::mesh_handle mesh{};
    render::material_handle material{};
};

/**
 * @brief LOD settings for render extraction.
 */
struct lod_component
{
    std::vector<lod_level> levels;
    bool enabled{ true };
};

/**
 * @brief Simple instance group using one mesh/material pair.
 */
struct instance_group_component
{
    render::mesh_handle mesh{};
    render::material_handle material{};
    std::uint32_t instance_count{ 1 };
    bool visible{ true };
};

/**
 * @brief Scene render layer mask used by extraction and culling.
 */
struct render_layer_component
{
    std::uint32_t mask{ 1u };
};

/**
 * @brief Directional light. Direction is the owning transform's local -Z axis.
 */
struct directional_light_component
{
    math::vector3f color{ 1.0f, 1.0f, 1.0f };
    float intensity{ 1.0f };
    bool casts_shadows{ false };
    bool enabled{ true };
    bool use_color_temperature{};
    float temperature_kelvin{ 6500.0f };
    render::light_intensity_unit intensity_unit{ render::light_intensity_unit::unitless };
    render::texture_handle cookie_texture{};
    render::shadow_settings shadow{};
};

/**
 * @brief Omnidirectional point light.
 */
struct point_light_component
{
    math::vector3f color{ 1.0f, 1.0f, 1.0f };
    float intensity{ 1.0f };
    float range{ 10.0f };
    bool casts_shadows{ false };
    bool enabled{ true };
    bool use_color_temperature{};
    float temperature_kelvin{ 6500.0f };
    render::light_intensity_unit intensity_unit{ render::light_intensity_unit::unitless };
    render::texture_handle cookie_texture{};
    render::shadow_settings shadow{ .enabled = false };
};

/**
 * @brief Cone light. Direction is the owning transform's local -Z axis.
 */
struct spot_light_component
{
    math::vector3f color{ 1.0f, 1.0f, 1.0f };
    float intensity{ 1.0f };
    float range{ 10.0f };
    float inner_angle{ 0.35f };
    float outer_angle{ 0.75f };
    bool casts_shadows{ false };
    bool enabled{ true };
    bool use_color_temperature{};
    float temperature_kelvin{ 6500.0f };
    render::light_intensity_unit intensity_unit{ render::light_intensity_unit::unitless };
    render::texture_handle cookie_texture{};
    render::shadow_settings shadow{ .enabled = false };
};

/**
 * @brief Reflection probe scaffold for future local specular environment capture.
 */
struct reflection_probe_component
{
    float radius{ 5.0f };
    float intensity{ 1.0f };
    bool enabled{ true };
};

/**
 * @brief Irradiance probe scaffold for future diffuse global illumination.
 */
struct irradiance_probe_component
{
    float radius{ 5.0f };
    float intensity{ 1.0f };
    bool enabled{ true };
};

/**
 * @brief Source used to draw the background sky.
 */
enum class sky_source : std::uint8_t
{
    physical_atmosphere,
    hdri,
    solid_color
};

/**
 * @brief Source used for diffuse and specular environment lighting.
 */
enum class environment_lighting_source : std::uint8_t
{
    follow_sky,
    hdri,
    constant_color
};

/**
 * @brief Scene-wide world environment selection and visibility policy.
 */
struct world_environment_component
{
    bool enabled{ true };
    bool sky_visible{ true };
    bool affect_lighting{ true };
    sky_source source{ sky_source::physical_atmosphere };
    math::vector3f solid_color{ 0.08f, 0.13f, 0.22f };
    render::texture_handle hdri_texture{};
    float hdri_rotation_degrees{};
    float radiance_intensity{ 1.0f };
};

/**
 * @brief Procedural outdoor sky atmosphere settings.
 */
struct sky_atmosphere_component
{
    bool enabled{ true };
    float planet_radius{ 6360.0f };
    float atmosphere_radius{ 6420.0f };
    float rayleigh_strength{ 1.0f };
    float mie_strength{ 0.35f };
    float ozone_strength{ 0.15f };
    math::vector3f tint{ 0.56f, 0.72f, 1.0f };
    math::vector3f ground_albedo{ 0.18f, 0.18f, 0.18f };
    float mie_anisotropy{ 0.8f };
    float rayleigh_scale_height{ 8.0f };
    float mie_scale_height{ 1.2f };
    float multi_scattering_factor{ 1.0f };
    float exposure{ 1.0f };
    float sun_disk_size{ 0.025f };
    float sun_disk_intensity{ 1.4f };
};

enum class sun_position_mode : std::uint8_t
{
    manual_light,
    geographic
};

enum class celestial_time_mode : std::uint8_t
{
    fixed,
    simulated,
    system_clock
};

/**
 * @brief Geographic clock and celestial appearance for an outdoor sky.
 */
struct celestial_sky_component
{
    bool enabled{ true };
    sun_position_mode sun_mode{ sun_position_mode::manual_light };
    celestial_time_mode time_mode{ celestial_time_mode::fixed };
    entity sun_light{};
    float latitude_degrees{ 46.8f };
    float longitude_degrees{ 8.2f };
    float north_offset_degrees{};
    std::int32_t year{ 2026 };
    std::int32_t month{ 7 };
    std::int32_t day{ 14 };
    float local_time_hours{ 10.5f };
    float utc_offset_hours{ 2.0f };
    bool playing{};
    bool loop_day{ true };
    float time_scale{ 60.0f };
    float animation_time_seconds{};
    bool automatic_sun_light{ true };
    float sun_intensity_multiplier{ 1.0f };
    float sun_temperature_multiplier{ 1.0f };
    bool moon_enabled{ true };
    bool automatic_moon_phase{ true };
    float moon_phase{ 0.65f };
    float moon_intensity{ 0.22f };
    float moon_angular_radius_degrees{ 0.2725f };
    bool stars_enabled{ true };
    float star_density{ 0.42f };
    float star_intensity{ 0.75f };
    float star_twinkle{ 0.08f };
};

/**
 * @brief One artist-authored cloud deck.
 */
struct cloud_layer_settings
{
    bool enabled{ true };
    float coverage{ 0.28f };
    float density{ 0.58f };
    float altitude{ 1800.0f };
    float thickness{ 450.0f };
    float scale{ 0.00045f };
    float detail{ 0.55f };
    float softness{ 0.18f };
    math::vector2f wind_direction{ 0.8f, 0.35f };
    float wind_speed{ 8.0f };
    float lighting_strength{ 1.0f };
    float silver_lining{ 0.35f };
};

/**
 * @brief Scalable lower and upper procedural cloud decks.
 */
struct cloud_layers_component
{
    bool enabled{ true };
    bool cast_shadows{ true };
    cloud_layer_settings cumulus{};
    cloud_layer_settings cirrus{
        .coverage = 0.12f,
        .density = 0.28f,
        .altitude = 6200.0f,
        .thickness = 180.0f,
        .scale = 0.00018f,
        .detail = 0.72f,
        .softness = 0.32f,
        .wind_direction = { -0.4f, 0.9f },
        .wind_speed = 18.0f,
        .lighting_strength = 0.75f,
        .silver_lining = 0.12f
    };
};

/**
 * @brief Ambient environment lighting source independent of sky visibility.
 */
struct environment_lighting_component
{
    bool enabled{ true };
    environment_lighting_source source{ environment_lighting_source::follow_sky };
    render::environment_handle environment{};
    render::texture_handle hdri_texture{};
    math::vector3f constant_color{ 0.18f, 0.23f, 0.29f };
    float diffuse_intensity{ 1.0f };
    float specular_intensity{ 1.0f };
};

/**
 * @brief Distance and altitude based fog settings.
 */
struct height_fog_component
{
    bool enabled{ true };
    math::vector3f color{ 0.58f, 0.67f, 0.76f };
    float density{ 0.035f };
    float height_falloff{ 0.12f };
    float start_distance{ 8.0f };
    float max_opacity{ 0.55f };
    float sun_scattering_strength{ 0.25f };
};

/**
 * @brief Generated terrain surface settings for the editor/world foundation.
 */
struct terrain_component
{
    bool enabled{ true };
    float size{ 32.0f };
    std::uint32_t subdivisions{ 64 };
    float height_scale{ 1.45f };
    math::vector3f base_color{ 1.0f, 1.0f, 1.0f };
    render::material_handle material{};
    bool receive_shadows{ true };
};

/**
 * @brief Flat water body settings for the first outdoor scene pass.
 */
struct water_component
{
    bool enabled{ true };
    float size{ 8.0f };
    math::vector3f color{ 0.16f, 0.35f, 0.48f };
    float roughness{ 0.18f };
    float wave_scale{ 0.08f };
    float wave_speed{ 0.45f };
    float transparency{ 0.45f };
};

/**
 * @brief Simple generated vegetation patch settings.
 */
struct vegetation_component
{
    bool enabled{ true };
    std::uint32_t density{ 96 };
    float patch_size{ 8.0f };
    math::vector3f color{ 0.22f, 0.46f, 0.18f };
    float wind_strength{ 0.25f };
    float wind_speed{ 0.8f };
};

/**
 * @brief Decal projection scaffold for editor-authored surface detail.
 */
struct decal_component
{
    bool enabled{ true };
    geometric::box3f local_bounds{
        geometric::point3f{ -0.5f, -0.5f, -0.5f },
        geometric::point3f{ 0.5f, 0.5f, 0.5f }
    };
    math::vector4f color{ 1.0f, 1.0f, 1.0f, 0.75f };
    render::texture_handle texture{};
    float opacity{ 0.75f };
};

} // namespace arc::scene
