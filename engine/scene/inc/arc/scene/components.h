#pragma once

#include <arc/render/handles.h>
#include <arc/render/lighting.h>
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
    float exposure{ 1.0f };
    float sun_disk_size{ 0.025f };
    float sun_disk_intensity{ 1.4f };
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
