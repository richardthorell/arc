#pragma once

#include <arc/render/events.h>
#include <arc/render/handles.h>
#include <arc/render/material.h>
#include <arc/render/render_backend.h>
#include <arc/render/render_graph.h>
#include <arc/geometric/box.h>
#include <arc/math/matrix.h>
#include <arc/math/vector.h>

#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

namespace arc::render
{

/**
 * @brief Render queue/pass bucket used by scene draw preparation.
 */
enum class scene_render_pass : std::uint8_t
{
    depth_prepass,
    gbuffer,
    forward_transparent,
    editor_picking,
    selection_outline
};

/**
 * @brief One camera extracted for scene rendering.
 */
struct render_camera
{
    math::matrix4f view{ math::identity<float, 4>() };
    math::matrix4f projection{ math::identity<float, 4>() };
    math::matrix4f view_projection{ math::identity<float, 4>() };
    math::matrix4f previous_view_projection{ math::identity<float, 4>() };
    math::matrix4f inverse_view_projection{ math::identity<float, 4>() };
    math::vector2f jitter{};
    math::vector3f position{};
    math::vector3f forward{ 0.0f, 0.0f, -1.0f };
    math::vector3f up{ 0.0f, 1.0f, 0.0f };
    math::vector4f clear_color{ 0.118f, 0.118f, 0.118f, 1.0f };
    float near_plane{ 0.01f };
    float far_plane{ 1000.0f };
    std::uint32_t render_width{};
    std::uint32_t render_height{};
    std::uint32_t output_width{};
    std::uint32_t output_height{};
};

/**
 * @brief Optional GPU skin palette reference for skinned draws.
 */
struct render_skin
{
    buffer_handle joint_matrices{};
    std::uint32_t joint_count{};
};

/**
 * @brief One extracted scene draw candidate before backend command generation.
 */
struct render_item
{
    mesh_handle mesh{};
    material_handle material{};
    std::uint32_t submesh{};
    math::matrix4f model{ math::identity<float, 4>() };
    math::matrix4f previous_model{ math::identity<float, 4>() };
    geometric::box3f world_bounds{};
    std::uint32_t render_layer_mask{ 1u };
    std::uint64_t sort_key{};
    render_object_id object_id{};
    buffer_handle skin_matrices{};
    std::uint32_t skin_joint_count{};
    std::uint32_t instance_start{};
    std::uint32_t instance_count{ 1 };
    bool visible{ true };
    bool selected{};
    bool transparent{};
    bool casts_shadows{ true };
    math::vector4f base_color_tint{ 1.0f, 1.0f, 1.0f, 1.0f };
    std::string label;
};

/**
 * @brief One CPU-visible virtual mesh cluster draw candidate before backend command generation.
 */
struct virtual_render_item
{
    virtual_mesh_handle mesh{};
    material_handle material{};
    std::uint32_t cluster_index{};
    math::matrix4f model{ math::identity<float, 4>() };
    math::matrix4f previous_model{ math::identity<float, 4>() };
    geometric::box3f world_bounds{};
    std::uint32_t render_layer_mask{ 1u };
    std::uint64_t sort_key{};
    render_object_id object_id{};
    bool visible{ true };
    bool selected{};
    bool casts_shadows{ true };
    math::vector4f base_color_tint{ 1.0f, 1.0f, 1.0f, 1.0f };
    std::string label;
};

/**
 * @brief Reflection probe extracted for future local specular environment lighting.
 */
struct reflection_probe_data
{
    math::vector3f position{};
    float radius{ 5.0f };
    float intensity{ 1.0f };
    std::string label;
};

/**
 * @brief Irradiance probe extracted for future diffuse environment lighting.
 */
struct irradiance_probe_data
{
    math::vector3f position{};
    float radius{ 5.0f };
    float intensity{ 1.0f };
    std::string label;
};

enum class sky_source_mode : std::uint8_t
{
    physical_atmosphere,
    hdri,
    solid_color
};

enum class environment_lighting_source_mode : std::uint8_t
{
    follow_sky,
    hdri,
    constant_color
};

/** @brief Procedural atmosphere parameters extracted for one frame. */
struct sky_atmosphere_data
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
    std::string label;
};

struct celestial_sky_data
{
    bool enabled{ true };
    math::vector3f sun_direction{ 0.35f, -0.85f, -0.40f };
    math::vector3f moon_direction{ -0.35f, 0.85f, 0.40f };
    float sun_angular_radius_degrees{ 0.2666f };
    float sun_intensity{ 1.0f };
    bool moon_enabled{ true };
    float moon_phase{ 0.65f };
    float moon_intensity{ 0.22f };
    float moon_angular_radius_degrees{ 0.2725f };
    bool stars_enabled{ true };
    float star_density{ 0.42f };
    float star_intensity{ 0.75f };
    float star_twinkle{ 0.08f };
    float time_seconds{};
};

struct cloud_layer_data
{
    bool enabled{};
    float coverage{};
    float density{};
    float altitude{};
    float thickness{};
    float scale{};
    float detail{};
    float softness{};
    math::vector2f wind_direction{};
    float wind_speed{};
    float lighting_strength{};
    float silver_lining{};
};

struct cloud_layers_data
{
    bool enabled{};
    bool cast_shadows{};
    cloud_layer_data cumulus;
    cloud_layer_data cirrus;
};

struct environment_lighting_data
{
    bool enabled{};
    environment_lighting_source_mode source{ environment_lighting_source_mode::follow_sky };
    environment_handle environment{};
    texture_handle hdri_texture{};
    math::vector3f constant_color{ 0.12f, 0.12f, 0.12f };
    float diffuse_intensity{ 1.0f };
    float specular_intensity{ 1.0f };
};

/**
 * @brief Height fog resolved with the active world environment.
 */
struct height_fog_data
{
    bool enabled{};
    math::vector3f color{ 0.58f, 0.67f, 0.76f };
    float density{ 0.035f };
    float height_falloff{ 0.12f };
    float start_distance{ 8.0f };
    float max_opacity{ 0.55f };
    float sun_scattering_strength{ 0.25f };
    std::string label;
};

/** @brief Immutable resolved world-environment state for one frame. */
struct world_environment_data
{
    bool enabled{};
    bool sky_visible{};
    bool affect_lighting{};
    sky_source_mode source{ sky_source_mode::physical_atmosphere };
    math::vector3f solid_color{ 0.08f, 0.13f, 0.22f };
    texture_handle hdri_texture{};
    float hdri_rotation_degrees{};
    float radiance_intensity{ 1.0f };
    sky_atmosphere_data atmosphere;
    celestial_sky_data celestial;
    cloud_layers_data clouds;
    height_fog_data fog;
    environment_lighting_data lighting;
    std::string label;
    std::string fallback_reason;
};

/**
 * @brief Extracted terrain metadata for editor/render diagnostics.
 */
struct terrain_render_data
{
    render_object_id object_id{};
    math::vector3f position{};
    float size{ 1.0f };
    std::uint32_t subdivisions{};
    float height_scale{};
    bool receive_shadows{ true };
    std::string label;
};

/**
 * @brief Extracted water metadata for simple animated water rendering.
 */
struct water_render_data
{
    render_object_id object_id{};
    math::vector3f position{};
    float size{ 1.0f };
    math::vector3f color{ 0.16f, 0.35f, 0.48f };
    float roughness{ 0.18f };
    float wave_scale{ 0.08f };
    float wave_speed{ 0.45f };
    float transparency{ 0.45f };
    std::string label;
};

/**
 * @brief Extracted vegetation metadata for simple generated foliage.
 */
struct vegetation_render_data
{
    render_object_id object_id{};
    math::vector3f position{};
    std::uint32_t density{};
    float patch_size{ 1.0f };
    math::vector3f color{ 0.22f, 0.46f, 0.18f };
    float wind_strength{ 0.25f };
    float wind_speed{ 0.8f };
    std::string label;
};

/**
 * @brief Extracted decal metadata for projected decal scaffolding.
 */
struct decal_render_data
{
    render_object_id object_id{};
    math::matrix4f model{ math::identity<float, 4>() };
    geometric::box3f world_bounds{};
    math::vector4f color{ 1.0f, 1.0f, 1.0f, 0.75f };
    texture_handle texture{};
    float opacity{ 0.75f };
    std::string label;
};

/**
 * @brief Batch of adjacent render items that can be instanced together.
 */
struct render_instance_batch
{
    mesh_handle mesh{};
    material_handle material{};
    scene_render_pass pass{ scene_render_pass::gbuffer };
    std::uint32_t first_item{};
    std::uint32_t item_count{};
    std::uint64_t sort_key{};
};

/**
 * @brief One indirect indexed draw command in backend-neutral form.
 */
struct indirect_draw_command
{
    std::uint32_t index_count{};
    std::uint32_t instance_count{ 1 };
    std::uint32_t first_index{};
    std::int32_t vertex_offset{};
    std::uint32_t first_instance{};
};

/**
 * @brief Plane used by CPU frustum culling.
 */
struct frustum_plane
{
    math::vector3f normal{};
    float distance{};
};

/**
 * @brief View frustum extracted from a view-projection matrix.
 */
struct view_frustum
{
    frustum_plane planes[6]{};
};

/**
 * @brief Render world packet consumed by backends for one scene frame.
 */
struct render_world_packet
{
    render_camera camera;
    render_mode mode{ render_mode::shaded };
    mesh_visualization_mode visualization{ mesh_visualization_mode::standard };
    editor_overlay_mode overlay{ editor_overlay_mode::selected_wireframe };
    bool shadows_enabled{ true };
    std::vector<directional_light_event> directional_lights;
    std::vector<point_light_event> point_lights;
    std::vector<spot_light_event> spot_lights;
    std::vector<reflection_probe_data> reflection_probes;
    std::vector<irradiance_probe_data> irradiance_probes;
    world_environment_data environment;
    std::vector<terrain_render_data> terrains;
    std::vector<water_render_data> waters;
    std::vector<vegetation_render_data> vegetation;
    std::vector<decal_render_data> decals;
    std::vector<render_item> items;
    std::vector<std::uint32_t> visible_items;
    std::vector<virtual_render_item> virtual_items;
    std::vector<std::uint32_t> visible_virtual_items;
    std::vector<render_instance_batch> instance_batches;
    std::vector<indirect_draw_command> indirect_draws;
    std::uint32_t viewport_width{};
    std::uint32_t viewport_height{};
    std::size_t culled_item_count{};
    std::size_t culled_virtual_cluster_count{};
};

/**
 * @brief Options controlling render-world preparation.
 */
struct render_world_prepare_options
{
    bool enable_frustum_culling{ true };
    bool enable_instancing{ true };
    bool enable_indirect_draws{ true };
    std::uint32_t render_layer_mask{ 0xffffffffu };
};

/**
 * @brief Build a stable object id from an index/generation pair.
 */
constexpr render_object_id make_render_object_id(std::uint32_t index, std::uint32_t generation) noexcept
{
    return { .index = index, .generation = generation };
}

/**
 * @brief Build a frustum from a view-projection matrix.
 */
view_frustum make_view_frustum(const math::matrix4f& view_projection);

/**
 * @brief Return whether an axis-aligned box intersects a frustum.
 */
bool intersects(const view_frustum& frustum, const geometric::box3f& bounds);

/**
 * @brief Build a material/mesh/depth sort key.
 */
std::uint64_t make_render_sort_key(scene_render_pass pass, material_handle material, mesh_handle mesh, float depth);

/**
 * @brief Cull, sort, batch, and generate backend-neutral indirect draw commands.
 */
void prepare_render_world(render_world_packet& packet, const render_world_prepare_options& options = {});

/**
 * @brief Create the standard scene draw graph for viewport rendering.
 */
render_graph make_scene_draw_graph(
    std::string_view target_name,
    render_path path = render_path::deferred,
    bool editor_view = true);

render_graph make_scene_draw_graph(
    std::string_view target_name,
    const resolved_render_config& config,
    bool editor_view = true);

render_graph make_scene_draw_graph(
    std::string_view target_name,
    const resolved_render_config& config,
    bool editor_view,
    const world_environment_data& environment);

} // namespace arc::render
