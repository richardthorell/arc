#pragma once

#include <arc/render/events.h>
#include <arc/render/handles.h>
#include <arc/render/material.h>
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
 * @brief Stable editor/game object id carried into picking and outline passes.
 */
struct render_object_id
{
    std::uint32_t index{ resource_handle::invalid_index };
    std::uint32_t generation{};

    constexpr bool valid() const noexcept
    {
        return index != resource_handle::invalid_index;
    }
};

/**
 * @brief One camera extracted for scene rendering.
 */
struct render_camera
{
    math::matrix4f view_projection{ math::identity<float, 4>() };
    math::vector3f position{};
    math::vector4f clear_color{ 0.118f, 0.118f, 0.118f, 1.0f };
    float near_plane{ 0.01f };
    float far_plane{ 1000.0f };
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
    std::vector<directional_light_event> directional_lights;
    std::vector<point_light_event> point_lights;
    std::vector<spot_light_event> spot_lights;
    std::vector<reflection_probe_data> reflection_probes;
    std::vector<irradiance_probe_data> irradiance_probes;
    std::vector<render_item> items;
    std::vector<std::uint32_t> visible_items;
    std::vector<render_instance_batch> instance_batches;
    std::vector<indirect_draw_command> indirect_draws;
    std::uint32_t viewport_width{};
    std::uint32_t viewport_height{};
    std::size_t culled_item_count{};
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
render_graph make_scene_draw_graph(std::string_view target_name);

} // namespace arc::render
