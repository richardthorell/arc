#include <arc/editor/viewport_render_stats.h>

#include <arc/editor/editor_state.h>
#include <arc/render/render.h>
#include <arc/scene/scene.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>

namespace arc::editor
{
namespace
{

bool entity_is_active(const ecs::world& world, ecs::entity entity) noexcept
{
    const auto* active = world.try_get<scene::active_component>(entity);
    return !active || active->active;
}

void add_mesh_stats(viewport_render_stats& stats, const render::renderer& renderer, render::mesh_handle mesh,
                    std::uint32_t instance_count = 1) noexcept
{
    if (!mesh.valid() || instance_count == 0) return;
    const auto* data = renderer.mesh_data_for(mesh);
    if (!data)
    {
        stats.vertices_complete = false;
        return;
    }

    const auto instances = static_cast<std::uint64_t>(instance_count);
    const auto triangle_count = data->indices.empty() ? data->vertices.size() / 3u : data->indices.size() / 3u;
    stats.triangles += static_cast<std::uint64_t>(triangle_count) * instances;
    stats.vertices += static_cast<std::uint64_t>(data->vertices.size()) * instances;
}

render::mesh_handle conventional_mesh_for(const editor_scene_state& state, ecs::entity entity,
                                          const scene::mesh_renderer_component& renderer_component) noexcept
{
    auto mesh = renderer_component.mesh.conventional;
    if (renderer_component.mesh.conventional_lod_count > 1) return mesh;

    const auto* lod = state.scene.try_get<scene::lod_component>(entity);
    if (!lod || !lod->enabled) return mesh;
    for (const auto& level : lod->levels)
    {
        if (level.mesh.valid()) return level.mesh;
    }
    return mesh;
}

} // namespace

viewport_render_stats collect_viewport_render_stats(const editor_scene_state& state,
                                                    const render::renderer& renderer) noexcept
{
    viewport_render_stats stats{};
    const bool virtual_geometry_enabled = renderer.resolved_config().features.virtual_geometry;

    state.scene.view<scene::transform_component, scene::mesh_renderer_component>().each(
        [&](ecs::entity entity, const scene::transform_component&, const scene::mesh_renderer_component& mesh_renderer)
        {
            if (!entity_is_active(state.scene, entity) || !mesh_renderer.visible) return;

            const bool uses_virtual_geometry =
                mesh_renderer.representation != render::geometry_representation_policy::conventional &&
                virtual_geometry_enabled && renderer.virtual_mesh_alive(mesh_renderer.mesh.virtualized);
            if (uses_virtual_geometry) return;

            add_mesh_stats(stats, renderer, conventional_mesh_for(state, entity, mesh_renderer));
        });

    state.scene.view<scene::transform_component, scene::skinned_mesh_renderer_component>().each(
        [&](ecs::entity entity, const scene::transform_component&,
            const scene::skinned_mesh_renderer_component& mesh_renderer)
        {
            if (!entity_is_active(state.scene, entity) || !mesh_renderer.visible) return;
            add_mesh_stats(stats, renderer, mesh_renderer.mesh);
        });

    state.scene.view<scene::transform_component, scene::instance_group_component>().each(
        [&](ecs::entity entity, const scene::transform_component&, const scene::instance_group_component& group)
        {
            if (!entity_is_active(state.scene, entity) || !group.visible) return;
            add_mesh_stats(stats, renderer, group.mesh, group.instance_count);
        });

    const auto profile = renderer.last_frame_profile();
    stats.triangles += profile.virtual_geometry.visible_triangles;
    stats.triangles += profile.terrain.rendered_triangles;
    if (profile.virtual_geometry.visible_triangles != 0 || profile.terrain.rendered_triangles != 0)
        stats.vertices_complete = false;

    if (const auto* backend = renderer.backend())
    {
        const auto& capabilities = backend->capabilities();
        stats.gpu_memory_used_bytes = capabilities.memory_usage;
        stats.gpu_memory_budget_bytes = capabilities.memory_budget;
        stats.gpu_memory_available = capabilities.memory_usage != 0;
    }

    return stats;
}

} // namespace arc::editor
