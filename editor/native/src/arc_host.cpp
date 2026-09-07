#include <arc/editor/editor_gizmo.h>
#include <arc/editor/editor_interaction.h>
#include <arc/editor/editor_state.h>
#include <arc/editor/viewport_render_stats.h>
#include <arc/geometric/box.h>
#include <arc/scene/scene.h>

#include <algorithm>
#include <limits>
#include <string_view>
#include <nlohmann/json.hpp>

namespace arc::editor
{
namespace
{
bool arc_model_preview_focus(const ecs::world& registry, ecs::entity selected,
                             editor_camera_controller& camera) noexcept
{
    math::vector3f minimum{std::numeric_limits<float>::max(), std::numeric_limits<float>::max(),
                           std::numeric_limits<float>::max()};
    math::vector3f maximum{std::numeric_limits<float>::lowest(), std::numeric_limits<float>::lowest(),
                           std::numeric_limits<float>::lowest()};
    bool found{};

    registry.view<scene::transform_component, scene::bounds_component>().each(
        [&](ecs::entity entity, const scene::transform_component& transform, const scene::bounds_component& bounds)
        {
            if (!registry.has<scene::mesh_renderer_component>(entity) &&
                !registry.has<scene::skinned_mesh_renderer_component>(entity))
                return;
            const auto world = transformed_bounds(bounds.local_bounds, transform);
            for (std::size_t axis = 0; axis < 3; ++axis)
            {
                minimum[axis] = std::min(minimum[axis], world.min[axis]);
                maximum[axis] = std::max(maximum[axis], world.max[axis]);
            }
            found = true;
        });

    if (!found) return focus_selected_entity(registry, selected, camera);
    const auto center = math::mul(math::add(minimum, maximum), 0.5f);
    const float radius = std::max(0.1f, math::length(math::sub(maximum, minimum)) * 0.5f);
    camera.focus(center, radius);
    return true;
}

viewport_render_stats arc_model_preview_render_stats(const editor_scene_state& scene,
                                                     const render::renderer& renderer) noexcept
{
    return collect_viewport_render_stats(scene, renderer);
}

void arc_append_model_preview_metadata(nlohmann::json& payload, const editor_scene_state& model_scene)
{
    nlohmann::json meshes = nlohmann::json::array();
    for (const auto entity : model_scene.imported_scene_entities)
    {
        const bool static_mesh = model_scene.scene.has<scene::mesh_renderer_component>(entity);
        const bool skinned_mesh = model_scene.scene.has<scene::skinned_mesh_renderer_component>(entity);
        if (!static_mesh && !skinned_mesh) continue;

        std::string name = "Mesh";
        if (const auto* named = model_scene.scene.try_get<scene::name_component>(entity);
            named && !named->value.empty())
            name = named->value;
        meshes.push_back({{"name", name}, {"skinned", skinned_mesh}});
    }
    payload["modelMeshes"] = std::move(meshes);

    if (model_scene.imported_skeletons.empty()) return;
    const auto& skeleton = model_scene.imported_skeletons.front().skeleton;
    nlohmann::json joints = nlohmann::json::array();
    std::uint32_t hierarchy_depth{};
    for (std::size_t index = 0; index < skeleton.joints.size(); ++index)
    {
        const auto& joint = skeleton.joints[index];
        joints.push_back({{"index", index}, {"name", joint.name}, {"parent", joint.parent}});
        std::uint32_t depth = 1;
        auto parent = joint.parent;
        std::size_t guard{};
        while (parent >= 0 && static_cast<std::size_t>(parent) < skeleton.joints.size() &&
               guard++ < skeleton.joints.size())
        {
            ++depth;
            parent = skeleton.joints[static_cast<std::size_t>(parent)].parent;
        }
        hierarchy_depth = std::max(hierarchy_depth, depth);
    }

    nlohmann::json skeleton_json = {{"name", skeleton.name},
                                    {"boneCount", skeleton.joints.size()},
                                    {"hierarchyDepth", hierarchy_depth},
                                    {"joints", std::move(joints)}};
    if (skeleton.root_joint < skeleton.joints.size())
        skeleton_json["rootBone"] = skeleton.joints[skeleton.root_joint].name;
    payload["modelSkeleton"] = std::move(skeleton_json);
}
} // namespace
} // namespace arc::editor

#define focus_selected_entity(registry, selected, camera) arc_model_preview_focus(registry, selected, camera)
#define collect_viewport_render_stats(scene, renderer)                                                                 \
    (                                                                                                                  \
        [&]()                                                                                                          \
        {                                                                                                              \
            if (viewport_surface && viewport_surface->preview_kind == asset_preview_kind::model &&                     \
                viewport_surface->preview_scene)                                                                       \
                arc_append_model_preview_metadata(payload, *viewport_surface->preview_scene);                          \
            return arc_model_preview_render_stats(scene, renderer);                                                    \
        }())

#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wsubobject-linkage"
#endif
#include "arc_host_impl.inc"
#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC diagnostic pop
#endif

#undef collect_viewport_render_stats
#undef focus_selected_entity