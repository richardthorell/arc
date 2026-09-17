#include <arc/editor/editor_gizmo.h>
#include <arc/editor/editor_interaction.h>
#include <arc/editor/editor_state.h>
#include <arc/editor/viewport_render_stats.h>
#include <arc/geometric/box.h>
#include <arc/scene/scene.h>

#include <algorithm>
#include <array>
#include <limits>
#include <string>
#include <string_view>
#include <unordered_map>
#include <nlohmann/json.hpp>

namespace arc::editor
{
namespace
{
constexpr std::string_view arc_material_preview_scene_name = "Asset Preview: material";

struct arc_material_preview_panel
{
    const char* name;
    math::vector3f position;
    math::vector3f scale;
};

std::unordered_map<editor_scene_state*, render::mesh_handle> arc_material_preview_room_meshes;

void arc_configure_material_preview_panel(editor_scene_state& state, ecs::entity entity,
                                          const arc_material_preview_panel& panel)
{
    if (auto* name = state.scene.try_get<scene::name_component>(entity)) name->value = panel.name;
    if (auto* tag = state.scene.try_get<scene::tag_component>(entity)) tag->value = "Environment";
    if (auto* selection = state.scene.try_get<scene::selection_component>(entity)) selection->selected = false;
    if (auto* transform = state.scene.try_get<scene::transform_component>(entity))
    {
        transform->set_position(panel.position);
        transform->set_scale(panel.scale);
    }
}

ecs::entity arc_duplicate_material_preview_panel(editor_scene_state& state, ecs::entity source,
                                                 const arc_material_preview_panel& panel)
{
    const auto* source_bounds = state.scene.try_get<scene::bounds_component>(source);
    const auto* source_renderer = state.scene.try_get<scene::mesh_renderer_component>(source);
    if (!source_bounds || !source_renderer) return {};

    const auto entity = state.scene.create();
    scene::transform_component transform;
    transform.set_position(panel.position);
    transform.set_scale(panel.scale);
    state.scene.emplace<scene::name_component>(entity, panel.name);
    state.scene.emplace<scene::tag_component>(entity, "Environment");
    state.scene.emplace<scene::active_component>(entity);
    state.scene.emplace<scene::selection_component>(entity, false);
    state.scene.emplace<scene::bounds_component>(entity, source_bounds->local_bounds, source_bounds->local_bounds,
                                                 true);
    state.scene.emplace<scene::transform_component>(entity, transform);
    state.scene.emplace<scene::mesh_renderer_component>(entity, *source_renderer);
    state.scene.emplace<scene::persistent_id_component>(entity, ecs::generate_entity_guid());
    state.scene.emplace<scene::hierarchy_component>(entity);
    state.primitive_entities.push_back(entity);
    return entity;
}

ecs::entity arc_material_preview_add_primitive(editor_scene_state& state, render::renderer& renderer,
                                               editor_primitive_type type)
{
    const auto entity = add_primitive_to_scene(state, renderer, type);
    if (type != editor_primitive_type::sphere || state.scene_name != arc_material_preview_scene_name ||
        !state.scene.alive(entity) || arc_material_preview_room_meshes.contains(&state))
        return entity;

    // The material sphere has a 0.5-unit radius. Keep its center at the orbit
    // pivot and place the studio floor at y=-0.5 so it physically rests on it.
    constexpr float half_extent = 4.0f;
    constexpr float room_height = 5.5f;
    constexpr float panel_thickness = 0.10f;
    constexpr float floor_surface_y = -0.5f;
    constexpr float wall_center_y = floor_surface_y + room_height * 0.5f;
    constexpr float ceiling_center_y = floor_surface_y + room_height + panel_thickness * 0.5f;

    const auto room_template = add_primitive_to_scene(state, renderer, editor_primitive_type::cube);
    if (!state.scene.alive(room_template)) return entity;
    const auto* room_renderer = state.scene.try_get<scene::mesh_renderer_component>(room_template);
    if (!room_renderer || !room_renderer->mesh.valid()) return entity;
    arc_material_preview_room_meshes[&state] = room_renderer->mesh;

    constexpr std::array<arc_material_preview_panel, 5> panels{{
        {"Material Preview Floor",
         {0.0f, floor_surface_y - panel_thickness * 0.5f, 0.0f},
         {half_extent * 2.0f, panel_thickness, half_extent * 2.0f}},
        {"Material Preview Back Wall",
         {0.0f, wall_center_y, -half_extent},
         {half_extent * 2.0f, room_height, panel_thickness}},
        {"Material Preview Left Wall",
         {-half_extent, wall_center_y, 0.0f},
         {panel_thickness, room_height, half_extent * 2.0f}},
        {"Material Preview Right Wall",
         {half_extent, wall_center_y, 0.0f},
         {panel_thickness, room_height, half_extent * 2.0f}},
        {"Material Preview Ceiling",
         {0.0f, ceiling_center_y, 0.0f},
         {half_extent * 2.0f, panel_thickness, half_extent * 2.0f}},
    }};

    arc_configure_material_preview_panel(state, room_template, panels.front());
    for (std::size_t index = 1; index < panels.size(); ++index)
        (void)arc_duplicate_material_preview_panel(state, room_template, panels[index]);
    return entity;
}

void arc_clear_preview_imported_content(editor_scene_state& state, render::renderer& renderer)
{
    render::mesh_handle room_mesh{};
    if (const auto found = arc_material_preview_room_meshes.find(&state);
        found != arc_material_preview_room_meshes.end())
    {
        room_mesh = found->second;
        arc_material_preview_room_meshes.erase(found);
    }

    clear_imported_scene_content(state, renderer);
    if (room_mesh.valid() && renderer.mesh_alive(room_mesh)) (void)renderer.destroy_mesh(room_mesh);
}

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

#define add_primitive_to_scene(state, renderer, type) arc_material_preview_add_primitive(state, renderer, type)
#define clear_imported_scene_content(state, renderer) arc_clear_preview_imported_content(state, renderer)
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
#if defined(_MSC_VER)
#pragma warning(push)
#pragma warning(disable : 4456)
#endif
#include "arc_host_impl.inc"
#if defined(_MSC_VER)
#pragma warning(pop)
#endif
#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC diagnostic pop
#endif

#undef collect_viewport_render_stats
#undef focus_selected_entity
#undef clear_imported_scene_content
#undef add_primitive_to_scene