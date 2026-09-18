#include <arc/editor/editor_gizmo.h>
#include <arc/editor/editor_interaction.h>
#include <arc/editor/editor_state.h>
#include <arc/editor/viewport_render_stats.h>
#include <arc/diagnostics/log.h>
#include <arc/geometric/box.h>
#include <arc/render/render.h>
#include <arc/scene/scene.h>

#include <algorithm>
#include <filesystem>
#include <limits>
#include <optional>
#include <string>
#include <string_view>
#include <unordered_map>
#include <nlohmann/json.hpp>

namespace arc::editor
{
namespace
{
constexpr std::string_view arc_material_preview_scene_name = "Asset Preview: material";
constexpr std::string_view arc_material_preview_environment_name = "material_preview_studio_4k.exr";

struct arc_material_preview_resources
{
    render::texture_handle environment_texture{};
    render::environment_handle environment{};
};

std::unordered_map<editor_scene_state*, arc_material_preview_resources> arc_material_preview_scene_resources;

std::optional<std::filesystem::path>
arc_material_preview_environment_path(const editor_asset_state& editor_assets)
{
    const auto relative = std::filesystem::path{"assets"} / "environments" / arc_material_preview_environment_name;
    const auto available = [](const std::filesystem::path& candidate) -> std::optional<std::filesystem::path>
    {
        std::error_code error;
        if (!std::filesystem::is_regular_file(candidate, error) || error) return std::nullopt;
        return candidate.lexically_normal();
    };

    // Built-in roots are the authoritative location in installed/editor builds.
    // Source-tree probing remains only as a development fallback.
    for (const auto& builtin_root : editor_assets.builtin_roots)
        if (const auto path = available(builtin_root / "environments" / arc_material_preview_environment_name))
            return path;
    if (!editor_assets.root.empty())
        if (const auto path = available(editor_assets.root / "environments" / arc_material_preview_environment_name))
            return path;

    std::filesystem::path source_path{__FILE__};
    if (source_path.is_absolute())
    {
        auto source_root = source_path;
        for (int depth = 0; depth < 4; ++depth)
            source_root = source_root.parent_path();
        if (const auto path = available(source_root / relative)) return path;
    }

    std::error_code error;
    auto current = std::filesystem::current_path(error);
    if (error) return std::nullopt;
    for (int depth = 0; depth < 8; ++depth)
    {
        if (const auto path = available(current / relative)) return path;
        const auto parent = current.parent_path();
        if (parent.empty() || parent == current) break;
        current = parent;
    }
    return std::nullopt;
}

bool arc_configure_material_preview_environment(editor_scene_state& state, render::renderer& renderer,
                                                const editor_asset_state& editor_assets,
                                                arc_material_preview_resources& resources)
{
    if (resources.environment_texture.valid()) return true;

    const auto path = arc_material_preview_environment_path(editor_assets);
    if (!path)
    {
        arc::diagnostics::warn("editor.materials", "Material preview HDRI is unavailable; using the neutral fallback");
        return false;
    }

    auto loaded = render::load_texture_asset(*path);
    if (!loaded.succeeded())
    {
        arc::diagnostics::warn("editor.materials", "Material preview HDRI failed to decode: " + loaded.message);
        return false;
    }
    loaded.texture.name = "Material Preview Studio HDRI";
    const auto texture = renderer.create_texture(std::move(loaded.texture));
    if (!texture.valid())
    {
        arc::diagnostics::warn("editor.materials", "Material preview HDRI could not be uploaded");
        return false;
    }

    const auto environment_entity = add_world_environment_to_scene(state);
    auto settings = scene::read_world_environment_settings(state.scene, environment_entity);
    if (!settings)
    {
        (void)renderer.destroy_texture(texture);
        arc::diagnostics::warn("editor.materials", "Material preview world environment is incomplete");
        return false;
    }

    // Material preview uses the EXR directly as its visible equirectangular
    // world. Disable the analytic sky decorations so clouds/sun/stars are not
    // composited on top of the authored HDRI.
    settings->world.enabled = true;
    settings->world.sky_visible = true;
    settings->world.affect_lighting = true;
    settings->world.source = scene::sky_source::hdri;
    settings->world.hdri_texture = texture;
    settings->world.hdri_rotation_degrees = 0.0f;
    settings->world.radiance_intensity = 1.0f;
    settings->atmosphere.exposure = 1.0f;
    settings->atmosphere.sun_disk_intensity = 0.0f;
    settings->celestial.stars_enabled = false;
    settings->celestial.moon_enabled = false;
    settings->clouds.enabled = false;
    settings->fog.enabled = false;
    settings->lighting.enabled = true;
    settings->lighting.source = scene::environment_lighting_source::hdri;
    settings->lighting.hdri_texture = texture;

    render::environment_descriptor environment;
    environment.name = "Material Preview Studio Environment";
    environment.equirectangular_texture = texture;
    environment.fallback_color = settings->world.solid_color;
    environment.intensity = settings->world.radiance_intensity;
    environment.diffuse_irradiance = settings->lighting.constant_color;
    environment.diffuse_intensity = settings->lighting.diffuse_intensity;
    const auto environment_handle = renderer.create_environment(std::move(environment));
    if (environment_handle.valid())
    {
        settings->lighting.environment = environment_handle;
        state.environment_lighting_resource = environment_handle;
    }

    if (!scene::set_world_environment_settings(state.scene, environment_entity, *settings))
    {
        if (environment_handle.valid()) (void)renderer.destroy_environment(environment_handle);
        (void)renderer.destroy_texture(texture);
        state.environment_lighting_resource = {};
        arc::diagnostics::warn("editor.materials", "Material preview HDRI environment failed validation");
        return false;
    }

    // The preview is intentionally HDRI-lit; do not retain the default outdoor
    // sun that create_blank_scene installs for normal editor scenes.
    if (auto* sun = state.scene.try_get<scene::directional_light_component>(state.sun_entity)) sun->enabled = false;

    resources.environment_texture = texture;
    resources.environment = environment_handle;
    state.world_environment_hdri_path = std::filesystem::path{"environments"} / arc_material_preview_environment_name;
    arc::diagnostics::info("editor.materials",
                           "Material preview HDRI skybox active: " + path->generic_string());
    return true;
}

editor_primitive_type arc_material_preview_primitive_type(std::string_view viewport_id)
{
    const auto selector = viewport_id.find('~');
    if (selector == std::string_view::npos) return editor_primitive_type::sphere;

    const auto token = viewport_id.substr(selector + 1);
    if (token.starts_with("cube")) return editor_primitive_type::cube;
    if (token.starts_with("pill")) return editor_primitive_type::capsule;
    return editor_primitive_type::sphere;
}

ecs::entity arc_material_preview_add_primitive(editor_scene_state& state, render::renderer& renderer,
                                               editor_primitive_type type, const editor_asset_state& editor_assets)
{
    const auto entity = add_primitive_to_scene(state, renderer, type);
    if (state.scene_name != arc_material_preview_scene_name || !state.scene.alive(entity) ||
        arc_material_preview_scene_resources.contains(&state))
        return entity;

    // The EXR is the preview world itself. Do not add floor/wall helper meshes:
    // the material object should be surrounded directly by the equirectangular
    // environment, matching a normal HDRI material-preview viewport.
    auto& resources = arc_material_preview_scene_resources[&state];
    (void)arc_configure_material_preview_environment(state, renderer, editor_assets, resources);
    return entity;
}

void arc_clear_preview_imported_content(editor_scene_state& state, render::renderer& renderer)
{
    arc_material_preview_resources resources{};
    if (const auto found = arc_material_preview_scene_resources.find(&state);
        found != arc_material_preview_scene_resources.end())
    {
        resources = found->second;
        arc_material_preview_scene_resources.erase(found);
    }

    clear_imported_scene_content(state, renderer);
    if (resources.environment.valid() && renderer.environment_alive(resources.environment))
        (void)renderer.destroy_environment(resources.environment);
    if (state.environment_lighting_resource == resources.environment) state.environment_lighting_resource = {};
    if (resources.environment_texture.valid() && renderer.texture_alive(resources.environment_texture))
        (void)renderer.destroy_texture(resources.environment_texture);
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
