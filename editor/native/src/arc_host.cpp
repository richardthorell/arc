#include <arc/editor/editor_gizmo.h>
#include <arc/editor/editor_interaction.h>
#include <arc/editor/editor_state.h>
#include <arc/editor/viewport_render_stats.h>
#include <arc/geometric/box.h>
#include <arc/render/texture.h>
#include <arc/scene/scene.h>

#include <algorithm>
#include <array>
#include <cstring>
#include <limits>
#include <optional>
#include <source_location>
#include <string_view>
#include <nlohmann/json.hpp>

namespace arc::render
{
namespace
{
std::optional<texture_data> cubemap_cross_thumbnail_source(const texture_data& texture)
{
    if (texture.dimension != texture_dimension::cube || texture.width == 0u || texture.height == 0u ||
        texture.width != texture.height || texture.array_layers == 0u || !texture.has_pixels() || texture.mips.empty())
        return std::nullopt;

    const std::size_t bytes_per_pixel = texture.format == texture_format::rgba32f   ? sizeof(float) * 4u
                                        : texture.format == texture_format::rgba8_unorm ||
                                                  texture.format == texture_format::rgba8_srgb
                                            ? 4u
                                            : 0u;
    if (bytes_per_pixel == 0u) return std::nullopt;

    const auto& base_mip = texture.mips.front();
    const std::size_t face_bytes = static_cast<std::size_t>(texture.width) * texture.height * bytes_per_pixel;
    constexpr std::size_t cube_face_count = 6u;
    const std::size_t cube_bytes = face_bytes * cube_face_count;
    if (base_mip.offset > texture.pixels.size() || base_mip.size < cube_bytes ||
        cube_bytes > texture.pixels.size() - base_mip.offset)
        return std::nullopt;

    texture_data cross;
    cross.name = texture.name;
    cross.source_path = texture.source_path;
    cross.width = texture.width * 4u;
    cross.height = texture.height * 3u;
    cross.depth = 1u;
    cross.dimension = texture_dimension::texture_2d;
    cross.format = texture.format;
    cross.color_space = texture.color_space;
    cross.semantic = texture.semantic;
    cross.mime_type = texture.mime_type;
    cross.array_layers = 1u;
    cross.mip_levels = 1u;
    cross.compressed = false;
    cross.dds = false;
    cross.pixels.resize(static_cast<std::size_t>(cross.width) * cross.height * bytes_per_pixel);

    // ARC cube payloads use the conventional +X, -X, +Y, -Y, +Z, -Z face order.
    // Lay those faces out as a conventional unfolded cross. This is deliberately
    // topology-based rather than source-format-based so any future cube loader gets
    // the same Content Browser thumbnail automatically.
    constexpr std::array<std::array<std::uint32_t, 2>, cube_face_count> placements{{
        {{2u, 1u}}, // +X
        {{0u, 1u}}, // -X
        {{1u, 0u}}, // +Y
        {{1u, 2u}}, // -Y
        {{1u, 1u}}, // +Z
        {{3u, 1u}}, // -Z
    }};

    const std::size_t source_row_bytes = static_cast<std::size_t>(texture.width) * bytes_per_pixel;
    const std::size_t target_row_bytes = static_cast<std::size_t>(cross.width) * bytes_per_pixel;
    for (std::size_t face = 0; face < cube_face_count; ++face)
    {
        const auto target_x = static_cast<std::size_t>(placements[face][0]) * texture.width;
        const auto target_y = static_cast<std::size_t>(placements[face][1]) * texture.height;
        for (std::uint32_t y = 0; y < texture.height; ++y)
        {
            const auto source_offset = base_mip.offset + face * face_bytes + static_cast<std::size_t>(y) * source_row_bytes;
            const auto target_offset =
                (target_y + y) * target_row_bytes + target_x * bytes_per_pixel;
            std::memcpy(cross.pixels.data() + target_offset, texture.pixels.data() + source_offset, source_row_bytes);
        }
    }
    cross.mips.push_back({.width = cross.width, .height = cross.height, .offset = 0u, .size = cross.pixels.size()});
    return cross;
}
} // namespace

texture_load_result load_texture_asset_for_editor_host(const std::filesystem::path& path,
                                                       const std::source_location& caller)
{
    auto loaded = load_texture_asset(path);
    if (!loaded.succeeded() || std::string_view{caller.function_name()}.find("asset_thumbnail") == std::string_view::npos)
        return loaded;
    if (auto cross = cubemap_cross_thumbnail_source(loaded.texture)) loaded.texture = std::move(*cross);
    return loaded;
}
} // namespace arc::render

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

// The legacy host owns thumbnail generation inside arc_host_base.inc. Redirect
// only its texture-load call through an editor-local adapter so cube topology can
// be unfolded before the existing BMP/tonemap path runs. Non-thumbnail callers
// receive the original texture unchanged.
#define load_texture_asset(path) load_texture_asset_for_editor_host((path), std::source_location::current())

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

#undef load_texture_asset
#undef collect_viewport_render_stats
#undef focus_selected_entity