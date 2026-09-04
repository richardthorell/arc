#include <arc/editor/arc_host.h>

#include <arc/diagnostics/diagnostics.h>
#include <arc/editor/editor_defaults.h>
#include <arc/editor/editor_gizmo.h>
#include <arc/editor/editor_history.h>
#include <arc/editor/editor_interaction.h>
#include <arc/editor/editor_state.h>
#include <arc/editor/material_preview.h>
#include <arc/editor/model_preview.h>
#include <arc/editor/prefab_document.h>
#include <arc/editor/scene_document.h>
#include <arc/editor/terrain_heightmap_io.h>
#include "project_module_loader.h"
#include <arc/editor/world_environment_host.h>
#include <arc/assets/assets.h>
#include <arc/assets/cook.h>
#include <arc/geometric/box.h>
#include <arc/framework/framework.h>
#include <arc/project/project.h>
#include <arc/render/render.h>
#include <arc/render_tools/texture_cooker.h>
#include <arc/scene/scene.h>

#include <algorithm>
#include <array>
#include <bit>
#include <chrono>
#include <cctype>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <iomanip>
#include <limits>
#include <memory>
#include <optional>
#include <sstream>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>
#include <nlohmann/json.hpp>
#include <png.h>

namespace arc::editor
{
namespace
{

constexpr std::string_view base64_alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

template <class Component>
void copy_component(const ecs::world& source, ecs::world& target, ecs::entity from, ecs::entity to)
{
    if (const auto* value = source.try_get<Component>(from)) target.emplace<Component>(to, *value);
}

ecs::entity duplicate_entity_subtree(editor_scene_state& state, ecs::entity source, ecs::entity parent = {})
{
    if (!state.scene.alive(source)) return {};
    const auto duplicate = state.scene.create();
    copy_component<scene::name_component>(state.scene, state.scene, source, duplicate);
    if (auto* name = state.scene.try_get<scene::name_component>(duplicate)) name->value += " Copy";
    copy_component<scene::tag_component>(state.scene, state.scene, source, duplicate);
    copy_component<scene::active_component>(state.scene, state.scene, source, duplicate);
    copy_component<scene::transform_component>(state.scene, state.scene, source, duplicate);
    copy_component<scene::bounds_component>(state.scene, state.scene, source, duplicate);
    copy_component<scene::camera_component>(state.scene, state.scene, source, duplicate);
    copy_component<scene::mesh_renderer_component>(state.scene, state.scene, source, duplicate);
    copy_component<scene::skinned_mesh_renderer_component>(state.scene, state.scene, source, duplicate);
    copy_component<scene::lod_component>(state.scene, state.scene, source, duplicate);
    copy_component<scene::render_layer_component>(state.scene, state.scene, source, duplicate);
    copy_component<scene::directional_light_component>(state.scene, state.scene, source, duplicate);
    copy_component<scene::point_light_component>(state.scene, state.scene, source, duplicate);
    copy_component<scene::spot_light_component>(state.scene, state.scene, source, duplicate);
    copy_component<scene::area_light_component>(state.scene, state.scene, source, duplicate);
    copy_component<scene::world_environment_component>(state.scene, state.scene, source, duplicate);
    copy_component<scene::sky_atmosphere_component>(state.scene, state.scene, source, duplicate);
    copy_component<scene::celestial_sky_component>(state.scene, state.scene, source, duplicate);
    copy_component<scene::cloud_layers_component>(state.scene, state.scene, source, duplicate);
    copy_component<scene::environment_lighting_component>(state.scene, state.scene, source, duplicate);
    copy_component<scene::height_fog_component>(state.scene, state.scene, source, duplicate);
    copy_component<scene::terrain_component>(state.scene, state.scene, source, duplicate);
    copy_component<scene::water_component>(state.scene, state.scene, source, duplicate);
    copy_component<scene::vegetation_component>(state.scene, state.scene, source, duplicate);
    copy_component<scene::decal_component>(state.scene, state.scene, source, duplicate);
    state.scene.emplace<scene::persistent_id_component>(duplicate, ecs::generate_entity_guid());
    state.scene.emplace<scene::hierarchy_component>(duplicate);
    state.scene.emplace<scene::selection_component>(duplicate, false);
    const auto new_guid = entity_guid_of(state, duplicate);
    if (const auto* binding = find_asset_binding(state, entity_guid_of(state, source)))
    {
        auto copied = *binding;
        copied.entity = new_guid;
        state.asset_bindings.push_back(std::move(copied));
    }
    const auto source_guid = entity_guid_of(state, source);
    const auto unknown_count = state.unknown_component_records.size();
    for (std::size_t index = 0; index < unknown_count; ++index)
        if (state.unknown_component_records[index].first == source_guid)
        {
            const auto record = state.unknown_component_records[index].second;
            state.unknown_component_records.emplace_back(new_guid, record);
        }
    const auto preserved_count = state.preserved_component_records.size();
    for (std::size_t index = 0; index < preserved_count; ++index)
    {
        const auto preserved = state.preserved_component_records[index];
        if (preserved.entity == source_guid)
            state.preserved_component_records.push_back({new_guid, preserved.component_name, preserved.json});
    }
    if (state.scene.alive(parent))
        scene::reparent(state.scene, duplicate, parent, {}, scene::reparent_transform_policy::preserve_local);
    for (const auto child : scene::children(state.scene, source))
        duplicate_entity_subtree(state, child, duplicate);
    return duplicate;
}

void resolve_editor_asset_bindings(editor_scene_state& state, assets::asset_manager& registry,
                                   const std::filesystem::path& project_root)
{
    const auto resolve = [&](assets::asset_reference& reference)
    {
        if (reference.guid.valid() || reference.path_hint.empty()) return;
        std::filesystem::path path(reference.path_hint);
        if (path.is_absolute()) path = path.lexically_relative(project_root);
        auto normalized = assets::normalize_asset_path(path);
        if (!registry.find(normalized))
        {
            const auto under_assets = assets::normalize_asset_path(std::filesystem::path("assets") / path);
            if (registry.find(under_assets)) normalized = under_assets;
        }
        auto resolved = registry.resolve(normalized, reference.expected_type);
        if (!resolved.path_hint.empty())
            reference = std::move(resolved);
        else
            reference.path_hint = normalized;
    };
    resolve(state.primitive_material_asset);
    for (auto& binding : state.asset_bindings)
    {
        resolve(binding.source);
        resolve(binding.material);
    }
}

void register_editor_asset_fallbacks(const editor_scene_state& state, assets::asset_manager& registry)
{
    if (state.default_mesh.valid())
    {
        registry.register_virtual_asset(
            assets::fallback_assets::missing_mesh, assets::asset_types::static_mesh,
            assets::asset_payload::make<render::mesh_handle>(
                assets::asset_types::static_mesh, std::make_shared<const render::mesh_handle>(state.default_mesh)),
            "missing-mesh");
        registry.set_fallback(assets::asset_types::static_mesh, assets::fallback_assets::missing_mesh);
    }
    if (state.default_material.valid())
    {
        registry.register_virtual_asset(
            assets::fallback_assets::error_material, assets::asset_types::material,
            assets::asset_payload::make<render::material_handle>(
                assets::asset_types::material, std::make_shared<const render::material_handle>(state.default_material)),
            "error-material");
        registry.set_fallback(assets::asset_types::material, assets::fallback_assets::error_material);
    }
    if (state.environment_lighting_resource.valid())
    {
        registry.register_virtual_asset(
            assets::fallback_assets::neutral_environment, assets::asset_types::environment,
            assets::asset_payload::make<render::environment_handle>(
                assets::asset_types::environment,
                std::make_shared<const render::environment_handle>(state.environment_lighting_resource)),
            "neutral-environment");
        registry.set_fallback(assets::asset_types::environment, assets::fallback_assets::neutral_environment);
    }
}

template <class Command> constexpr bool is_authoring_command() noexcept
{
    return std::is_same_v<Command, host_create_entity_command> || std::is_same_v<Command, host_delete_entity_command> ||
           std::is_same_v<Command, host_duplicate_entity_command> ||
           std::is_same_v<Command, host_reparent_entity_command> ||
           std::is_same_v<Command, host_create_prefab_command> ||
           std::is_same_v<Command, host_instantiate_prefab_command> ||
           std::is_same_v<Command, host_apply_prefab_command> || std::is_same_v<Command, host_revert_prefab_command> ||
           std::is_same_v<Command, host_revert_prefab_override_command> ||
           std::is_same_v<Command, host_unpack_prefab_command> ||
           std::is_same_v<Command, host_reorder_entity_command> ||
           std::is_same_v<Command, host_rename_entity_command> || std::is_same_v<Command, host_set_active_command> ||
           std::is_same_v<Command, host_set_tag_command> || std::is_same_v<Command, host_set_transform_command> ||
           std::is_same_v<Command, host_set_render_layer_command> ||
           std::is_same_v<Command, host_set_mobility_command> || std::is_same_v<Command, host_set_camera_command> ||
           std::is_same_v<Command, host_set_light_command> || std::is_same_v<Command, host_set_mesh_renderer_command> ||
           std::is_same_v<Command, host_set_terrain_command> || std::is_same_v<Command, host_terrain_stroke_command> ||
           std::is_same_v<Command, host_set_terrain_layer_command> ||
           std::is_same_v<Command, host_create_terrain_command> ||
           std::is_same_v<Command, host_generate_terrain_command> ||
           std::is_same_v<Command, host_import_terrain_heightmap_command> ||
           std::is_same_v<Command, host_resample_terrain_command> ||
           std::is_same_v<Command, host_set_entity_material_command> ||
           std::is_same_v<Command, host_component_operation_command> ||
           std::is_same_v<Command, host_patch_project_component_command> ||
           std::is_same_v<Command, host_set_world_environment_command> ||
           std::is_same_v<Command, host_apply_world_environment_preset_command> ||
           std::is_same_v<Command, host_set_environment_hdri_command>;
}

template <class Command> constexpr bool is_project_mutation() noexcept
{
    return is_authoring_command<Command>() || std::is_same_v<Command, host_save_scene_command> ||
           std::is_same_v<Command, host_save_scene_as_command> ||
           std::is_same_v<Command, host_reload_project_module_command> ||
           std::is_same_v<Command, host_asset_reimport_command> ||
           std::is_same_v<Command, host_set_texture_streaming_mode_command> ||
           std::is_same_v<Command, host_patch_texture_settings_command> ||
           std::is_same_v<Command, host_asset_cancel_import_command> ||
           std::is_same_v<Command, host_asset_move_command> || std::is_same_v<Command, host_asset_rename_command>;
}

std::string history_label(const host_command_payload& command)
{
    return std::visit(
        [](const auto& value) -> std::string
        {
            using type = std::decay_t<decltype(value)>;
            if constexpr (std::is_same_v<type, host_create_entity_command>)
                return "Create Entity";
            else if constexpr (std::is_same_v<type, host_delete_entity_command>)
                return "Delete Entity";
            else if constexpr (std::is_same_v<type, host_duplicate_entity_command>)
                return "Duplicate Entity";
            else if constexpr (std::is_same_v<type, host_create_prefab_command>)
                return "Create Prefab";
            else if constexpr (std::is_same_v<type, host_instantiate_prefab_command>)
                return "Instantiate Prefab";
            else if constexpr (std::is_same_v<type, host_apply_prefab_command>)
                return "Apply Prefab";
            else if constexpr (std::is_same_v<type, host_revert_prefab_command>)
                return "Revert Prefab";
            else if constexpr (std::is_same_v<type, host_revert_prefab_override_command>)
                return "Revert Prefab Override";
            else if constexpr (std::is_same_v<type, host_unpack_prefab_command>)
                return "Unpack Prefab";
            else if constexpr (std::is_same_v<type, host_reparent_entity_command>)
                return "Reparent Entity";
            else if constexpr (std::is_same_v<type, host_reorder_entity_command>)
                return "Reorder Entity";
            else if constexpr (std::is_same_v<type, host_rename_entity_command>)
                return "Rename Entity";
            else if constexpr (std::is_same_v<type, host_set_transform_command>)
                return "Transform Entity";
            else if constexpr (std::is_same_v<type, host_set_mobility_command>)
                return "Set Mobility";
            else if constexpr (std::is_same_v<type, host_set_entity_material_command>)
                return "Assign Material";
            else if constexpr (std::is_same_v<type, host_set_terrain_command>)
                return "Edit Terrain";
            else if constexpr (std::is_same_v<type, host_set_terrain_layer_command>)
                return "Assign Terrain Layer";
            else if constexpr (std::is_same_v<type, host_terrain_stroke_command>)
                return "Terrain Stroke";
            else if constexpr (std::is_same_v<type, host_create_terrain_command>)
                return "Create Terrain";
            else if constexpr (std::is_same_v<type, host_generate_terrain_command>)
                return "Generate Terrain";
            else if constexpr (std::is_same_v<type, host_import_terrain_heightmap_command>)
                return "Import Terrain Heightmap";
            else if constexpr (std::is_same_v<type, host_resample_terrain_command>)
                return "Resample Terrain";
            else if constexpr (std::is_same_v<type, host_set_world_environment_command> ||
                               std::is_same_v<type, host_apply_world_environment_preset_command> ||
                               std::is_same_v<type, host_set_environment_hdri_command>)
                return "Edit World Environment";
            else
                return "Edit Component";
        },
        command);
}

std::string base64_encode(const std::vector<std::byte>& bytes)
{
    std::string encoded;
    encoded.reserve(((bytes.size() + 2u) / 3u) * 4u);
    for (std::size_t offset = 0; offset < bytes.size(); offset += 3u)
    {
        const auto first = std::to_integer<std::uint32_t>(bytes[offset]);
        const auto second = offset + 1u < bytes.size() ? std::to_integer<std::uint32_t>(bytes[offset + 1u]) : 0u;
        const auto third = offset + 2u < bytes.size() ? std::to_integer<std::uint32_t>(bytes[offset + 2u]) : 0u;
        const auto packed = (first << 16u) | (second << 8u) | third;
        encoded.push_back(base64_alphabet[(packed >> 18u) & 0x3fu]);
        encoded.push_back(base64_alphabet[(packed >> 12u) & 0x3fu]);
        encoded.push_back(offset + 1u < bytes.size() ? base64_alphabet[(packed >> 6u) & 0x3fu] : '=');
        encoded.push_back(offset + 2u < bytes.size() ? base64_alphabet[packed & 0x3fu] : '=');
    }
    return encoded;
}

void write_u16(std::vector<std::byte>& bytes, std::size_t offset, std::uint16_t value)
{
    bytes[offset] = static_cast<std::byte>(value & 0xffu);
    bytes[offset + 1u] = static_cast<std::byte>((value >> 8u) & 0xffu);
}

void write_u32(std::vector<std::byte>& bytes, std::size_t offset, std::uint32_t value)
{
    for (std::size_t index = 0; index < 4u; ++index)
        bytes[offset + index] = static_cast<std::byte>((value >> (index * 8u)) & 0xffu);
}

std::uint8_t preview_channel(float linear_value)
{
    const float mapped = std::max(0.0f, linear_value) / (1.0f + std::max(0.0f, linear_value));
    const float srgb = mapped <= 0.0031308f ? mapped * 12.92f : 1.055f * std::pow(mapped, 1.0f / 2.4f) - 0.055f;
    return static_cast<std::uint8_t>(std::clamp(std::lround(srgb * 255.0f), 0l, 255l));
}

std::vector<std::byte> texture_preview_png_rgba(const render::texture_data& texture)
{
    if (!texture.has_pixels() || texture.width == 0u || texture.height == 0u) return {};
    if (texture.format != render::texture_format::rgba8_unorm && texture.format != render::texture_format::rgba8_srgb)
        return {};

    png_image image{};
    image.version = PNG_IMAGE_VERSION;
    image.width = texture.width;
    image.height = texture.height;
    image.format = PNG_FORMAT_RGBA;

    png_alloc_size_t byte_count{};
    if (!png_image_write_to_memory(&image, nullptr, &byte_count, 0, texture.pixels.data(), 0, nullptr)) return {};
    std::vector<std::byte> bytes(static_cast<std::size_t>(byte_count));
    if (!png_image_write_to_memory(&image, bytes.data(), &byte_count, 0, texture.pixels.data(), 0, nullptr)) return {};
    bytes.resize(static_cast<std::size_t>(byte_count));
    return bytes;
}

std::vector<std::byte> texture_preview_bmp(const render::texture_data& texture, std::uint32_t max_size)
{
    if (!texture.has_pixels() || texture.width == 0u || texture.height == 0u) return {};
    const bool float_pixels = texture.format == render::texture_format::rgba32f;
    const bool byte_pixels =
        texture.format == render::texture_format::rgba8_unorm || texture.format == render::texture_format::rgba8_srgb;
    const std::size_t bytes_per_pixel = float_pixels ? sizeof(float) * 4u : 4u;
    const std::size_t base_level_size = static_cast<std::size_t>(texture.width) * texture.height * bytes_per_pixel;
    if ((!float_pixels && !byte_pixels) || texture.pixels.size() < base_level_size) return {};

    const float scale =
        std::min(1.0f, static_cast<float>(max_size) / static_cast<float>(std::max(texture.width, texture.height)));
    const auto width = std::max(1u, static_cast<std::uint32_t>(std::lround(texture.width * scale)));
    const auto height = std::max(1u, static_cast<std::uint32_t>(std::lround(texture.height * scale)));
    constexpr std::size_t header_size = 54u;
    std::vector<std::byte> bmp(header_size + static_cast<std::size_t>(width) * height * 4u);
    bmp[0] = std::byte{'B'};
    bmp[1] = std::byte{'M'};
    write_u32(bmp, 2u, static_cast<std::uint32_t>(bmp.size()));
    write_u32(bmp, 10u, static_cast<std::uint32_t>(header_size));
    write_u32(bmp, 14u, 40u);
    write_u32(bmp, 18u, width);
    write_u32(bmp, 22u, static_cast<std::uint32_t>(-static_cast<std::int32_t>(height)));
    write_u16(bmp, 26u, 1u);
    write_u16(bmp, 28u, 32u);
    write_u32(bmp, 34u, width * height * 4u);

    for (std::uint32_t y = 0; y < height; ++y)
    {
        const auto source_y = std::min(texture.height - 1u, y * texture.height / height);
        for (std::uint32_t x = 0; x < width; ++x)
        {
            const auto source_x = std::min(texture.width - 1u, x * texture.width / width);
            const auto source_pixel = static_cast<std::size_t>(source_y * texture.width + source_x);
            std::array<std::uint8_t, 4> rgba{};
            if (float_pixels)
            {
                std::array<float, 4> linear{};
                std::memcpy(linear.data(), texture.pixels.data() + source_pixel * sizeof(float) * 4u, sizeof(linear));
                rgba = {preview_channel(linear[0]), preview_channel(linear[1]), preview_channel(linear[2]), 255u};
            }
            else
            {
                const auto offset = source_pixel * 4u;
                rgba = {std::to_integer<std::uint8_t>(texture.pixels[offset]),
                        std::to_integer<std::uint8_t>(texture.pixels[offset + 1u]),
                        std::to_integer<std::uint8_t>(texture.pixels[offset + 2u]),
                        std::to_integer<std::uint8_t>(texture.pixels[offset + 3u])};
            }
            const auto target = header_size + static_cast<std::size_t>(y * width + x) * 4u;
            bmp[target] = static_cast<std::byte>(rgba[2]);
            bmp[target + 1u] = static_cast<std::byte>(rgba[1]);
            bmp[target + 2u] = static_cast<std::byte>(rgba[0]);
            bmp[target + 3u] = static_cast<std::byte>(rgba[3]);
        }
    }
    return bmp;
}

std::optional<std::filesystem::path> resolve_project_asset(const std::filesystem::path& asset_root,
                                                           const std::filesystem::path& relative_path)
{
    if (asset_root.empty() || relative_path.empty()) return std::nullopt;
    std::error_code error;
    const auto root = std::filesystem::weakly_canonical(asset_root, error);
    if (error) return std::nullopt;
    auto candidate = std::filesystem::path(relative_path);
    if (candidate.is_relative()) candidate = root / candidate;
    candidate = std::filesystem::weakly_canonical(candidate, error);
    if (error || !std::filesystem::is_regular_file(candidate, error)) return std::nullopt;
    const auto relative = std::filesystem::relative(candidate, root, error);
    if (error || relative.empty()) return std::nullopt;
    for (const auto& part : relative)
    {
        if (part == "..") return std::nullopt;
    }
    return candidate;
}

struct resolved_editor_asset
{
    std::filesystem::path path;
    std::filesystem::path asset_root;
    bool read_only{};
};

std::optional<resolved_editor_asset> resolve_editor_asset(const editor_asset_state& asset_state,
                                                          const assets::asset_manager* registry,
                                                          const std::filesystem::path& project_root,
                                                          const std::filesystem::path& reference)
{
    if (reference.empty()) return std::nullopt;
    const auto normalized = arc::assets::normalize_asset_path(reference);
    if (normalized == "builtin" || normalized.starts_with("builtin/"))
    {
        const auto relative = std::filesystem::path(normalized).lexically_relative("builtin");
        if (relative.empty()) return std::nullopt;
        for (const auto& builtin_root : asset_state.builtin_roots)
        {
            if (const auto resolved = resolve_project_asset(builtin_root, relative))
                return resolved_editor_asset{.path = *resolved, .asset_root = builtin_root, .read_only = true};
        }
        return std::nullopt;
    }
    if (const auto resolved = resolve_project_asset(asset_state.root, reference))
        return resolved_editor_asset{.path = *resolved, .asset_root = asset_state.root, .read_only = false};

    // Assets from secondary project content roots are exposed using their registry path rather than a path
    // relative to the primary asset root. Resolve that catalog identity back to the owning source mount so
    // previews and material assignment use the same asset that the picker displays.
    if (!registry || project_root.empty()) return std::nullopt;
    const auto snapshot = registry->find(normalized);
    if (!snapshot || snapshot->read_only) return std::nullopt;
    const auto candidate = (project_root / snapshot->source_path).lexically_normal();
    const auto& config = registry->config();
    if (const auto resolved = resolve_project_asset(config.asset_root, candidate))
        return resolved_editor_asset{.path = *resolved, .asset_root = config.asset_root, .read_only = false};
    for (const auto& source_root : config.additional_source_roots)
    {
        if (const auto resolved = resolve_project_asset(source_root, candidate))
            return resolved_editor_asset{.path = *resolved, .asset_root = source_root, .read_only = false};
    }
    return std::nullopt;
}

std::optional<std::filesystem::path> resolve_project_document(const std::filesystem::path& project_root,
                                                              const std::filesystem::path& authored, bool must_exist)
{
    if (project_root.empty() || authored.empty()) return std::nullopt;
    std::error_code error;
    const auto root = std::filesystem::weakly_canonical(project_root, error);
    if (error) return std::nullopt;
    auto candidate = authored.is_absolute() ? authored : root / authored;
    candidate = must_exist ? std::filesystem::weakly_canonical(candidate, error) : candidate.lexically_normal();
    if (error || (must_exist && !std::filesystem::is_regular_file(candidate, error))) return std::nullopt;
    const auto relative = std::filesystem::relative(candidate, root, error);
    if (error || relative.empty() || relative.is_absolute()) return std::nullopt;
    for (const auto& part : relative)
        if (part == "..") return std::nullopt;
    return candidate;
}

ecs::entity to_scene_entity(host_entity_id entity) noexcept
{
    return {.index = entity.index, .generation = entity.generation};
}

host_entity_id to_host_entity(ecs::entity entity) noexcept
{
    return {.index = entity.index, .generation = entity.generation};
}

host_vec3 to_host_vec3(const math::vector3f& value) noexcept
{
    return {value[0], value[1], value[2]};
}

math::vector3f to_math_vec3(host_vec3 value) noexcept
{
    return {value.x, value.y, value.z};
}

host_quat to_host_quat(const math::quatf& value) noexcept
{
    return {value[0], value[1], value[2], value[3]};
}

math::quatf to_math_quat(host_quat value) noexcept
{
    return {value.x, value.y, value.z, value.w};
}

host_vec4 to_host_vec4(const math::vector4f& value) noexcept
{
    return {value[0], value[1], value[2], value[3]};
}

math::vector4f to_math_vec4(host_vec4 value) noexcept
{
    return {value.x, value.y, value.z, value.w};
}

host_transform to_host_transform(const scene::transform_component& transform) noexcept
{
    return {.position = to_host_vec3(transform.position),
            .rotation = to_host_quat(transform.rotation),
            .scale = to_host_vec3(transform.scale)};
}

scene::transform_component to_scene_transform(const host_transform& transform) noexcept
{
    scene::transform_component result;
    result.position = to_math_vec3(transform.position);
    result.rotation = to_math_quat(transform.rotation);
    result.scale = to_math_vec3(transform.scale);
    result.mark_dirty();
    return result;
}

host_camera_projection to_host_projection(scene::camera_projection projection) noexcept
{
    return projection == scene::camera_projection::orthographic ? host_camera_projection::orthographic
                                                                : host_camera_projection::perspective;
}

host_camera_snapshot to_host_camera(const scene::camera_component& camera) noexcept
{
    return {.projection = to_host_projection(camera.projection),
            .fov_y_degrees = math::to_degrees(camera.fov_y_radians),
            .orthographic_height = camera.orthographic_height,
            .near_plane = camera.near_plane,
            .far_plane = camera.far_plane,
            .active = camera.active,
            .clear_color = to_host_vec4(camera.clear_color),
            .exposure_mode = camera.exposure.mode == render::exposure_mode::manual ? host_exposure_mode::manual
                                                                                   : host_exposure_mode::automatic,
            .exposure_metering = camera.exposure.metering == render::exposure_metering_mode::center_weighted
                                     ? host_exposure_metering_mode::center_weighted
                                     : host_exposure_metering_mode::average,
            .manual_ev100 = camera.exposure.manual_ev100,
            .exposure_compensation = camera.exposure.compensation_ev,
            .minimum_ev100 = camera.exposure.minimum_ev100,
            .maximum_ev100 = camera.exposure.maximum_ev100,
            .brighten_speed = camera.exposure.brighten_speed,
            .darken_speed = camera.exposure.darken_speed,
            .anti_aliasing = static_cast<host_camera_anti_aliasing>(camera.anti_aliasing)};
}

bool valid_camera(const host_camera_snapshot& camera) noexcept
{
    const auto finite = [](float value) { return std::isfinite(value); };
    return finite(camera.fov_y_degrees) && camera.fov_y_degrees > 1.0f && camera.fov_y_degrees < 179.0f &&
           finite(camera.orthographic_height) && camera.orthographic_height > 0.0f && finite(camera.near_plane) &&
           camera.near_plane > 0.0f && finite(camera.far_plane) && camera.far_plane > camera.near_plane &&
           finite(camera.manual_ev100) && finite(camera.exposure_compensation) && finite(camera.minimum_ev100) &&
           finite(camera.maximum_ev100) && camera.maximum_ev100 > camera.minimum_ev100 &&
           finite(camera.brighten_speed) && camera.brighten_speed >= 0.0f && finite(camera.darken_speed) &&
           camera.darken_speed >= 0.0f && finite(camera.clear_color.x) && camera.clear_color.x >= 0.0f &&
           camera.clear_color.x <= 1.0f && finite(camera.clear_color.y) && camera.clear_color.y >= 0.0f &&
           camera.clear_color.y <= 1.0f && finite(camera.clear_color.z) && camera.clear_color.z >= 0.0f &&
           camera.clear_color.z <= 1.0f && finite(camera.clear_color.w) && camera.clear_color.w >= 0.0f &&
           camera.clear_color.w <= 1.0f;
}

bool valid_base_color_tint(host_vec4 tint) noexcept
{
    const auto valid_channel = [](float value) { return std::isfinite(value) && value >= 0.0f && value <= 1.0f; };
    return valid_channel(tint.x) && valid_channel(tint.y) && valid_channel(tint.z) && valid_channel(tint.w);
}

scene::camera_component to_scene_camera(const host_camera_snapshot& camera) noexcept
{
    return {.projection = camera.projection == host_camera_projection::orthographic
                              ? scene::camera_projection::orthographic
                              : scene::camera_projection::perspective,
            .fov_y_radians = math::to_radians(camera.fov_y_degrees),
            .near_plane = camera.near_plane,
            .far_plane = camera.far_plane,
            .orthographic_height = camera.orthographic_height,
            .active = camera.active,
            .clear_color = to_math_vec4(camera.clear_color),
            .exposure = {.mode = camera.exposure_mode == host_exposure_mode::manual ? render::exposure_mode::manual
                                                                                    : render::exposure_mode::automatic,
                         .metering = camera.exposure_metering == host_exposure_metering_mode::center_weighted
                                         ? render::exposure_metering_mode::center_weighted
                                         : render::exposure_metering_mode::average,
                         .manual_ev100 = camera.manual_ev100,
                         .compensation_ev = camera.exposure_compensation,
                         .minimum_ev100 = camera.minimum_ev100,
                         .maximum_ev100 = camera.maximum_ev100,
                         .brighten_speed = camera.brighten_speed,
                         .darken_speed = camera.darken_speed},
            .anti_aliasing = static_cast<render::camera_anti_aliasing_override>(camera.anti_aliasing)};
}

host_light_unit to_host_light_unit(render::light_intensity_unit unit) noexcept
{
    switch (unit)
    {
        case render::light_intensity_unit::lumen:
            return host_light_unit::lumen;
        case render::light_intensity_unit::candela:
            return host_light_unit::candela;
        case render::light_intensity_unit::lux:
            return host_light_unit::lux;
        case render::light_intensity_unit::nit:
            return host_light_unit::nit;
        case render::light_intensity_unit::unitless:
            return host_light_unit::unitless;
    }
    return host_light_unit::unitless;
}

render::light_intensity_unit to_render_light_unit(host_light_unit unit) noexcept
{
    switch (unit)
    {
        case host_light_unit::lumen:
            return render::light_intensity_unit::lumen;
        case host_light_unit::candela:
            return render::light_intensity_unit::candela;
        case host_light_unit::lux:
            return render::light_intensity_unit::lux;
        case host_light_unit::nit:
            return render::light_intensity_unit::nit;
        case host_light_unit::unitless:
            return render::light_intensity_unit::unitless;
    }
    return render::light_intensity_unit::unitless;
}

host_mobility to_host_mobility(render::render_mobility mobility) noexcept
{
    switch (mobility)
    {
        case render::render_mobility::static_object:
            return host_mobility::static_object;
        case render::render_mobility::stationary:
            return host_mobility::stationary;
        case render::render_mobility::movable:
            return host_mobility::movable;
    }
    return host_mobility::movable;
}

render::render_mobility to_render_mobility(host_mobility mobility) noexcept
{
    switch (mobility)
    {
        case host_mobility::static_object:
            return render::render_mobility::static_object;
        case host_mobility::stationary:
            return render::render_mobility::stationary;
        case host_mobility::movable:
            return render::render_mobility::movable;
    }
    return render::render_mobility::movable;
}

void copy_shadow_to_host(host_light_snapshot& target, const render::shadow_settings& source) noexcept
{
    target.shadow_resolution = source.resolution;
    target.shadow_priority = source.priority;
    target.shadow_strength = source.strength;
    target.shadow_bias = source.bias;
    target.shadow_normal_bias = source.normal_bias;
    target.shadow_filter = static_cast<std::uint8_t>(source.filter);
    target.contact_shadows = source.contact_shadows;
    target.contact_shadow_length = source.contact_shadow_length;
    target.shadow_cache_mode = static_cast<std::uint8_t>(source.cache_mode);
    target.shadow_map_method = static_cast<std::uint8_t>(source.map_method);
}

void copy_shadow_from_host(render::shadow_settings& target, const host_light_snapshot& source) noexcept
{
    target.enabled = source.casts_shadows;
    target.resolution = source.shadow_resolution;
    target.priority = source.shadow_priority;
    target.strength = source.shadow_strength;
    target.bias = source.shadow_bias;
    target.normal_bias = source.shadow_normal_bias;
    target.filter = static_cast<render::shadow_filter>(source.shadow_filter);
    target.contact_shadows = source.contact_shadows;
    target.contact_shadow_length = source.contact_shadow_length;
    target.cache_mode = static_cast<render::shadow_cache_mode>(source.shadow_cache_mode);
    target.map_method = static_cast<render::shadow_map_method>(source.shadow_map_method);
}

host_light_snapshot to_host_light(const scene::directional_light_component& light) noexcept
{
    host_light_snapshot result{.kind = host_light_kind::directional,
                               .unit = to_host_light_unit(light.intensity_unit),
                               .color = to_host_vec3(light.color),
                               .intensity = light.intensity,
                               .enabled = light.enabled,
                               .casts_shadows = light.casts_shadows,
                               .use_color_temperature = light.use_color_temperature,
                               .temperature_kelvin = light.temperature_kelvin};
    copy_shadow_to_host(result, light.shadow);
    result.cascade_count = light.cascades.cascade_count;
    result.shadow_distance = light.cascades.maximum_distance;
    result.cascade_split_lambda = light.cascades.split_lambda;
    result.cascade_blend_fraction = light.cascades.blend_fraction;
    result.stable_cascades = light.cascades.stable;
    return result;
}

host_light_snapshot to_host_light(const scene::point_light_component& light) noexcept
{
    host_light_snapshot result{.kind = host_light_kind::point,
                               .unit = to_host_light_unit(light.intensity_unit),
                               .color = to_host_vec3(light.color),
                               .intensity = light.intensity,
                               .range = light.range,
                               .enabled = light.enabled,
                               .casts_shadows = light.casts_shadows,
                               .use_color_temperature = light.use_color_temperature,
                               .temperature_kelvin = light.temperature_kelvin};
    copy_shadow_to_host(result, light.shadow);
    return result;
}

host_light_snapshot to_host_light(const scene::spot_light_component& light) noexcept
{
    host_light_snapshot result{.kind = host_light_kind::spot,
                               .unit = to_host_light_unit(light.intensity_unit),
                               .color = to_host_vec3(light.color),
                               .intensity = light.intensity,
                               .range = light.range,
                               .inner_angle_degrees = math::to_degrees(light.inner_angle),
                               .outer_angle_degrees = math::to_degrees(light.outer_angle),
                               .enabled = light.enabled,
                               .casts_shadows = light.casts_shadows,
                               .use_color_temperature = light.use_color_temperature,
                               .temperature_kelvin = light.temperature_kelvin};
    copy_shadow_to_host(result, light.shadow);
    return result;
}

host_light_snapshot to_host_light(const scene::area_light_component& light) noexcept
{
    host_light_snapshot result{.kind = light.shape == render::area_light_shape::disk ? host_light_kind::disk
                                                                                     : host_light_kind::rectangle,
                               .unit = to_host_light_unit(light.intensity_unit),
                               .color = to_host_vec3(light.color),
                               .intensity = light.intensity,
                               .width = light.width,
                               .height = light.height,
                               .two_sided = light.two_sided,
                               .enabled = light.enabled,
                               .casts_shadows = light.casts_shadows,
                               .use_color_temperature = light.use_color_temperature,
                               .temperature_kelvin = light.temperature_kelvin};
    copy_shadow_to_host(result, light.shadow);
    return result;
}

bool valid_light(const host_light_snapshot& light) noexcept
{
    const auto finite_nonnegative = [](float value) { return std::isfinite(value) && value >= 0.0f; };
    const bool color_valid =
        finite_nonnegative(light.color.x) && finite_nonnegative(light.color.y) && finite_nonnegative(light.color.z);
    const bool common_valid =
        color_valid && finite_nonnegative(light.intensity) && std::isfinite(light.temperature_kelvin) &&
        light.temperature_kelvin >= 1000.0f && light.temperature_kelvin <= 40000.0f &&
        light.shadow_resolution >= 128u && light.shadow_resolution <= 8192u && std::isfinite(light.shadow_strength) &&
        light.shadow_strength >= 0.0f && light.shadow_strength <= 1.0f && finite_nonnegative(light.shadow_bias) &&
        finite_nonnegative(light.shadow_normal_bias) &&
        light.shadow_filter <= static_cast<std::uint8_t>(render::shadow_filter::pcss) &&
        std::isfinite(light.contact_shadow_length) && light.contact_shadow_length >= 0.0f &&
        light.shadow_cache_mode <= static_cast<std::uint8_t>(render::shadow_cache_mode::static_only) &&
        light.shadow_map_method <= static_cast<std::uint8_t>(render::shadow_map_method::virtualized) &&
        light.cascade_count >= 1u && light.cascade_count <= render::maximum_directional_shadow_cascades &&
        std::isfinite(light.shadow_distance) && light.shadow_distance > 0.0f &&
        std::isfinite(light.cascade_split_lambda) && light.cascade_split_lambda >= 0.0f &&
        light.cascade_split_lambda <= 1.0f && std::isfinite(light.cascade_blend_fraction) &&
        light.cascade_blend_fraction >= 0.0f && light.cascade_blend_fraction <= 0.3f;
    if (!common_valid) return false;
    const bool unit_valid = (light.kind == host_light_kind::directional &&
                             (light.unit == host_light_unit::lux || light.unit == host_light_unit::unitless)) ||
                            ((light.kind == host_light_kind::point || light.kind == host_light_kind::spot) &&
                             (light.unit == host_light_unit::lumen || light.unit == host_light_unit::candela ||
                              light.unit == host_light_unit::unitless)) ||
                            ((light.kind == host_light_kind::rectangle || light.kind == host_light_kind::disk) &&
                             (light.unit == host_light_unit::lumen || light.unit == host_light_unit::nit ||
                              light.unit == host_light_unit::unitless));
    if (!unit_valid) return false;
    if ((light.kind == host_light_kind::point || light.kind == host_light_kind::spot) &&
        (!std::isfinite(light.range) || light.range <= 0.0f))
        return false;
    if (light.kind == host_light_kind::spot &&
        (!std::isfinite(light.inner_angle_degrees) || !std::isfinite(light.outer_angle_degrees) ||
         light.inner_angle_degrees < 0.0f || light.outer_angle_degrees <= light.inner_angle_degrees ||
         light.outer_angle_degrees >= 179.0f))
        return false;
    return (light.kind != host_light_kind::rectangle && light.kind != host_light_kind::disk) ||
           (std::isfinite(light.width) && light.width > 0.0f && std::isfinite(light.height) && light.height > 0.0f);
}

editor_primitive_type primitive_type_for(host_create_entity_kind kind) noexcept
{
    switch (kind)
    {
        case host_create_entity_kind::plane:
            return editor_primitive_type::plane;
        case host_create_entity_kind::cube:
            return editor_primitive_type::cube;
        case host_create_entity_kind::sphere:
            return editor_primitive_type::sphere;
        case host_create_entity_kind::cylinder:
            return editor_primitive_type::cylinder;
        case host_create_entity_kind::cone:
            return editor_primitive_type::cone;
        case host_create_entity_kind::capsule:
            return editor_primitive_type::capsule;
        default:
            return editor_primitive_type::cube;
    }
}

const char* create_entity_kind_label(host_create_entity_kind kind) noexcept
{
    switch (kind)
    {
        case host_create_entity_kind::empty:
            return "Entity";
        case host_create_entity_kind::plane:
            return "Plane";
        case host_create_entity_kind::cube:
            return "Cube";
        case host_create_entity_kind::sphere:
            return "Sphere";
        case host_create_entity_kind::cylinder:
            return "Cylinder";
        case host_create_entity_kind::cone:
            return "Cone";
        case host_create_entity_kind::capsule:
            return "Capsule";
        case host_create_entity_kind::world_environment:
            return "World Environment";
        case host_create_entity_kind::terrain:
            return "Terrain";
        case host_create_entity_kind::water:
            return "Water Plane";
        case host_create_entity_kind::grass_patch:
            return "Grass Patch";
        case host_create_entity_kind::decal:
            return "Decal";
    }
    return "Entity";
}

scene::camera_projection to_scene_projection(host_camera_projection projection) noexcept
{
    return projection == host_camera_projection::orthographic ? scene::camera_projection::orthographic
                                                              : scene::camera_projection::perspective;
}

render::render_mode to_render_mode(host_render_mode mode) noexcept
{
    return mode == host_render_mode::wireframe ? render::render_mode::wireframe : render::render_mode::shaded;
}

render::anti_aliasing_method to_render_anti_aliasing(host_camera_anti_aliasing method) noexcept
{
    switch (method)
    {
        case host_camera_anti_aliasing::disabled:
            return render::anti_aliasing_method::disabled;
        case host_camera_anti_aliasing::fxaa:
            return render::anti_aliasing_method::fxaa;
        case host_camera_anti_aliasing::taa:
            return render::anti_aliasing_method::taa;
        case host_camera_anti_aliasing::taau:
            return render::anti_aliasing_method::taau;
        case host_camera_anti_aliasing::inherit:
            break;
    }
    return render::anti_aliasing_method::auto_select;
}

render::mesh_visualization_mode to_visualization(host_visualization_mode mode) noexcept
{
    switch (mode)
    {
        case host_visualization_mode::albedo:
            return render::mesh_visualization_mode::albedo;
        case host_visualization_mode::opacity:
            return render::mesh_visualization_mode::opacity;
        case host_visualization_mode::world_normal:
            return render::mesh_visualization_mode::world_normal;
        case host_visualization_mode::specularity:
            return render::mesh_visualization_mode::specularity;
        case host_visualization_mode::gloss:
            return render::mesh_visualization_mode::gloss;
        case host_visualization_mode::metalness:
            return render::mesh_visualization_mode::metalness;
        case host_visualization_mode::ao:
            return render::mesh_visualization_mode::ao;
        case host_visualization_mode::emission:
            return render::mesh_visualization_mode::emission;
        case host_visualization_mode::lighting:
            return render::mesh_visualization_mode::lighting;
        case host_visualization_mode::uv0:
            return render::mesh_visualization_mode::uv0;
        case host_visualization_mode::cascade_debug:
            return render::mesh_visualization_mode::cascade_debug;
        case host_visualization_mode::shadow_mask:
            return render::mesh_visualization_mode::shadow_mask;
        case host_visualization_mode::light_complexity:
            return render::mesh_visualization_mode::light_complexity;
        case host_visualization_mode::cluster_debug:
            return render::mesh_visualization_mode::cluster_debug;
        case host_visualization_mode::virtual_hierarchy_level:
            return render::mesh_visualization_mode::virtual_hierarchy_level;
        case host_visualization_mode::virtual_geometric_error:
            return render::mesh_visualization_mode::virtual_geometric_error;
        case host_visualization_mode::virtual_page_residency:
            return render::mesh_visualization_mode::virtual_page_residency;
        case host_visualization_mode::virtual_overdraw:
            return render::mesh_visualization_mode::virtual_overdraw;
        case host_visualization_mode::virtual_triangles_per_pixel:
            return render::mesh_visualization_mode::virtual_triangles_per_pixel;
        case host_visualization_mode::surface_cards:
            return render::mesh_visualization_mode::surface_cards;
        case host_visualization_mode::surface_card_residency:
            return render::mesh_visualization_mode::surface_card_residency;
        case host_visualization_mode::surface_material_cache:
            return render::mesh_visualization_mode::surface_material_cache;
        case host_visualization_mode::surface_radiance_cache:
            return render::mesh_visualization_mode::surface_radiance_cache;
        case host_visualization_mode::mesh_distance_fields:
            return render::mesh_visualization_mode::mesh_distance_fields;
        case host_visualization_mode::global_distance_field:
            return render::mesh_visualization_mode::global_distance_field;
        case host_visualization_mode::radiance_probes:
            return render::mesh_visualization_mode::radiance_probes;
        case host_visualization_mode::lighting_trace_source:
            return render::mesh_visualization_mode::lighting_trace_source;
        case host_visualization_mode::lighting_hit_distance:
            return render::mesh_visualization_mode::lighting_hit_distance;
        case host_visualization_mode::lighting_temporal_confidence:
            return render::mesh_visualization_mode::lighting_temporal_confidence;
        case host_visualization_mode::indirect_diffuse:
            return render::mesh_visualization_mode::indirect_diffuse;
        case host_visualization_mode::reflections:
            return render::mesh_visualization_mode::reflections;
        case host_visualization_mode::denoiser_variance:
            return render::mesh_visualization_mode::denoiser_variance;
        case host_visualization_mode::terrain_patch_boundaries:
            return render::mesh_visualization_mode::terrain_patch_boundaries;
        case host_visualization_mode::terrain_lod_level:
            return render::mesh_visualization_mode::terrain_lod_level;
        case host_visualization_mode::terrain_hierarchy_nodes:
            return render::mesh_visualization_mode::terrain_hierarchy_nodes;
        case host_visualization_mode::terrain_geometric_error:
            return render::mesh_visualization_mode::terrain_geometric_error;
        case host_visualization_mode::terrain_culled_nodes:
            return render::mesh_visualization_mode::terrain_culled_nodes;
        case host_visualization_mode::terrain_triangle_density:
            return render::mesh_visualization_mode::terrain_triangle_density;
        case host_visualization_mode::terrain_bounds:
            return render::mesh_visualization_mode::terrain_bounds;
        case host_visualization_mode::hzb_minimum_depth:
            return render::mesh_visualization_mode::hzb_minimum_depth;
        case host_visualization_mode::hzb_maximum_depth:
            return render::mesh_visualization_mode::hzb_maximum_depth;
        case host_visualization_mode::motion_vectors:
            return render::mesh_visualization_mode::motion_vectors;
        case host_visualization_mode::temporal_reactive_mask:
            return render::mesh_visualization_mode::temporal_reactive_mask;
        case host_visualization_mode::temporal_disocclusion:
            return render::mesh_visualization_mode::temporal_disocclusion;
        case host_visualization_mode::temporal_confidence:
            return render::mesh_visualization_mode::temporal_confidence;
        case host_visualization_mode::temporal_rejection:
            return render::mesh_visualization_mode::temporal_rejection;
        case host_visualization_mode::temporal_sample_weight:
            return render::mesh_visualization_mode::temporal_sample_weight;
        case host_visualization_mode::texture_desired_mip:
            return render::mesh_visualization_mode::texture_desired_mip;
        case host_visualization_mode::texture_resident_mip:
            return render::mesh_visualization_mode::texture_resident_mip;
        case host_visualization_mode::virtual_texture_page_residency:
            return render::mesh_visualization_mode::virtual_texture_page_residency;
        case host_visualization_mode::virtual_texture_recent_requests:
            return render::mesh_visualization_mode::virtual_texture_recent_requests;
        case host_visualization_mode::standard:
            break;
    }
    return render::mesh_visualization_mode::standard;
}

render::editor_overlay_mode to_overlay(host_overlay_mode mode) noexcept
{
    switch (mode)
    {
        case host_overlay_mode::none:
            return render::editor_overlay_mode::none;
        case host_overlay_mode::all_wireframe:
            return render::editor_overlay_mode::all_wireframe;
        case host_overlay_mode::selected_wireframe:
            break;
    }
    return render::editor_overlay_mode::selected_wireframe;
}

editor_tool to_editor_tool(host_viewport_tool tool) noexcept
{
    switch (tool)
    {
        case host_viewport_tool::translate:
            return editor_tool::translate;
        case host_viewport_tool::rotate:
            return editor_tool::rotate;
        case host_viewport_tool::scale:
            return editor_tool::scale;
        case host_viewport_tool::select:
            return editor_tool::select;
        case host_viewport_tool::terrain:
            return editor_tool::select;
    }
    return editor_tool::select;
}

scene::scene_render_visibility to_scene_visibility(host_environment_visibility visibility) noexcept
{
    return {.sky = visibility.sky,
            .fog = visibility.fog,
            .terrain = visibility.terrain,
            .water = visibility.water,
            .vegetation = visibility.vegetation,
            .decals = visibility.decals};
}

scene::world_environment_preset to_scene_preset(host_world_environment_preset preset) noexcept
{
    switch (preset)
    {
        case host_world_environment_preset::clear_day:
            return scene::world_environment_preset::clear_day;
        case host_world_environment_preset::golden_hour:
            return scene::world_environment_preset::golden_hour;
        case host_world_environment_preset::overcast:
            return scene::world_environment_preset::overcast;
        case host_world_environment_preset::night:
            return scene::world_environment_preset::night;
        case host_world_environment_preset::indoor_neutral:
            return scene::world_environment_preset::indoor_neutral;
        case host_world_environment_preset::alpine_late_morning:
            break;
    }
    return scene::world_environment_preset::alpine_late_morning;
}

std::string validation_message(const scene::environment_validation_result& validation)
{
    std::ostringstream message;
    message << "Invalid world environment";
    for (const auto& error : validation.errors)
        message << "; " << error;
    return message.str();
}

void remove_entity_ref(std::vector<ecs::entity>& entities, ecs::entity entity)
{
    entities.erase(std::remove(entities.begin(), entities.end(), entity), entities.end());
}

void forget_entity(editor_scene_state& scene, ecs::entity entity)
{
    if (scene.camera_entity == entity) scene.camera_entity = {};
    if (scene.game_camera_entity == entity) scene.game_camera_entity = {};
    if (scene.sun_entity == entity) scene.sun_entity = {};
    if (scene.world_environment_entity == entity) scene.world_environment_entity = {};
    if (scene.mesh_entity == entity) scene.mesh_entity = {};
    if (scene.terrain_entity == entity) scene.terrain_entity = {};
    if (scene.water_entity == entity) scene.water_entity = {};
    if (scene.vegetation_entity == entity) scene.vegetation_entity = {};
    if (scene.selected_entity == entity) scene.selected_entity = {};
    remove_entity_ref(scene.primitive_entities, entity);
    remove_entity_ref(scene.imported_scene_entities, entity);
    remove_entity_ref(scene.world_feature_entities, entity);
}

void synchronize_all_terrain_resources(editor_scene_state& state, render::renderer& renderer)
{
    for (const auto entity : state.scene.entities())
        if (state.scene.has<scene::terrain_component>(entity))
            synchronize_terrain_render_resource(state, renderer, entity);
}

std::string entity_name(const editor_scene_state& state, ecs::entity entity, const char* fallback)
{
    if (const auto* name = state.scene.try_get<scene::name_component>(entity)) return name->value;
    return fallback;
}

bool entity_active(const editor_scene_state& state, ecs::entity entity)
{
    const auto* active = state.scene.try_get<scene::active_component>(entity);
    return !active || active->active;
}

std::vector<ecs::entity> edit_targets(ecs::world& world, ecs::entity primary, bool apply_to_selection)
{
    if (!world.alive(primary)) return {};
    if (!apply_to_selection) return {primary};
    std::vector<ecs::entity> targets;
    world.view<scene::selection_component>().each(
        [&](ecs::entity entity, const scene::selection_component& selection)
        {
            if (selection.selected) targets.push_back(entity);
        });
    if (std::find(targets.begin(), targets.end(), primary) == targets.end()) return {};
    std::sort(targets.begin(), targets.end(),
              [](ecs::entity left, ecs::entity right)
              {
                  if (left.index != right.index) return left.index < right.index;
                  return left.generation < right.generation;
              });
    return targets;
}

void push_event(std::vector<host_event>& events, std::uint64_t& sequence, host_event_type type, std::string message,
                ecs::entity entity = {}, std::string payload_json = {})
{
    events.push_back({.sequence = ++sequence,
                      .event_type = type,
                      .entity = to_host_entity(entity),
                      .message = std::move(message),
                      .payload_json = std::move(payload_json)});
}

template <class Component>
void add_component_snapshot(const ecs::world& world, ecs::entity entity,
                            std::vector<host_component_snapshot>& components, host_component_kind kind,
                            const char* label)
{
    const auto type = ecs::component_type<Component>();
    const auto change = world.component_change_for(entity, type);
    components.push_back({.kind = kind,
                          .type_id = ecs::to_string(type),
                          .label = label,
                          .revision = change.revision,
                          .dirty_fields = change.fields,
                          .editable = true});
}

std::string asset_relative_path(const std::filesystem::path& root, const std::filesystem::path& path)
{
    if (path.empty()) return {};
    std::error_code error;
    const auto relative = std::filesystem::relative(path, root, error);
    return (error ? path.lexically_normal() : relative.lexically_normal()).generic_string();
}

const char* reflected_kind_name(ecs::reflected_field_kind kind) noexcept
{
    switch (kind)
    {
        case ecs::reflected_field_kind::boolean:
            return "boolean";
        case ecs::reflected_field_kind::signed_integer:
            return "integer";
        case ecs::reflected_field_kind::unsigned_integer:
            return "unsignedInteger";
        case ecs::reflected_field_kind::floating_point:
            return "number";
        case ecs::reflected_field_kind::string:
            return "string";
        case ecs::reflected_field_kind::vector2:
            return "vector2";
        case ecs::reflected_field_kind::vector3:
            return "vector3";
        case ecs::reflected_field_kind::vector4:
            return "vector4";
        case ecs::reflected_field_kind::quaternion:
            return "quaternion";
        case ecs::reflected_field_kind::matrix:
            return "matrix";
        case ecs::reflected_field_kind::enumeration:
            return "enum";
        case ecs::reflected_field_kind::entity_reference:
            return "entity";
        case ecs::reflected_field_kind::asset_reference:
            return "asset";
        case ecs::reflected_field_kind::structure:
            return "struct";
        case ecs::reflected_field_kind::sequence:
            return "sequence";
        case ecs::reflected_field_kind::unknown:
            break;
    }
    return "unknown";
}

template <class Component> void append_component_schema(std::string& json, bool& first)
{
    const auto& descriptor = ecs::component_metadata<Component>();
    if (!first) json += ',';
    first = false;
    json += "{\"id\":" + to_json_string(ecs::to_string(descriptor.id)) +
            ",\"name\":" + to_json_string(descriptor.canonical_name) +
            ",\"displayName\":" + to_json_string(descriptor.display_name) +
            ",\"schemaVersion\":" + std::to_string(descriptor.schema_version) +
            ",\"customSerialization\":" + std::string(descriptor.custom_serialization ? "true" : "false") +
            ",\"fields\":[";
    for (std::size_t index = 0; index < descriptor.fields.size(); ++index)
    {
        if (index != 0) json += ',';
        const auto& field = descriptor.fields[index];
        json += "{\"id\":" +
                to_json_string(
                    [&]
                    {
                        std::ostringstream value;
                        value << std::hex << std::setfill('0') << std::setw(16) << field.id;
                        return value.str();
                    }()) +
                ",\"name\":" + to_json_string(field.name) + ",\"displayName\":" + to_json_string(field.display_name) +
                ",\"kind\":" + to_json_string(reflected_kind_name(field.kind)) + ",\"editable\":" +
                std::string(ecs::has_flag(field.flags, ecs::reflected_field_flags::editable) ? "true" : "false") +
                ",\"transient\":" +
                std::string(ecs::has_flag(field.flags, ecs::reflected_field_flags::transient) ? "true" : "false") + '}';
    }
    json += "]}";
}

std::string_view reflected_kind_name(project::game_field_kind_v1 kind)
{
    using enum project::game_field_kind_v1;
    switch (kind)
    {
        case boolean:
            return "boolean";
        case signed_integer:
            return "signedInteger";
        case unsigned_integer:
            return "unsignedInteger";
        case floating_point:
            return "number";
        case string:
            return "string";
        case enumeration:
            return "enum";
        case vector2:
            return "vector2";
        case vector3:
            return "vector3";
        case vector4:
            return "vector4";
        case quaternion:
            return "quaternion";
        case entity_reference:
            return "entity";
        case asset_reference:
            return "asset";
        case structure:
            return "struct";
        case sequence:
            return "sequence";
    }
    return "unknown";
}

bool has_flag(project::game_field_flags_v1 value, project::game_field_flags_v1 flag)
{
    return (static_cast<std::uint32_t>(value) & static_cast<std::uint32_t>(flag)) != 0;
}

void append_project_component_schema(std::string& json, bool& first, const project_component_schema& component)
{
    if (!first) json += ',';
    first = false;
    json += "{\"id\":" + to_json_string(component.stable_id) + ",\"name\":" + to_json_string(component.canonical_name) +
            ",\"displayName\":" + to_json_string(component.display_name) +
            ",\"category\":" + to_json_string(component.category) +
            ",\"tooltip\":" + to_json_string(component.tooltip) +
            ",\"schemaVersion\":" + std::to_string(component.schema_version) +
            ",\"projectComponent\":true,\"customSerialization\":false,\"fields\":[";
    for (std::size_t index = 0; index < component.fields.size(); ++index)
    {
        if (index) json += ',';
        const auto& field = component.fields[index];
        std::ostringstream id;
        id << std::hex << std::setfill('0') << std::setw(16) << field.stable_id;
        json += "{\"id\":" + to_json_string(id.str()) + ",\"name\":" + to_json_string(field.name) +
                ",\"displayName\":" + to_json_string(field.display_name) +
                ",\"category\":" + to_json_string(field.category) + ",\"tooltip\":" + to_json_string(field.tooltip) +
                ",\"kind\":" + to_json_string(reflected_kind_name(field.kind)) + ",\"editable\":" +
                std::string(has_flag(field.flags, project::game_field_flags_v1::editable) &&
                                    !has_flag(field.flags, project::game_field_flags_v1::read_only)
                                ? "true"
                                : "false") +
                ",\"readOnly\":" +
                std::string(has_flag(field.flags, project::game_field_flags_v1::read_only) ? "true" : "false") +
                ",\"transient\":" +
                std::string(has_flag(field.flags, project::game_field_flags_v1::transient) ? "true" : "false") +
                ",\"saveGame\":" +
                std::string(has_flag(field.flags, project::game_field_flags_v1::save_game) ? "true" : "false") +
                ",\"prefabOverride\":" +
                std::string(has_flag(field.flags, project::game_field_flags_v1::prefab_override) ? "true" : "false") +
                ",\"replicated\":" +
                std::string(has_flag(field.flags, project::game_field_flags_v1::replicated) ? "true" : "false") +
                ",\"default\":" + field.default_json;
        if (field.has_minimum) json += ",\"minimum\":" + std::to_string(field.minimum);
        if (field.has_maximum) json += ",\"maximum\":" + std::to_string(field.maximum);
        if (!field.asset_type_restriction.empty())
            json += ",\"assetType\":" + to_json_string(field.asset_type_restriction);
        if (!field.entity_component_restriction.empty())
            json += ",\"entityComponent\":" + to_json_string(field.entity_component_restriction);
        json += '}';
    }
    json += "]}";
}

std::string component_schema_json(const std::vector<project_component_schema>& project_components)
{
    std::string json = "{\"schemaVersion\":1,\"components\":[";
    bool first = true;
    append_component_schema<scene::name_component>(json, first);
    append_component_schema<scene::transform_component>(json, first);
    append_component_schema<scene::tag_component>(json, first);
    append_component_schema<scene::active_component>(json, first);
    append_component_schema<scene::bounds_component>(json, first);
    append_component_schema<scene::camera_component>(json, first);
    append_component_schema<scene::mesh_renderer_component>(json, first);
    append_component_schema<scene::render_layer_component>(json, first);
    append_component_schema<scene::mobility_component>(json, first);
    append_component_schema<scene::directional_light_component>(json, first);
    append_component_schema<scene::point_light_component>(json, first);
    append_component_schema<scene::spot_light_component>(json, first);
    append_component_schema<scene::area_light_component>(json, first);
    append_component_schema<scene::world_environment_component>(json, first);
    append_component_schema<scene::sky_atmosphere_component>(json, first);
    append_component_schema<scene::celestial_sky_component>(json, first);
    append_component_schema<scene::cloud_layers_component>(json, first);
    append_component_schema<scene::environment_lighting_component>(json, first);
    append_component_schema<scene::height_fog_component>(json, first);
    append_component_schema<scene::terrain_component>(json, first);
    append_component_schema<scene::water_component>(json, first);
    append_component_schema<scene::vegetation_component>(json, first);
    append_component_schema<scene::decal_component>(json, first);
    for (const auto& component : project_components)
        append_project_component_schema(json, first, component);
    json += "]}";
    return json;
}

const project_component_schema* find_project_component(const project_module_loader& module, std::string_view identity)
{
    const auto& components = module.component_schemas();
    const auto found =
        std::find_if(components.begin(), components.end(), [&](const auto& component)
                     { return component.stable_id == identity || component.canonical_name == identity; });
    return found == components.end() ? nullptr : &*found;
}

nlohmann::json project_component_records(const editor_scene_state& scene, ecs::entity_guid guid)
{
    const auto record = std::find_if(scene.unknown_component_records.begin(), scene.unknown_component_records.end(),
                                     [guid](const auto& candidate) { return candidate.first == guid; });
    if (record == scene.unknown_component_records.end()) return nlohmann::json::object();
    auto parsed = nlohmann::json::parse(record->second, nullptr, false);
    return parsed.is_object() ? std::move(parsed) : nlohmann::json::object();
}

void set_project_component_records(editor_scene_state& scene, ecs::entity_guid guid, const nlohmann::json& records)
{
    const auto record = std::find_if(scene.unknown_component_records.begin(), scene.unknown_component_records.end(),
                                     [guid](const auto& candidate) { return candidate.first == guid; });
    if (records.empty())
    {
        if (record != scene.unknown_component_records.end()) scene.unknown_component_records.erase(record);
        return;
    }
    if (record == scene.unknown_component_records.end())
        scene.unknown_component_records.emplace_back(guid, records.dump());
    else
        record->second = records.dump();
}

nlohmann::json default_project_component(const project_component_schema& schema)
{
    nlohmann::json component{
        {"typeId", schema.stable_id}, {"version", schema.schema_version}, {"_arcFieldIds", nlohmann::json::object()}};
    for (const auto& field : schema.fields)
    {
        std::ostringstream id;
        id << std::hex << std::setfill('0') << std::setw(16) << field.stable_id;
        component["_arcFieldIds"][id.str()] = field.name;
        if (has_flag(field.flags, project::game_field_flags_v1::transient)) continue;
        auto value = nlohmann::json::parse(field.default_json, nullptr, false);
        if (!value.is_discarded()) component[field.name] = std::move(value);
    }
    return component;
}

std::optional<std::string> project_component_record_key(const nlohmann::json& records,
                                                        const project_component_schema& schema)
{
    if (records.contains(schema.canonical_name) && records.at(schema.canonical_name).is_object())
        return schema.canonical_name;
    for (const auto& [name, component] : records.items())
    {
        if (component.is_object() && component.value("typeId", std::string{}) == schema.stable_id) return name;
    }
    return std::nullopt;
}

void migrate_project_component_records(editor_scene_state& scene, const std::vector<project_component_schema>& schemas)
{
    for (auto& [_, record_text] : scene.unknown_component_records)
    {
        auto records = nlohmann::json::parse(record_text, nullptr, false);
        if (!records.is_object()) continue;
        bool changed{};
        for (const auto& schema : schemas)
        {
            const auto record_key = project_component_record_key(records, schema);
            if (!record_key) continue;
            if (*record_key != schema.canonical_name)
            {
                records[schema.canonical_name] = std::move(records[*record_key]);
                records.erase(*record_key);
                changed = true;
            }
            auto& component = records[schema.canonical_name];
            const auto defaults = default_project_component(schema);
            const auto previous_ids = component.value("_arcFieldIds", nlohmann::json::object());
            for (const auto& field : schema.fields)
            {
                std::ostringstream id;
                id << std::hex << std::setfill('0') << std::setw(16) << field.stable_id;
                if (previous_ids.contains(id.str()) && previous_ids[id.str()].is_string())
                {
                    const auto previous_name = previous_ids[id.str()].get<std::string>();
                    if (previous_name != field.name && component.contains(previous_name) &&
                        !component.contains(field.name))
                    {
                        component[field.name] = std::move(component[previous_name]);
                        component.erase(previous_name);
                        changed = true;
                    }
                }
                if (!component.contains(field.name) && defaults.contains(field.name))
                {
                    component[field.name] = defaults[field.name];
                    changed = true;
                }
            }
            component["_arcFieldIds"] = defaults["_arcFieldIds"];
            component["typeId"] = schema.stable_id;
            component["version"] = schema.schema_version;
            changed = true;
        }
        if (changed) record_text = records.dump();
    }
}

bool validate_project_field_value(const project_field_schema& field, const nlohmann::json& value, std::string& error)
{
    using enum project::game_field_kind_v1;
    const bool type_matches =
        (field.kind == boolean && value.is_boolean()) ||
        ((field.kind == signed_integer || field.kind == unsigned_integer) && value.is_number_integer()) ||
        (field.kind == floating_point && value.is_number()) ||
        ((field.kind == string || field.kind == enumeration || field.kind == entity_reference ||
          field.kind == asset_reference) &&
         (value.is_string() || value.is_null())) ||
        ((field.kind == vector2 || field.kind == vector3 || field.kind == vector4 || field.kind == quaternion) &&
         value.is_array()) ||
        (field.kind == structure && value.is_object()) || (field.kind == sequence && value.is_array());
    if (!type_matches)
    {
        error = "Project component field value has the wrong reflected type";
        return false;
    }
    if (value.is_number())
    {
        const double number = value.get<double>();
        if (!std::isfinite(number) || (field.kind == unsigned_integer && number < 0.0) ||
            (field.has_minimum && number < field.minimum) || (field.has_maximum && number > field.maximum))
        {
            error = "Project component field value is outside its authored range";
            return false;
        }
    }
    const auto expected_elements = field.kind == vector2                                 ? 2u
                                   : field.kind == vector3                               ? 3u
                                   : (field.kind == vector4 || field.kind == quaternion) ? 4u
                                                                                         : 0u;
    if (expected_elements && (value.size() != expected_elements ||
                              !std::all_of(value.begin(), value.end(), [](const auto& item)
                                           { return item.is_number() && std::isfinite(item.template get<double>()); })))
    {
        error = "Project vector field has an invalid element count or non-finite value";
        return false;
    }
    return true;
}

host_mesh_renderer_snapshot mesh_renderer_snapshot(const editor_scene_state& state, const editor_asset_state& assets,
                                                   const std::filesystem::path& project_root, ecs::entity entity,
                                                   const scene::mesh_renderer_component& mesh_renderer)
{
    host_mesh_renderer_snapshot snapshot;
    snapshot.representation = static_cast<std::uint8_t>(mesh_renderer.representation);
    snapshot.visible = mesh_renderer.visible;
    snapshot.casts_shadows = mesh_renderer.casts_shadows;
    snapshot.receives_shadows = mesh_renderer.receives_shadows;
    snapshot.shadow_lod_bias = mesh_renderer.shadow_lod_bias;
    snapshot.maximum_shadow_distance = mesh_renderer.maximum_shadow_distance;
    snapshot.base_color_tint = to_host_vec4(mesh_renderer.base_color_tint);
    snapshot.has_material = mesh_renderer.material.valid();
    for (const auto& record : state.material_library.materials)
    {
        if (record.material != mesh_renderer.material) continue;
        snapshot.asset_backed_material = true;
        snapshot.material_name = record.asset.name;
        const auto relative_to_primary = record.path.lexically_relative(assets.root);
        const auto relative_to_project = record.path.lexically_relative(project_root);
        const auto stays_within = [](const std::filesystem::path& relative)
        { return !relative.empty() && *relative.begin() != std::filesystem::path(".."); };
        if (!assets.root.empty() && stays_within(relative_to_primary))
            snapshot.material_path = arc::assets::normalize_asset_path(relative_to_primary);
        else if (!project_root.empty() && stays_within(relative_to_project))
            snapshot.material_path = arc::assets::normalize_asset_path(relative_to_project);
        else if (const auto* binding = find_asset_binding(state, entity_guid_of(state, entity));
                 binding && !binding->material.path_hint.empty())
            snapshot.material_path = arc::assets::normalize_asset_path(binding->material.path_hint);
        else
            snapshot.material_path = asset_relative_path(assets.root, record.path);
        return snapshot;
    }
    if (mesh_renderer.material == state.default_material)
        snapshot.material_name = "Default Mesh Material";
    else if (mesh_renderer.material == state.primitive_material)
        snapshot.material_name = "Primitive Material";
    else if (mesh_renderer.material == state.terrain_material)
        snapshot.material_name = "Terrain Material";
    else if (mesh_renderer.material == state.water_material)
        snapshot.material_name = "Water Material";
    else if (mesh_renderer.material == state.vegetation_material)
        snapshot.material_name = "Vegetation Material";
    else if (mesh_renderer.material.valid())
        snapshot.material_name = "Embedded Material";
    return snapshot;
}

ecs::entity add_default_sky_to_scene(editor_scene_state& state)
{
    const auto sky = state.scene.create();
    state.sun_entity = sky;
    state.world_environment_entity = sky;

    scene::transform_component transform;
    transform.position = {0.0f, 5.0f, 0.0f};
    transform.rotation = math::normalize(math::quatf{-0.258819f, 0.0f, 0.0f, 0.965926f});
    scene::directional_light_component light;
    light.color = {1.0f, 0.95f, 0.88f};
    light.intensity = 85000.0f;
    light.casts_shadows = true;
    light.use_color_temperature = true;
    light.temperature_kelvin = 5500.0f;
    light.shadow.enabled = true;

    state.scene.emplace<scene::persistent_id_component>(sky, ecs::generate_entity_guid());
    state.scene.emplace<scene::hierarchy_component>(sky);
    state.scene.emplace<scene::name_component>(sky, "Default Sky");
    state.scene.emplace<scene::tag_component>(sky, "Environment");
    state.scene.emplace<scene::active_component>(sky);
    state.scene.emplace<scene::selection_component>(sky, false);
    state.scene.emplace<scene::transform_component>(sky, transform);
    state.scene.emplace<scene::directional_light_component>(sky, light);

    scene::world_environment_settings settings;
    scene::apply_world_environment_preset(scene::world_environment_preset::alpine_late_morning, settings);
    settings.celestial.sun_mode = scene::sun_position_mode::manual_light;
    settings.celestial.sun_light = sky;
    settings.celestial.automatic_sun_light = false;
    scene::set_world_environment_settings(state.scene, sky, settings);
    state.world_feature_entities.push_back(sky);
    return sky;
}

editor_scene_state create_blank_scene(render::renderer& renderer, bool include_floor = true,
                                      const editor_asset_state* editor_assets = nullptr)
{
    editor_scene_state state;
    const auto editor_camera = state.scene.create();
    state.camera_entity = editor_camera;
    scene::transform_component editor_transform;
    editor_transform.position = defaults::default_camera_position;
    state.scene.emplace<scene::name_component>(editor_camera, "Editor Camera");
    state.scene.emplace<scene::tag_component>(editor_camera, "EditorOnly");
    state.scene.emplace<scene::active_component>(editor_camera);
    state.scene.emplace<scene::transform_component>(editor_camera, editor_transform);
    scene::camera_component editor_camera_settings;
    editor_camera_settings.near_plane = defaults::default_camera_near_plane;
    editor_camera_settings.far_plane = defaults::default_camera_far_plane;
    state.scene.emplace<scene::camera_component>(editor_camera, editor_camera_settings);

    const auto game_camera = state.scene.create();
    state.game_camera_entity = game_camera;
    scene::transform_component game_transform;
    game_transform.position = {0.0f, 2.0f, 6.0f};
    scene::camera_component game_camera_settings;
    game_camera_settings.near_plane = 0.1f;
    game_camera_settings.far_plane = 2000.0f;
    state.scene.emplace<scene::name_component>(game_camera, "Main Camera");
    state.scene.emplace<scene::tag_component>(game_camera, "Camera");
    state.scene.emplace<scene::active_component>(game_camera);
    state.scene.emplace<scene::transform_component>(game_camera, game_transform);
    state.scene.emplace<scene::camera_component>(game_camera, game_camera_settings);

    add_default_sky_to_scene(state);
    if (editor_assets != nullptr) create_default_primitive_material(state, renderer, *editor_assets);
    if (include_floor)
    {
        const auto floor = add_default_floor_to_scene(state, renderer);
        if (state.scene.alive(floor))
        {
            state.scene.get<scene::name_component>(floor).value = "Floor";
            state.scene.get<scene::tag_component>(floor).value = "Environment";
            auto& floor_transform = state.scene.get<scene::transform_component>(floor);
            floor_transform.position = {0.0f, -0.02f, 0.0f};
            floor_transform.scale = {10.0f, 1.0f, 10.0f};
            auto& floor_mesh = state.scene.get<scene::mesh_renderer_component>(floor);
            floor_mesh.base_color_tint = {0.48f, 0.5f, 0.53f, 1.0f};
            floor_mesh.casts_shadows = false;
            floor_mesh.receives_shadows = true;
        }
    }
    select_entity(state.scene, game_camera, state.selected_entity);
    ensure_scene_authoring_metadata(state);
    return state;
}

void reset_editor_camera(editor_scene_state& scene, editor_camera_controller& controller)
{
    controller = {};
    controller.focus(defaults::default_camera_focus, defaults::default_camera_focus_radius);
    controller.orbit(defaults::default_camera_orbit_x, defaults::default_camera_orbit_y);
    if (auto* camera_transform = scene.scene.try_get<scene::transform_component>(scene.camera_entity))
        controller.apply_to(*camera_transform);
}

} // namespace

class editor_preview_application final : public framework::application
{
public:
    framework::application_config configure() const override
    {
        framework::application_config config{};
        config.title = "ARC Editor Preview Runtime";
        config.visible = false;
        config.simulation.snapshot_budget_bytes = 64u * 1024u * 1024u;
        config.simulation.presentation_enabled = false;
        return config;
    }

    void register_worlds(framework::runtime_world_manager& worlds) override
    {
        worlds.create({.name = "Editor Preview World",
                       .role = framework::runtime_world_role::editor_preview,
                       .seed = 0x4152435f45444954ull,
                       .presentation_enabled = false});
    }
};

struct arc_host::state
{
    explicit state(std::unique_ptr<render::renderer> renderer_value)
        : renderer(std::move(renderer_value)), simulation(simulation_application)
    {
        // Viewport input can arrive while the project browser is transitioning
        // into the workbench. Keep an editor-only camera available for the
        // complete host lifetime rather than creating it only in project.open.
        scene = create_blank_scene(*renderer);
        reset_editor_camera(scene, camera_controller);
        simulation.start();
        simulation.pause();
        const auto worlds = simulation.worlds().ordered_worlds();
        if (!worlds.empty())
        {
            if (framework::runtime_world* world = simulation.worlds().find(worlds.front()))
                world->attach_entities(scene.scene);
        }
    }

    std::unique_ptr<render::renderer> renderer;
    editor_preview_application simulation_application;
    framework::runtime simulation;
    std::unique_ptr<io::async_file_service> asset_files;
    std::unique_ptr<arc::assets::asset_manager> asset_registry;
    std::unique_ptr<arc::assets::derived_data_cache> derived_cache;
    project_module_loader project_module;
    std::uint64_t asset_event_cursor{};
    std::uint64_t runtime_revision{1};
    std::uint64_t last_runtime_tick_event{};
    bool preview_stopped{true};
    editor_project_state project;
    editor_asset_state assets;
    editor_scene_state scene;
    editor_camera_controller camera_controller;
    host_viewport_request viewport_options;
    std::string active_viewport_id{"viewport-1"};
    std::chrono::steady_clock::time_point last_viewport_frame_time{};
    double viewport_fps{};
    double viewport_frame_ms{};
    std::uint32_t viewport_draw_calls{};
    std::uint64_t viewport_frame_index{};
    bool viewport_submitted{};
    struct pending_viewport_pick
    {
        std::uint64_t request_id{};
        std::uint32_t x{};
        std::uint32_t y{};
        editor_pick_result cpu_fallback{};
        std::uint64_t requested_after_frame{};
    };
    std::optional<pending_viewport_pick> pending_pick;
    std::uint64_t next_pick_request_id{1};
    editor_history history;
    host_viewport_set_tool_command viewport_tool;
    scene::terrain_brush_settings terrain_brush;
    bool terrain_flatten_height_captured{};
    std::optional<math::vector3f> terrain_brush_local_position;
    std::optional<math::vector3f> terrain_stroke_previous_position;
    host_terrain_operation_snapshot terrain_operation;
    std::uint64_t next_terrain_operation_id{1};
    gizmo_axis gizmo_highlight{gizmo_axis::none};
    std::vector<host_event> events;
    std::uint64_t event_sequence{};
    std::uint64_t scene_revision{1};
    std::uint64_t world_epoch{1};
    bool project_open{};
    bool project_read_only{};
};

arc_host::arc_host(std::unique_ptr<render::renderer> renderer) : state_(std::make_unique<state>(std::move(renderer)))
{
    arc::diagnostics::info("editor.host", "Arc Host started");
    push_event(state_->events, state_->event_sequence, host_event_type::host_started, "Arc Host started");
}

arc_host::~arc_host()
{
    if (state_)
    {
        state_->project_module.unload();
        arc::diagnostics::info("editor.host", "Arc Host shutdown");
        push_event(state_->events, state_->event_sequence, host_event_type::host_shutdown, "Arc Host shutdown");
        if (state_->asset_registry)
        {
            framework::runtime_service_context context(state_->simulation.services());
            state_->asset_registry->on_shutdown(context);
            state_->asset_registry.reset();
            state_->derived_cache.reset();
            state_->asset_files.reset();
        }
        state_->simulation.shutdown();
    }
}

host_response arc_host::open_project(const host_open_project_command& command, const editor_asset_state& assets,
                                     std::uint64_t request_id)
{
    state_->project_module.unload();
    if (!command.read_only && !command.editor_module_path.empty())
    {
        const auto module_path = resolve_project_document(command.root, command.editor_module_path, true);
        if (!module_path)
            return {.request_id = request_id,
                    .succeeded = false,
                    .error = "Project editor module must be a file contained by the project root"};
        const auto module_result = state_->project_module.load(*module_path, command.engine_version,
                                                               command.project_guid, command.editor_module_id);
        if (!module_result.succeeded)
            return {.request_id = request_id, .succeeded = false, .error = module_result.message};
    }
    state_->assets = assets;
    state_->project.name = command.name.empty() ? "Arc Project" : command.name;
    state_->project.root = command.root;
    if (state_->asset_registry)
    {
        framework::runtime_service_context context(state_->simulation.services());
        state_->asset_registry->on_shutdown(context);
    }
    state_->asset_registry.reset();
    state_->derived_cache.reset();
    state_->asset_files.reset();
    if (!command.root.empty())
    {
        const auto cache_root =
            command.cache_root.empty() ? command.root / "Intermediate" / "Cache" : command.cache_root;
        std::vector<std::filesystem::path> content_roots = command.content_roots;
        if (content_roots.empty())
            content_roots.push_back(assets.root.empty() ? command.root / "Content" : assets.root);
        state_->derived_cache = std::make_unique<arc::assets::derived_data_cache>(
            arc::assets::derived_data_cache_config{.root = cache_root});
        state_->asset_files = std::make_unique<io::async_file_service>(state_->simulation.jobs());
        state_->asset_registry = std::make_unique<arc::assets::asset_manager>(
            arc::assets::asset_manager_config{.project_root = command.root,
                                              .asset_root = content_roots.front(),
                                              .additional_source_roots = std::vector<std::filesystem::path>(
                                                  content_roots.begin() + 1, content_roots.end()),
                                              .read_only_source_roots = command.builtin_content_roots,
                                              .cache_root = cache_root},
            state_->simulation.jobs(), *state_->asset_files, state_->simulation.memory());
        framework::runtime_service_context context(state_->simulation.services());
        state_->asset_registry->on_start(context);
    }
    state_->asset_event_cursor = 0;
    // Project opening never synthesizes sample content. Rendering samples are
    // ordinary persisted scenes supplied by the selected project template.
    state_->scene.terrain_render_proxies.clear(*state_->renderer);
    state_->scene = create_blank_scene(*state_->renderer, command.default_scene.empty(), &state_->assets);
    if (state_->asset_registry)
    {
        register_editor_asset_fallbacks(state_->scene, *state_->asset_registry);
        resolve_editor_asset_bindings(state_->scene, *state_->asset_registry, state_->project.root);
    }
    std::string loaded_scene_message = "Blank authoring scene created";
    if (!command.default_scene.empty())
    {
        const auto scene_path =
            command.default_scene.is_absolute() ? command.default_scene : command.root / command.default_scene;
        const auto loaded = load_scene_document(state_->scene, *state_->renderer, state_->project.root, scene_path,
                                                state_->asset_registry.get());
        if (!loaded.succeeded)
        {
            state_->project_module.unload();
            return {.request_id = request_id, .succeeded = false, .error = loaded.message};
        }
        loaded_scene_message = "Project default scene loaded";
    }
    ++state_->scene_revision;
    ++state_->world_epoch;
    state_->history.clear(state_->scene, false);
    reset_editor_camera(state_->scene, state_->camera_controller);
    state_->project_open = true;
    state_->project_read_only = command.read_only;

    const std::string message = "Opened project '" + state_->project.name + "'";
    arc::diagnostics::info("editor.host", message);
    push_event(state_->events, state_->event_sequence, host_event_type::project_opened, message);
    push_event(state_->events, state_->event_sequence, host_event_type::scene_changed, loaded_scene_message);
    return {.request_id = request_id,
            .succeeded = true,
            .payload_json = "{\"entity\":" + to_json(to_host_entity(state_->scene.selected_entity)) + '}'};
}

host_response arc_host::execute(host_command_payload command)
{
    return execute({.command_type = command_type(command), .payload = std::move(command)});
}

host_response arc_host::execute(const host_command_envelope& command)
{
    const bool authoring = std::visit(
        [](const auto& payload) { return is_authoring_command<std::decay_t<decltype(payload)>>(); }, command.payload);
    const bool project_mutation = std::visit(
        [](const auto& payload) { return is_project_mutation<std::decay_t<decltype(payload)>>(); }, command.payload);
    if (state_->project_read_only && project_mutation)
        return {.request_id = command.request_id,
                .succeeded = false,
                .error = "The project is open read-only",
                .scene_revision = state_->scene_revision,
                .world_epoch = state_->world_epoch,
                .frame_revision = state_->viewport_frame_index};
    const std::string edit_label =
        command.edit && !command.edit->label.empty() ? command.edit->label : history_label(command.payload);
    if (command.expected_scene_revision && *command.expected_scene_revision != state_->scene_revision)
    {
        return {.request_id = command.request_id,
                .succeeded = false,
                .error = "Scene revision is stale",
                .scene_revision = state_->scene_revision,
                .world_epoch = state_->world_epoch,
                .frame_revision = state_->viewport_frame_index};
    }
    if (command.edit && command.edit->phase == host_edit_phase::cancel)
    {
        if (!state_->history.cancel(command.edit->id, state_->scene))
            return {.request_id = command.request_id,
                    .succeeded = false,
                    .error = "No matching edit transaction to cancel",
                    .scene_revision = state_->scene_revision,
                    .world_epoch = state_->world_epoch,
                    .frame_revision = state_->viewport_frame_index};
        synchronize_all_terrain_resources(state_->scene, *state_->renderer);
        state_->terrain_brush_local_position.reset();
        state_->terrain_stroke_previous_position.reset();
        ++state_->scene_revision;
        push_event(state_->events, state_->event_sequence, host_event_type::scene_changed,
                   "Edit transaction cancelled");
        return {.request_id = command.request_id,
                .succeeded = true,
                .payload_json = "{}",
                .scene_revision = state_->scene_revision,
                .world_epoch = state_->world_epoch,
                .frame_revision = state_->viewport_frame_index};
    }
    if (authoring && command.edit && command.edit->phase == host_edit_phase::begin &&
        !state_->history.begin(command.edit->id, edit_label, state_->scene))
        return {.request_id = command.request_id, .succeeded = false, .error = "Could not begin edit transaction"};
    if (authoring && command.edit &&
        (command.edit->phase == host_edit_phase::update || command.edit->phase == host_edit_phase::commit) &&
        !state_->history.transaction_matches(command.edit->id))
        return {.request_id = command.request_id,
                .succeeded = false,
                .error = "Edit transaction does not match the active transaction"};
    std::optional<editor_scene_state> before;
    if (authoring && !command.edit) before = state_->scene;

    auto response = std::visit(
        [this, request_id = command.request_id](const auto& payload) -> host_response
        {
            using command_type = std::decay_t<decltype(payload)>;
            const auto fail = [this, request_id](std::string message, ecs::entity entity = {})
            {
                arc::diagnostics::warn("editor.host", message);
                push_event(state_->events, state_->event_sequence, host_event_type::command_failed, message, entity);
                return host_response{.request_id = request_id, .succeeded = false, .error = std::move(message)};
            };
            const auto success = [request_id](std::string payload_json = "{}") {
                return host_response{
                    .request_id = request_id, .succeeded = true, .payload_json = std::move(payload_json)};
            };

            if constexpr (std::is_same_v<command_type, host_open_project_command>)
            {
                auto resolved = payload;
                if (!payload.descriptor_path.empty())
                {
                    const auto descriptor = project::load_descriptor(payload.descriptor_path);
                    if (!descriptor) return fail(descriptor.error().message);
                    const auto validation = project::validate_descriptor(payload.descriptor_path, descriptor.value(),
                                                                         {.require_paths = true});
                    if (!validation) return fail(validation.error().message);
                    const auto context = project::resolve_context(payload.descriptor_path, descriptor.value());
                    if (!context) return fail(context.error().message);
                    if (!payload.project_guid.empty() && payload.project_guid != descriptor.value().guid)
                        return fail("Project GUID does not match the selected descriptor");
                    if (!payload.engine_version.empty() && payload.engine_version != descriptor.value().engine_version)
                        return fail("Project engine version does not match the selected descriptor");
                    if (!payload.editor_module_path.empty())
                    {
                        const auto editor_module =
                            std::find_if(descriptor.value().modules.begin(), descriptor.value().modules.end(),
                                         [&](const auto& module) {
                                             return module.enabled && module.kind == project::module_kind::editor &&
                                                    module.id == payload.editor_module_id;
                                         });
                        if (editor_module == descriptor.value().modules.end())
                            return fail("Editor module is not declared by the selected project");
                    }
                    resolved.name = descriptor.value().name;
                    resolved.root = context.value().root;
                    resolved.content_roots = context.value().asset_roots;
                    resolved.cache_root = context.value().asset_cache_root;
                    resolved.default_scene = descriptor.value().default_scene
                                                 ? std::filesystem::path{descriptor.value().default_scene->path_hint}
                                                 : std::filesystem::path{};
                    resolved.project_guid = descriptor.value().guid;
                    resolved.engine_version = descriptor.value().engine_version;
                }
                const auto asset_root =
                    !resolved.content_roots.empty() ? resolved.content_roots.front() : resolved.root / "Content";
                auto editor_assets = load_default_editor_assets(asset_root);
                editor_assets.builtin_roots = resolved.builtin_content_roots;
                return open_project(resolved, std::move(editor_assets), request_id);
            }
            else if constexpr (std::is_same_v<command_type, host_close_project_command>)
            {
                if (!state_->project_open) return success("{\"message\":\"No project is open\"}");

                const std::string message = "Closed project '" + state_->project.name + "'";
                if (state_->asset_registry)
                {
                    framework::runtime_service_context context(state_->simulation.services());
                    state_->asset_registry->on_shutdown(context);
                    state_->asset_registry.reset();
                    state_->derived_cache.reset();
                    state_->asset_files.reset();
                }
                state_->project_module.unload();
                state_->scene.terrain_render_proxies.clear(*state_->renderer);
                state_->scene = create_blank_scene(*state_->renderer);
                reset_editor_camera(state_->scene, state_->camera_controller);
                state_->project = {};
                state_->project_open = false;
                state_->project_read_only = false;
                arc::diagnostics::info("editor.host", message);
                push_event(state_->events, state_->event_sequence, host_event_type::project_closed, message);
                return success();
            }
            else if constexpr (std::is_same_v<command_type, host_reload_project_module_command>)
            {
                if (!state_->project_open) return fail("No project is open");
                if (state_->project_read_only) return fail("Project module reload is disabled in read-only mode");
                const auto module_path = resolve_project_document(state_->project.root, payload.path, true);
                if (!module_path) return fail("Project module must be a file contained by the project root");
                const bool was_running = !state_->simulation.paused();
                if (was_running) state_->simulation.pause();
                const auto result = state_->project_module.reload(*module_path, payload.engine_version,
                                                                  payload.project_guid, payload.module_id);
                if (was_running && result.succeeded &&
                    result.classification == module_reload_classification::safe_hot_reload)
                    state_->simulation.resume();
                const char* classification =
                    result.classification == module_reload_classification::safe_hot_reload ? "safeHotReload"
                    : result.classification == module_reload_classification::play_session_restart_required
                        ? "playSessionRestartRequired"
                    : result.classification == module_reload_classification::native_host_restart_required
                        ? "nativeHostRestartRequired"
                        : "initialLoad";
                if (!result.succeeded)
                {
                    if (result.generation &&
                        result.classification == module_reload_classification::native_host_restart_required)
                    {
                        push_event(state_->events, state_->event_sequence, host_event_type::project_module_reloaded,
                                   result.message);
                        return success("{\"generation\":" + std::to_string(result.generation) +
                                       ",\"classification\":" + to_json_string(classification) +
                                       ",\"message\":" + to_json_string(result.message) + '}');
                    }
                    return fail(result.message);
                }
                migrate_project_component_records(state_->scene, state_->project_module.component_schemas());
                push_event(state_->events, state_->event_sequence, host_event_type::project_module_reloaded,
                           result.message);
                push_event(state_->events, state_->event_sequence, host_event_type::component_changed,
                           "Project component metadata refreshed");
                return success("{\"generation\":" + std::to_string(result.generation) + ",\"classification\":" +
                               to_json_string(classification) + ",\"message\":" + to_json_string(result.message) + '}');
            }
            else if constexpr (std::is_same_v<command_type, host_asset_reimport_command> ||
                               std::is_same_v<command_type, host_set_texture_streaming_mode_command> ||
                               std::is_same_v<command_type, host_patch_texture_settings_command> ||
                               std::is_same_v<command_type, host_asset_cancel_import_command> ||
                               std::is_same_v<command_type, host_asset_move_command> ||
                               std::is_same_v<command_type, host_asset_rename_command>)
            {
                if (!state_->asset_registry) return fail("No project asset registry is active");
                const auto guid = assets::parse_asset_guid(payload.guid);
                if (!guid) return fail("Asset GUID is malformed");

                if constexpr (std::is_same_v<command_type, host_asset_reimport_command>)
                {
                    const auto asset = state_->asset_registry->find(*guid);
                    if (asset && asset->read_only) return fail("Built-in assets are read-only");
                    const auto job = state_->asset_registry->reimport(*guid, assets::asset_streaming_priority::high);
                    if (!job.valid()) return fail("Asset could not be queued for reimport");
                    return success("{\"queued\":true}");
                }
                else if constexpr (std::is_same_v<command_type, host_patch_texture_settings_command>)
                {
                    const auto asset = state_->asset_registry->find(*guid);
                    if (!asset) return fail("Texture asset was not found");
                    if (asset->read_only) return fail("Built-in assets are read-only");
                    if (asset->type != assets::asset_types::texture_2d) return fail("Asset is not a 2D texture");
                    const auto metadata_path = assets::metadata_path_for(state_->project.root / asset->source_path);
                    auto metadata = assets::load_asset_metadata(metadata_path);
                    if (!metadata) return fail(metadata.error().message);
                    auto parsed = render::tools::parse_texture_import_settings(metadata.value().canonical_settings,
                                                                               metadata.value().settings_version);
                    if (!parsed) return fail(parsed.error());
                    auto settings = parsed.value();
                    if (!payload.preset.empty())
                    {
                        const auto preset = render::tools::parse_texture_import_preset(payload.preset);
                        if (!preset) return fail("Unknown texture preset");
                        settings = render::tools::texture_import_settings_for_preset(*preset);
                    }
                    if (!payload.semantic.empty())
                    {
                        const auto semantic = render::tools::parse_texture_semantic(payload.semantic);
                        if (!semantic) return fail("Unknown texture semantic");
                        settings.semantic = *semantic;
                        settings.preset = render::tools::texture_import_preset::custom;
                    }
                    if (!payload.color_space.empty())
                    {
                        const auto color_space = render::tools::parse_texture_color_space(payload.color_space);
                        if (!color_space) return fail("Unknown texture color space");
                        settings.color_space = *color_space;
                        settings.preset = render::tools::texture_import_preset::custom;
                    }
                    if (!payload.streaming_mode.empty())
                    {
                        const auto mode = render::tools::parse_texture_streaming_mode(payload.streaming_mode);
                        if (!mode) return fail("Unknown texture streaming mode");
                        settings.streaming_mode = *mode;
                    }
                    const auto parse_setting = [&](const std::string& source, auto parser, auto& target)
                    {
                        if (source.empty()) return true;
                        const auto value = parser(source);
                        if (!value) return false;
                        target = *value;
                        return true;
                    };
                    if (!parse_setting(payload.compression, render::tools::parse_texture_compression_policy,
                                       settings.compression) ||
                        !parse_setting(payload.power_of_two, render::tools::parse_texture_power_of_two_policy,
                                       settings.power_of_two) ||
                        !parse_setting(payload.min_filter, render::tools::parse_texture_filter_mode,
                                       settings.min_filter) ||
                        !parse_setting(payload.mag_filter, render::tools::parse_texture_filter_mode,
                                       settings.mag_filter) ||
                        !parse_setting(payload.mip_filter, render::tools::parse_texture_mip_filter_mode,
                                       settings.mip_filter) ||
                        !parse_setting(payload.wrap_u, render::tools::parse_texture_address_mode, settings.wrap_u) ||
                        !parse_setting(payload.wrap_v, render::tools::parse_texture_address_mode, settings.wrap_v) ||
                        !parse_setting(payload.mip_generation_filter,
                                       render::tools::parse_texture_mip_generation_filter,
                                       settings.mip_generation_filter) ||
                        !parse_setting(payload.channel_r, render::tools::parse_texture_channel_source,
                                       settings.channel_r) ||
                        !parse_setting(payload.channel_g, render::tools::parse_texture_channel_source,
                                       settings.channel_g) ||
                        !parse_setting(payload.channel_b, render::tools::parse_texture_channel_source,
                                       settings.channel_b) ||
                        !parse_setting(payload.channel_a, render::tools::parse_texture_channel_source,
                                       settings.channel_a))
                        return fail("Unknown texture import policy value");
                    if (payload.max_size) settings.max_size = *payload.max_size;
                    if (payload.anisotropy) settings.anisotropy = *payload.anisotropy;
                    if (payload.lod_bias) settings.lod_bias = *payload.lod_bias;
                    if (payload.minimum_lod) settings.minimum_lod = *payload.minimum_lod;
                    if (payload.maximum_lod) settings.maximum_lod = *payload.maximum_lod;
                    if (payload.alpha_coverage_threshold)
                        settings.alpha_coverage_threshold = *payload.alpha_coverage_threshold;
                    if (payload.mip_sharpen) settings.mip_sharpen = *payload.mip_sharpen;
                    if (payload.deband_strength) settings.deband_strength = *payload.deband_strength;
                    if (payload.dither_mips) settings.dither_mips = *payload.dither_mips;
                    if (payload.deband_mips) settings.deband_mips = *payload.deband_mips;
                    if (payload.brightness) settings.brightness = *payload.brightness;
                    if (payload.gamma) settings.gamma = *payload.gamma;
                    if (payload.contrast) settings.contrast = *payload.contrast;
                    if (payload.saturation) settings.saturation = *payload.saturation;
                    if (payload.vibrance) settings.vibrance = *payload.vibrance;
                    if (payload.tint_r) settings.tint_r = *payload.tint_r;
                    if (payload.tint_g) settings.tint_g = *payload.tint_g;
                    if (payload.tint_b) settings.tint_b = *payload.tint_b;
                    if (payload.input_black) settings.input_black = *payload.input_black;
                    if (payload.input_white) settings.input_white = *payload.input_white;
                    if (payload.output_black) settings.output_black = *payload.output_black;
                    if (payload.output_white) settings.output_white = *payload.output_white;
                    if (payload.invert_r) settings.invert_r = *payload.invert_r;
                    if (payload.invert_g) settings.invert_g = *payload.invert_g;
                    if (payload.invert_b) settings.invert_b = *payload.invert_b;
                    if (payload.invert_a) settings.invert_a = *payload.invert_a;
                    if (payload.curves_enabled) settings.curves_enabled = *payload.curves_enabled;
                    const auto parse_curve_patch = [&](const std::string& source, render::tools::texture_curve& target)
                    {
                        if (source.empty()) return true;
                        const auto parsed = render::tools::parse_texture_curve(source);
                        if (!parsed) return false;
                        target = parsed.value();
                        return true;
                    };
                    if (!parse_curve_patch(payload.curve_master, settings.curve_master) ||
                        !parse_curve_patch(payload.curve_r, settings.curve_r) ||
                        !parse_curve_patch(payload.curve_g, settings.curve_g) ||
                        !parse_curve_patch(payload.curve_b, settings.curve_b) ||
                        !parse_curve_patch(payload.curve_a, settings.curve_a))
                        return fail("Texture curve payload is invalid");
                    if (payload.generate_mips) settings.generate_mips = *payload.generate_mips;
                    if (payload.preserve_alpha_coverage)
                        settings.preserve_alpha_coverage = *payload.preserve_alpha_coverage;
                    if (payload.preset.empty()) settings.preset = render::tools::texture_import_preset::custom;
                    const auto validated = render::tools::parse_texture_import_settings(
                        render::tools::serialize_texture_import_settings(settings),
                        render::tools::texture_import_settings::current_version);
                    if (!validated) return fail(validated.error());
                    settings = validated.value();
                    metadata.value().settings_version = render::tools::texture_import_settings::current_version;
                    metadata.value().canonical_settings = render::tools::serialize_texture_import_settings(settings);
                    const auto saved = assets::save_asset_metadata(metadata_path, metadata.value());
                    if (!saved) return fail(saved.error().message);
                    const auto job = state_->asset_registry->reimport(*guid, assets::asset_streaming_priority::high);
                    if (!job.valid()) return fail("Texture could not be queued for reimport");
                    return success("{\"queued\":true}");
                }
                else if constexpr (std::is_same_v<command_type, host_set_texture_streaming_mode_command>)
                {
                    const auto asset = state_->asset_registry->find(*guid);
                    if (!asset) return fail("Texture asset was not found");
                    if (asset->read_only) return fail("Built-in assets are read-only");
                    if (asset->type != assets::asset_types::texture_2d) return fail("Asset is not a 2D texture");
                    render::tools::texture_import_settings settings;
                    if (payload.mode == "resident")
                        settings.streaming_mode = render::texture_streaming_mode::resident;
                    else if (payload.mode == "streamed_mips")
                        settings.streaming_mode = render::texture_streaming_mode::streamed_mips;
                    else if (payload.mode == "virtual_tiles")
                        settings.streaming_mode = render::texture_streaming_mode::virtual_tiles;
                    else
                        return fail("Unknown texture streaming mode");
                    const auto absolute_source = state_->project.root / asset->source_path;
                    const auto metadata_path = assets::metadata_path_for(absolute_source);
                    auto metadata = assets::load_asset_metadata(metadata_path);
                    if (!metadata) return fail(metadata.error().message);
                    metadata.value().settings_version = render::tools::texture_import_settings::current_version;
                    metadata.value().canonical_settings = render::tools::serialize_texture_import_settings(settings);
                    const auto saved = assets::save_asset_metadata(metadata_path, metadata.value());
                    if (!saved) return fail(saved.error().message);
                    const auto job = state_->asset_registry->reimport(*guid, assets::asset_streaming_priority::high);
                    if (!job.valid()) return fail("Texture could not be queued for reimport");
                    return success("{\"queued\":true,\"mode\":" + to_json_string(payload.mode) + '}');
                }
                else if constexpr (std::is_same_v<command_type, host_asset_cancel_import_command>)
                {
                    if (!state_->asset_registry->cancel_import(*guid)) return fail("Asset has no cancellable import");
                    return success("{\"cancelled\":true}");
                }
                else
                {
                    const auto result = [&]
                    {
                        if constexpr (std::is_same_v<command_type, host_asset_move_command>)
                            return state_->asset_registry->move(*guid, payload.path);
                        else
                            return state_->asset_registry->rename(*guid, payload.name);
                    }();
                    if (!result.succeeded()) return fail(result.error.message);
                    return success("{\"guid\":" + to_json_string(payload.guid) +
                                   ",\"path\":" + to_json_string(result.current_path.generic_string()) + '}');
                }
            }
            else if constexpr (std::is_same_v<command_type, host_shader_compile_command>)
            {
                if (!state_->project_open) return fail("Cannot compile a shader before a project is open");
                const auto source_path = resolve_project_document(state_->project.root, payload.path, false);
                if (!source_path) return fail("Shader source path must remain inside the open project");
                const auto stage = payload.stage == "vertex"    ? render::shader_stage::vertex
                                   : payload.stage == "compute" ? render::shader_stage::compute
                                                                : render::shader_stage::fragment;
                const auto domain = payload.domain == "compute"       ? render::shader_domain::compute
                                    : payload.domain == "postProcess" ? render::shader_domain::post_process
                                                                      : render::shader_domain::surface;
                std::string source = payload.source;
                std::unordered_map<std::uint32_t, std::string> graph_line_nodes;
                std::vector<render::shader_parameter_descriptor> graph_parameters;
                if (payload.domain == "materialGraph")
                {
                    auto lowered = render::tools::lower_material_graph_json(payload.source);
                    if (!lowered)
                    {
                        nlohmann::json response{{"succeeded", false},
                                                {"message", lowered.error().message},
                                                {"diagnostics", nlohmann::json::array()}};
                        return success(response.dump());
                    }
                    source = std::move(lowered.value().source);
                    graph_line_nodes = std::move(lowered.value().generated_line_nodes);
                    graph_parameters = std::move(lowered.value().parameters);
                }
                render::tools::slang_shader_compiler compiler;
                render::shader_library_cache cache;
                render::shader_compile_request request{.source_path = source_path->generic_string(),
                                                       .source_override = std::move(source),
                                                       .entry_point =
                                                           payload.entry_point.empty() ? "main" : payload.entry_point,
                                                       .profile = "spirv_1_5",
                                                       .domain = domain,
                                                       .stage = stage,
                                                       .target = render::shader_target::spirv,
                                                       .optimization = render::shader_optimization::development,
                                                       .include_directories = {source_path->parent_path()},
                                                       .generated_line_nodes = std::move(graph_line_nodes),
                                                       .generate_debug_information = true};
                auto compiled = cache.compile_or_get(compiler, request);
                nlohmann::json response;
                response["compiler"] = compiler.fingerprint();
                response["entryPoint"] = request.entry_point;
                response["target"] = "spirv-vulkan1.2";
                response["diagnostics"] = nlohmann::json::array();
                const auto append_diagnostics = [&](const auto& diagnostics)
                {
                    for (const auto& diagnostic : diagnostics)
                        response["diagnostics"].push_back(
                            {{"severity", diagnostic.severity == render::shader_diagnostic_severity::error ? "error"
                                          : diagnostic.severity == render::shader_diagnostic_severity::warning
                                              ? "warning"
                                              : "information"},
                             {"code", diagnostic.code},
                             {"message", diagnostic.message},
                             {"path", diagnostic.location.path},
                             {"line", diagnostic.location.line},
                             {"column", diagnostic.location.column},
                             {"graphNode", diagnostic.location.graph_node_id}});
                };
                if (!compiled)
                {
                    append_diagnostics(compiled.error().diagnostics);
                    response["succeeded"] = false;
                    response["message"] = compiled.error().message;
                    return success(response.dump());
                }
                append_diagnostics(compiled.value().diagnostics);
                if (!graph_parameters.empty()) compiled.value().reflection.parameters = std::move(graph_parameters);
                response["succeeded"] = true;
                response["message"] = "Transient shader generation compiled";
                response["bytecodeBytes"] = compiled.value().bytecode.size();
                response["parameters"] = nlohmann::json::array();
                for (const auto& parameter : compiled.value().reflection.parameters)
                    response["parameters"].push_back({{"id", parameter.id.representation()},
                                                      {"name", parameter.name},
                                                      {"offset", parameter.offset},
                                                      {"size", parameter.size}});
                response["resources"] = nlohmann::json::array();
                for (const auto& resource : compiled.value().reflection.resources)
                    response["resources"].push_back({{"name", resource.name},
                                                     {"set", resource.set},
                                                     {"binding", resource.binding},
                                                     {"count", resource.count}});
                return success(response.dump());
            }
            else if constexpr (std::is_same_v<command_type, host_runtime_resume_command>)
            {
                state_->simulation.resume();
                state_->preview_stopped = false;
                ++state_->runtime_revision;
                const auto snapshot = runtime_snapshot();
                push_event(state_->events, state_->event_sequence, host_event_type::runtime_state_changed,
                           "Preview runtime resumed", {}, to_json(snapshot));
                return success(to_json(snapshot));
            }
            else if constexpr (std::is_same_v<command_type, host_runtime_pause_command> ||
                               std::is_same_v<command_type, host_runtime_stop_command>)
            {
                state_->simulation.pause();
                state_->preview_stopped = std::is_same_v<command_type, host_runtime_stop_command>;
                ++state_->runtime_revision;
                const auto snapshot = runtime_snapshot();
                push_event(state_->events, state_->event_sequence, host_event_type::runtime_state_changed,
                           std::is_same_v<command_type, host_runtime_stop_command> ? "Preview runtime stopped"
                                                                                   : "Preview runtime paused",
                           {}, to_json(snapshot));
                return success(to_json(snapshot));
            }
            else if constexpr (std::is_same_v<command_type, host_runtime_step_command>)
            {
                if (!state_->simulation.paused()) state_->simulation.pause();
                state_->preview_stopped = false;
                if (!state_->simulation.step(payload.ticks))
                    return fail("Preview runtime could not queue a fixed-step");
                const framework::frame_time stepped = state_->simulation.advance(0.0);
                ++state_->runtime_revision;
                const auto snapshot = runtime_snapshot();
                push_event(state_->events, state_->event_sequence, host_event_type::runtime_tick_completed,
                           "Preview runtime stepped " + std::to_string(stepped.completed_ticks) + " tick(s)", {},
                           to_json(snapshot));
                return success(to_json(snapshot));
            }
            else if constexpr (std::is_same_v<command_type, host_runtime_set_time_scale_command>)
            {
                if (!state_->simulation.set_time_scale(payload.value))
                    return fail("Runtime time scale must be finite and between 0 and 16");
                ++state_->runtime_revision;
                const auto snapshot = runtime_snapshot();
                push_event(state_->events, state_->event_sequence, host_event_type::runtime_state_changed,
                           "Preview runtime time scale changed", {}, to_json(snapshot));
                return success(to_json(snapshot));
            }
            else if constexpr (std::is_same_v<command_type, host_runtime_capture_snapshot_command>)
            {
                const auto worlds = state_->simulation.worlds().ordered_worlds();
                if (worlds.empty()) return fail("Preview runtime has no world to snapshot");
                const framework::world_snapshot_result captured =
                    state_->simulation.capture_snapshot(worlds.front(), payload.label);
                if (!captured.succeeded) return fail(captured.error);
                return success("{\"snapshotId\":" + std::to_string(captured.metadata.id.value) +
                               ",\"tickId\":" + std::to_string(captured.metadata.tick.value) + '}');
            }
            else if constexpr (std::is_same_v<command_type, host_runtime_restore_snapshot_command>)
            {
                const framework::world_snapshot_result restored =
                    state_->simulation.restore_snapshot({payload.snapshot_id});
                if (!restored.succeeded) return fail(restored.error);
                if (state_->scene.selected_entity.valid() && !state_->scene.scene.alive(state_->scene.selected_entity))
                    clear_selection(state_->scene.scene, state_->scene.selected_entity);
                synchronize_all_terrain_resources(state_->scene, *state_->renderer);
                if (auto* camera_transform =
                        state_->scene.scene.try_get<scene::transform_component>(state_->scene.camera_entity))
                    state_->camera_controller.synchronize_from(*camera_transform);
                ++state_->runtime_revision;
                const auto snapshot = runtime_snapshot();
                push_event(state_->events, state_->event_sequence, host_event_type::scene_changed,
                           "Preview runtime snapshot restored", state_->scene.selected_entity);
                push_event(state_->events, state_->event_sequence, host_event_type::runtime_state_changed,
                           "Preview runtime snapshot restored", {}, to_json(snapshot));
                return success(to_json(snapshot));
            }
            else if constexpr (std::is_same_v<command_type, host_open_scene_command>)
            {
                if (!state_->project_open) return fail("Cannot open a scene before a project is open");

                if (payload.path.extension() == ".arcscene")
                {
                    if (payload.append) return fail("Appending native ARC scene documents is not supported");
                    const auto path = payload.path.is_absolute() ? payload.path : state_->project.root / payload.path;
                    const auto loaded = load_scene_document(state_->scene, *state_->renderer, state_->project.root,
                                                            path, state_->asset_registry.get());
                    if (!loaded.succeeded) return fail(loaded.message);
                    // A recovered backup is intentionally a dirty document. The
                    // damaged primary is not replaced until the user explicitly
                    // saves the recovered scene.
                    state_->history.clear(state_->scene, !state_->scene.recovered_document);
                    if (auto* camera_transform =
                            state_->scene.scene.try_get<scene::transform_component>(state_->scene.camera_entity))
                        state_->camera_controller.synchronize_from(*camera_transform);
                    push_event(state_->events, state_->event_sequence, host_event_type::scene_changed,
                               "ARC scene loaded", state_->scene.selected_entity);
                    return success("{\"entityCount\":" + std::to_string(loaded.entity_count) + '}');
                }
                const auto mode = payload.append ? editor_scene_open_mode::append : editor_scene_open_mode::replace;
                std::optional<editor_scene_state> import_before;
                if (payload.append) import_before = state_->scene;
                const auto asset_root = payload.path.is_absolute() ? payload.path.parent_path() : state_->assets.root;
                const auto result =
                    open_scene_asset_in_editor(state_->scene, *state_->renderer, asset_root, payload.path, mode);
                if (!result.succeeded)
                    return fail(result.message.empty() ? "Failed to open scene asset" : result.message);

                ensure_scene_authoring_metadata(state_->scene);
                if (state_->asset_registry)
                    resolve_editor_asset_bindings(state_->scene, *state_->asset_registry, state_->project.root);
                if (auto* camera_transform =
                        state_->scene.scene.try_get<scene::transform_component>(state_->scene.camera_entity))
                    state_->camera_controller.synchronize_from(*camera_transform);
                if (import_before)
                    state_->history.record("Import Scene", std::move(*import_before), state_->scene);
                else
                {
                    state_->scene.scene_name = payload.path.stem().string();
                    state_->scene.active_scene_path.clear();
                    state_->history.clear(state_->scene, false);
                }

                const std::string message =
                    std::string(payload.append ? "Imported scene asset: " : "Opened scene asset: ") +
                    payload.path.filename().string();
                push_event(state_->events, state_->event_sequence, host_event_type::scene_changed, message,
                           state_->scene.selected_entity);
                return success("{\"entityCount\":" + std::to_string(result.entity_count) + '}');
            }
            else if constexpr (std::is_same_v<command_type, host_autosave_scene_command>)
            {
                if (!state_->project_open) return fail("Cannot autosave before a project is open");
                const auto path = payload.path.is_absolute() ? payload.path : state_->project.root / payload.path;
                auto snapshot = state_->scene;
                const auto saved =
                    save_scene_document(snapshot, state_->project.root, path, state_->asset_registry.get());
                if (!saved.succeeded) return fail(saved.message.empty() ? "Scene autosave failed" : saved.message);
                return success("{\"path\":" + to_json_string(path.generic_string()) +
                               ",\"sceneGuid\":" + to_json_string(ecs::to_string(state_->scene.scene_guid)) +
                               ",\"historyRevision\":" + std::to_string(state_->history.snapshot().revision) + '}');
            }
            else if constexpr (std::is_same_v<command_type, host_open_recovery_scene_command>)
            {
                if (!state_->project_open) return fail("Cannot recover a scene before a project is open");
                const auto recovery_path =
                    payload.path.is_absolute() ? payload.path : state_->project.root / payload.path;
                const auto loaded = load_scene_document(state_->scene, *state_->renderer, state_->project.root,
                                                        recovery_path, state_->asset_registry.get());
                if (!loaded.succeeded) return fail(loaded.message);
                state_->scene.active_scene_path = payload.original_path;
                state_->scene.recovered_document = true;
                state_->history.clear(state_->scene, false);
                if (auto* camera_transform =
                        state_->scene.scene.try_get<scene::transform_component>(state_->scene.camera_entity))
                    state_->camera_controller.synchronize_from(*camera_transform);
                ++state_->scene_revision;
                ++state_->world_epoch;
                push_event(state_->events, state_->event_sequence, host_event_type::scene_changed,
                           "Recovery scene loaded", state_->scene.selected_entity);
                return success("{\"entityCount\":" + std::to_string(loaded.entity_count) + ",\"recovered\":true}");
            }
            else if constexpr (std::is_same_v<command_type, host_new_scene_command>)
            {
                state_->scene.terrain_render_proxies.clear(*state_->renderer);
                state_->scene = create_blank_scene(*state_->renderer);
                state_->scene.scene_name = payload.name.empty() ? "Untitled" : payload.name;
                state_->scene.active_scene_path.clear();
                state_->history.clear(state_->scene, false);
                if (auto* camera_transform =
                        state_->scene.scene.try_get<scene::transform_component>(state_->scene.camera_entity))
                    state_->camera_controller.synchronize_from(*camera_transform);
                push_event(state_->events, state_->event_sequence, host_event_type::scene_changed, "New scene created",
                           state_->scene.selected_entity);
                return success();
            }
            else if constexpr (std::is_same_v<command_type, host_save_scene_command> ||
                               std::is_same_v<command_type, host_save_scene_as_command>)
            {
                std::filesystem::path path;
                if constexpr (std::is_same_v<command_type, host_save_scene_as_command>)
                    path = payload.path;
                else
                    path = state_->scene.active_scene_path;
                if (path.empty()) return fail("Scene has no path; use scene.saveAs first");
                if (path.is_relative()) path = state_->project.root / path;
                if (path.extension() != ".arcscene") path.replace_extension(".arcscene");
                const auto saved =
                    save_scene_document(state_->scene, state_->project.root, path, state_->asset_registry.get());
                if (!saved.succeeded) return fail(saved.message);
                state_->history.mark_saved();
                push_event(state_->events, state_->event_sequence, host_event_type::scene_changed, "Scene saved",
                           state_->scene.selected_entity);
                return success("{\"path\":" + to_json_string(path.generic_string()) + '}');
            }
            else if constexpr (std::is_same_v<command_type, host_create_entity_command>)
            {
                const auto requested_parent = to_scene_entity(payload.parent);
                if (payload.parent.valid() && !state_->scene.scene.alive(requested_parent))
                    return fail("Cannot create a child under a missing parent", requested_parent);
                ecs::entity created{};
                switch (payload.kind)
                {
                    case host_create_entity_kind::empty:
                        created = state_->scene.scene.create();
                        state_->scene.scene.emplace<scene::name_component>(created, "Entity");
                        state_->scene.scene.emplace<scene::tag_component>(created, "Untagged");
                        state_->scene.scene.emplace<scene::active_component>(created);
                        state_->scene.scene.emplace<scene::transform_component>(created);
                        state_->scene.scene.emplace<scene::selection_component>(created, false);
                        ensure_scene_authoring_metadata(state_->scene);
                        select_entity(state_->scene.scene, created, state_->scene.selected_entity);
                        break;
                    case host_create_entity_kind::plane:
                    case host_create_entity_kind::cube:
                    case host_create_entity_kind::sphere:
                    case host_create_entity_kind::cylinder:
                    case host_create_entity_kind::cone:
                    case host_create_entity_kind::capsule:
                        created =
                            add_primitive_to_scene(state_->scene, *state_->renderer, primitive_type_for(payload.kind));
                        break;
                    case host_create_entity_kind::world_environment:
                        created = add_world_environment_to_scene(state_->scene);
                        break;
                    case host_create_entity_kind::terrain:
                        created = add_terrain_to_scene(state_->scene, *state_->renderer);
                        break;
                    case host_create_entity_kind::water:
                        created = add_water_to_scene(state_->scene, *state_->renderer);
                        break;
                    case host_create_entity_kind::grass_patch:
                        created = add_grass_patch_to_scene(state_->scene, *state_->renderer);
                        break;
                    case host_create_entity_kind::decal:
                        created = add_decal_to_scene(state_->scene);
                        break;
                }

                const std::string label = create_entity_kind_label(payload.kind);
                if (!created.valid()) return fail("Failed to create entity: " + label);

                if (payload.parent.valid())
                {
                    if (!scene::reparent(state_->scene.scene, created, requested_parent, {},
                                         scene::reparent_transform_policy::preserve_local))
                    {
                        scene::destroy_subtree(state_->scene.scene, created);
                        return fail("Cannot create a child under an invalid parent", requested_parent);
                    }
                }

                const std::string message = "Created entity: " + label;
                arc::diagnostics::info("editor.host", message);
                push_event(state_->events, state_->event_sequence, host_event_type::entity_created, message, created);
                push_event(state_->events, state_->event_sequence, host_event_type::entity_selected, "Selected entity",
                           created);
                push_event(state_->events, state_->event_sequence, host_event_type::scene_changed, "Scene changed",
                           created);
                return success("{\"entity\":" + to_json(to_host_entity(created)) + '}');
            }
            else if constexpr (std::is_same_v<command_type, host_delete_entity_command>)
            {
                const auto entity = to_scene_entity(payload.entity);
                if (!state_->scene.scene.alive(entity)) return fail("Cannot delete a missing entity", entity);

                const std::string name = entity_name(state_->scene, entity, "Entity");
                const auto removed = scene::subtree(state_->scene.scene, entity);
                std::vector<ecs::entity_guid> removed_guids;
                removed_guids.reserve(removed.size());
                for (const auto nested : removed)
                {
                    const auto guid = entity_guid_of(state_->scene, nested);
                    if (guid.valid()) removed_guids.push_back(guid);
                }
                const auto was_removed = [&](ecs::entity candidate)
                { return std::ranges::find(removed, candidate) != removed.end(); };
                if (was_removed(state_->scene.selected_entity))
                    clear_selection(state_->scene.scene, state_->scene.selected_entity);
                if (state_->pending_pick && was_removed(state_->pending_pick->cpu_fallback.entity))
                    state_->pending_pick.reset();
                if (was_removed(state_->scene.terrain_entity))
                {
                    state_->terrain_brush_local_position.reset();
                    state_->terrain_stroke_previous_position.reset();
                    state_->terrain_flatten_height_captured = false;
                    if (state_->viewport_tool.tool == host_viewport_tool::terrain)
                        state_->viewport_tool.tool = host_viewport_tool::select;
                }
                for (const auto guid : removed_guids)
                    state_->scene.terrain_render_proxies.erase(guid, *state_->renderer);
                scene::destroy_subtree(state_->scene.scene, entity);
                for (const auto nested : removed)
                    forget_entity(state_->scene, nested);
                const auto removed_guid = [&](ecs::entity_guid guid)
                { return std::ranges::find(removed_guids, guid) != removed_guids.end(); };
                std::erase_if(state_->scene.asset_bindings,
                              [&](const auto& binding) { return removed_guid(binding.entity); });
                std::erase_if(state_->scene.unknown_component_records,
                              [&](const auto& record) { return removed_guid(record.first); });
                std::erase_if(state_->scene.preserved_component_records,
                              [&](const auto& record) { return removed_guid(record.entity); });
                state_->scene.last_render = {};
                const std::string message = "Deleted entity: " + name;
                arc::diagnostics::info("editor.host", message);
                push_event(state_->events, state_->event_sequence, host_event_type::entity_deleted, message, entity);
                push_event(state_->events, state_->event_sequence, host_event_type::scene_changed, "Scene changed",
                           entity);
                return success("{\"entity\":" + to_json(payload.entity) + '}');
            }
            else if constexpr (std::is_same_v<command_type, host_duplicate_entity_command>)
            {
                const auto source = to_scene_entity(payload.entity);
                if (!state_->scene.scene.alive(source)) return fail("Cannot duplicate a missing entity", source);
                const auto* links = state_->scene.scene.try_get<scene::hierarchy_component>(source);
                const auto duplicate =
                    duplicate_entity_subtree(state_->scene, source, links ? links->parent : ecs::entity{});
                if (!duplicate.valid()) return fail("Entity duplication failed", source);
                select_entity(state_->scene.scene, duplicate, state_->scene.selected_entity);
                push_event(state_->events, state_->event_sequence, host_event_type::entity_created, "Entity duplicated",
                           duplicate);
                push_event(state_->events, state_->event_sequence, host_event_type::scene_changed, "Scene changed",
                           duplicate);
                return success("{\"entity\":" + to_json(to_host_entity(duplicate)) + '}');
            }
            else if constexpr (std::is_same_v<command_type, host_create_prefab_command>)
            {
                const auto entity = to_scene_entity(payload.entity);
                if (!state_->scene.scene.alive(entity))
                    return fail("Cannot create a prefab from a missing entity", entity);
                auto path = payload.path;
                if (path.extension() != ".arcprefab") path.replace_extension(".arcprefab");
                const auto resolved = resolve_project_document(state_->project.root, path, false);
                if (!resolved) return fail("Prefab path must be inside the active project", entity);
                const auto result = save_prefab_document(state_->scene, state_->project.root, entity, *resolved,
                                                         state_->asset_registry.get());
                if (!result.succeeded) return fail(result.message, entity);
                push_event(state_->events, state_->event_sequence, host_event_type::component_changed, "Prefab created",
                           entity);
                push_event(state_->events, state_->event_sequence, host_event_type::scene_changed, "Scene changed",
                           entity);
                return success(
                    "{\"entity\":" + to_json(payload.entity) + ",\"path\":" +
                    to_json_string(std::filesystem::relative(*resolved, state_->project.root).generic_string()) + '}');
            }
            else if constexpr (std::is_same_v<command_type, host_instantiate_prefab_command>)
            {
                const auto parent = to_scene_entity(payload.parent);
                if (payload.parent.valid() && !state_->scene.scene.alive(parent))
                    return fail("Cannot instantiate a prefab below a missing parent", parent);
                const auto resolved = resolve_project_document(state_->project.root, payload.path, true);
                if (!resolved || resolved->extension() != ".arcprefab")
                    return fail("Prefab source is missing or outside the active project");
                const auto result = instantiate_prefab_document(state_->scene, *state_->renderer, state_->project.root,
                                                                *resolved, parent, state_->asset_registry.get());
                if (!result.succeeded) return fail(result.message);
                select_entity(state_->scene.scene, result.root, state_->scene.selected_entity);
                push_event(state_->events, state_->event_sequence, host_event_type::entity_created,
                           "Prefab instantiated", result.root);
                push_event(state_->events, state_->event_sequence, host_event_type::entity_selected,
                           "Selected prefab instance", result.root);
                push_event(state_->events, state_->event_sequence, host_event_type::scene_changed, "Scene changed",
                           result.root);
                return success("{\"entity\":" + to_json(to_host_entity(result.root)) +
                               ",\"entityCount\":" + std::to_string(result.entity_count) + '}');
            }
            else if constexpr (std::is_same_v<command_type, host_apply_prefab_command>)
            {
                const auto entity = to_scene_entity(payload.entity);
                if (!state_->scene.scene.alive(entity)) return fail("Cannot apply a missing prefab instance", entity);
                const auto result =
                    apply_prefab_instance(state_->scene, state_->project.root, entity, state_->asset_registry.get());
                if (!result.succeeded) return fail(result.message, entity);
                push_event(state_->events, state_->event_sequence, host_event_type::component_changed,
                           "Prefab changes applied", entity);
                push_event(state_->events, state_->event_sequence, host_event_type::scene_changed, "Scene changed",
                           entity);
                return success();
            }
            else if constexpr (std::is_same_v<command_type, host_revert_prefab_command>)
            {
                const auto entity = to_scene_entity(payload.entity);
                if (!state_->scene.scene.alive(entity)) return fail("Cannot revert a missing prefab instance", entity);
                const auto result = revert_prefab_instance(state_->scene, *state_->renderer, state_->project.root,
                                                           entity, state_->asset_registry.get());
                if (!result.succeeded) return fail(result.message, entity);
                select_entity(state_->scene.scene, result.root, state_->scene.selected_entity);
                push_event(state_->events, state_->event_sequence, host_event_type::entity_deleted,
                           "Prefab instance replaced", entity);
                push_event(state_->events, state_->event_sequence, host_event_type::entity_created,
                           "Prefab instance reverted", result.root);
                push_event(state_->events, state_->event_sequence, host_event_type::entity_selected,
                           "Selected prefab instance", result.root);
                push_event(state_->events, state_->event_sequence, host_event_type::scene_changed, "Scene changed",
                           result.root);
                return success("{\"entity\":" + to_json(to_host_entity(result.root)) + '}');
            }
            else if constexpr (std::is_same_v<command_type, host_unpack_prefab_command>)
            {
                const auto entity = to_scene_entity(payload.entity);
                if (!state_->scene.scene.alive(entity)) return fail("Cannot unpack a missing prefab instance", entity);
                if (!unpack_prefab_instance(state_->scene, entity))
                    return fail("Entity is not a prefab instance", entity);
                push_event(state_->events, state_->event_sequence, host_event_type::component_changed,
                           "Prefab instance unpacked", entity);
                push_event(state_->events, state_->event_sequence, host_event_type::scene_changed, "Scene changed",
                           entity);
                return success();
            }
            else if constexpr (std::is_same_v<command_type, host_reparent_entity_command>)
            {
                const auto entity = to_scene_entity(payload.entity);
                const auto parent = to_scene_entity(payload.parent);
                const auto before = to_scene_entity(payload.before_sibling);
                if (!scene::reparent(state_->scene.scene, entity, parent, before,
                                     payload.preserve_world ? scene::reparent_transform_policy::preserve_world
                                                            : scene::reparent_transform_policy::preserve_local))
                    return fail("Invalid hierarchy reparent operation", entity);
                push_event(state_->events, state_->event_sequence, host_event_type::scene_changed, "Entity reparented",
                           entity);
                return success();
            }
            else if constexpr (std::is_same_v<command_type, host_reorder_entity_command>)
            {
                const auto entity = to_scene_entity(payload.entity);
                if (!scene::reorder(state_->scene.scene, entity, to_scene_entity(payload.before_sibling)))
                    return fail("Invalid hierarchy reorder operation", entity);
                push_event(state_->events, state_->event_sequence, host_event_type::scene_changed, "Entity reordered",
                           entity);
                return success();
            }
            else if constexpr (std::is_same_v<command_type, host_rename_entity_command>)
            {
                const auto entity = to_scene_entity(payload.entity);
                if (!state_->scene.scene.alive(entity)) return fail("Cannot rename a missing entity", entity);

                state_->scene.scene.emplace<scene::name_component>(entity, payload.name);
                const std::string message = "Renamed entity to '" + payload.name + "'";
                arc::diagnostics::info("editor.host", message);
                push_event(state_->events, state_->event_sequence, host_event_type::component_changed, message, entity);
                return success("{\"entity\":" + to_json(payload.entity) + '}');
            }
            else if constexpr (std::is_same_v<command_type, host_select_entity_command>)
            {
                const auto entity = to_scene_entity(payload.entity);
                if (!state_->scene.scene.alive(entity)) return fail("Cannot select a missing entity", entity);
                const auto* current = state_->scene.scene.try_get<scene::selection_component>(entity);
                const bool was_selected = current && current->selected;
                if (!payload.additive && entity == state_->scene.selected_entity && was_selected)
                    return success("{\"entity\":" + to_json(payload.entity) + '}');

                bool changed{};
                if (!payload.additive)
                {
                    changed = select_entity(state_->scene.scene, entity, state_->scene.selected_entity);
                }
                else
                {
                    const bool next_selected = payload.toggle ? !was_selected : true;
                    state_->scene.scene.emplace<scene::selection_component>(entity, next_selected);
                    changed = next_selected != was_selected;
                    if (next_selected)
                    {
                        state_->scene.selected_entity = entity;
                    }
                    else if (state_->scene.selected_entity == entity)
                    {
                        state_->scene.selected_entity = {};
                        for (const auto candidate : state_->scene.scene.entities())
                        {
                            const auto* selection = state_->scene.scene.try_get<scene::selection_component>(candidate);
                            if (selection && selection->selected)
                            {
                                state_->scene.selected_entity = candidate;
                                break;
                            }
                        }
                    }
                }
                if (!changed) return success("{\"entity\":" + to_json(payload.entity) + '}');

                const std::string message = "Selected entity: " + entity_name(state_->scene, entity, "Entity");
                push_event(state_->events, state_->event_sequence, host_event_type::entity_selected, message,
                           state_->scene.selected_entity);
                return success("{\"entity\":" + to_json(payload.entity) + '}');
            }
            else if constexpr (std::is_same_v<command_type, host_clear_selection_command>)
            {
                if (!state_->scene.scene.alive(state_->scene.selected_entity)) return success();
                clear_selection(state_->scene.scene, state_->scene.selected_entity);
                push_event(state_->events, state_->event_sequence, host_event_type::entity_selected,
                           "Cleared entity selection");
                return success();
            }
            else if constexpr (std::is_same_v<command_type, host_set_active_command>)
            {
                const auto entity = to_scene_entity(payload.entity);
                const auto targets = edit_targets(state_->scene.scene, entity, payload.apply_to_selection);
                if (targets.empty()) return fail("Cannot edit a missing or unselected entity", entity);
                for (const auto target : targets)
                    state_->scene.scene.emplace<scene::active_component>(target, payload.active);
                push_event(state_->events, state_->event_sequence, host_event_type::component_changed,
                           "Entity active state changed for " + std::to_string(targets.size()) + " entity(s)", entity);
                return success("{\"entity\":" + to_json(payload.entity) + '}');
            }
            else if constexpr (std::is_same_v<command_type, host_set_tag_command>)
            {
                const auto entity = to_scene_entity(payload.entity);
                const auto targets = edit_targets(state_->scene.scene, entity, payload.apply_to_selection);
                if (targets.empty()) return fail("Cannot edit a missing or unselected entity", entity);
                for (const auto target : targets)
                    state_->scene.scene.emplace<scene::tag_component>(target, payload.tag);
                push_event(state_->events, state_->event_sequence, host_event_type::component_changed,
                           "Entity tag changed for " + std::to_string(targets.size()) + " entity(s)", entity);
                return success("{\"entity\":" + to_json(payload.entity) + '}');
            }
            else if constexpr (std::is_same_v<command_type, host_set_transform_command>)
            {
                const auto entity = to_scene_entity(payload.entity);
                const auto targets = edit_targets(state_->scene.scene, entity, payload.apply_to_selection);
                if (targets.empty()) return fail("Cannot edit a missing or unselected entity", entity);
                const auto transform = to_scene_transform(payload.transform);
                for (const auto target : targets)
                {
                    state_->scene.scene.emplace<scene::transform_component>(target, transform);
                    scene::mark_transform_subtree_dirty(state_->scene.scene, target);
                }
                scene::update_world_transforms(state_->scene.scene);
                if (std::find(targets.begin(), targets.end(), state_->scene.camera_entity) != targets.end())
                    state_->camera_controller.synchronize_from(
                        state_->scene.scene.get<scene::transform_component>(state_->scene.camera_entity));
                push_event(state_->events, state_->event_sequence, host_event_type::component_changed,
                           "Entity transform changed for " + std::to_string(targets.size()) + " entity(s)", entity);
                return success("{\"entity\":" + to_json(payload.entity) + '}');
            }
            else if constexpr (std::is_same_v<command_type, host_set_render_layer_command>)
            {
                const auto entity = to_scene_entity(payload.entity);
                if (payload.render_layer_mask == 0u)
                    return fail("Render layer mask must contain at least one layer", entity);
                const auto targets = edit_targets(state_->scene.scene, entity, payload.apply_to_selection);
                if (targets.empty()) return fail("Cannot edit a missing or unselected entity", entity);
                for (const auto target : targets)
                    state_->scene.scene.emplace<scene::render_layer_component>(target, payload.render_layer_mask);
                push_event(state_->events, state_->event_sequence, host_event_type::component_changed,
                           "Entity render layer changed for " + std::to_string(targets.size()) + " entity(s)", entity);
                return success("{\"entity\":" + to_json(payload.entity) + '}');
            }
            else if constexpr (std::is_same_v<command_type, host_set_mobility_command>)
            {
                const auto entity = to_scene_entity(payload.entity);
                const auto targets = edit_targets(state_->scene.scene, entity, payload.apply_to_selection);
                if (targets.empty()) return fail("Cannot edit mobility for a missing or unselected entity", entity);
                for (const auto target : targets)
                    state_->scene.scene.emplace<scene::mobility_component>(target,
                                                                           to_render_mobility(payload.mobility));
                push_event(state_->events, state_->event_sequence, host_event_type::component_changed,
                           "Entity mobility changed for " + std::to_string(targets.size()) + " entity(s)", entity);
                return success("{\"entity\":" + to_json(payload.entity) + '}');
            }
            else if constexpr (std::is_same_v<command_type, host_set_camera_command>)
            {
                const auto entity = to_scene_entity(payload.entity);
                if (!valid_camera(payload.camera))
                    return fail("Camera values are outside their valid authored ranges", entity);
                const auto targets = edit_targets(state_->scene.scene, entity, payload.apply_to_selection);
                if (targets.empty()) return fail("Cannot edit a missing or unselected camera", entity);
                if (std::any_of(targets.begin(), targets.end(), [&](ecs::entity target)
                                { return !state_->scene.scene.has<scene::camera_component>(target); }))
                    return fail("Every selected entity must have an editable camera component", entity);
                const auto camera = to_scene_camera(payload.camera);
                for (const auto target : targets)
                    state_->scene.scene.get<scene::camera_component>(target) = camera;
                push_event(state_->events, state_->event_sequence, host_event_type::component_changed,
                           "Entity camera changed for " + std::to_string(targets.size()) + " entity(s)", entity);
                return success("{\"entity\":" + to_json(payload.entity) + '}');
            }
            else if constexpr (std::is_same_v<command_type, host_set_light_command>)
            {
                const auto entity = to_scene_entity(payload.entity);
                if (!valid_light(payload.light))
                    return fail("Light values are outside their valid authored ranges", entity);
                const auto targets = edit_targets(state_->scene.scene, entity, payload.apply_to_selection);
                if (targets.empty()) return fail("Cannot edit a missing or unselected light entity", entity);
                const auto matches_kind = [&](ecs::entity target)
                {
                    return (payload.light.kind == host_light_kind::directional &&
                            state_->scene.scene.has<scene::directional_light_component>(target)) ||
                           (payload.light.kind == host_light_kind::point &&
                            state_->scene.scene.has<scene::point_light_component>(target)) ||
                           (payload.light.kind == host_light_kind::spot &&
                            state_->scene.scene.has<scene::spot_light_component>(target)) ||
                           ((payload.light.kind == host_light_kind::rectangle ||
                             payload.light.kind == host_light_kind::disk) &&
                            state_->scene.scene.has<scene::area_light_component>(target));
                };
                if (!std::all_of(targets.begin(), targets.end(), matches_kind))
                    return fail("Every selected entity must have the same editable light component", entity);

                const auto unit = to_render_light_unit(payload.light.unit);
                const auto color = to_math_vec3(payload.light.color);
                for (const auto target : targets)
                {
                    if (auto* directional_light =
                            state_->scene.scene.try_get<scene::directional_light_component>(target))
                    {
                        directional_light->color = color;
                        directional_light->intensity = payload.light.intensity;
                        directional_light->intensity_unit = unit;
                        directional_light->enabled = payload.light.enabled;
                        directional_light->casts_shadows = payload.light.casts_shadows;
                        directional_light->use_color_temperature = payload.light.use_color_temperature;
                        directional_light->temperature_kelvin = payload.light.temperature_kelvin;
                        copy_shadow_from_host(directional_light->shadow, payload.light);
                        directional_light->cascades.cascade_count = payload.light.cascade_count;
                        directional_light->cascades.maximum_distance = payload.light.shadow_distance;
                        directional_light->cascades.split_lambda = payload.light.cascade_split_lambda;
                        directional_light->cascades.blend_fraction = payload.light.cascade_blend_fraction;
                        directional_light->cascades.stable = payload.light.stable_cascades;
                    }
                    else if (auto* point_light = state_->scene.scene.try_get<scene::point_light_component>(target))
                    {
                        point_light->color = color;
                        point_light->intensity = payload.light.intensity;
                        point_light->range = payload.light.range;
                        point_light->intensity_unit = unit;
                        point_light->enabled = payload.light.enabled;
                        point_light->casts_shadows = payload.light.casts_shadows;
                        point_light->use_color_temperature = payload.light.use_color_temperature;
                        point_light->temperature_kelvin = payload.light.temperature_kelvin;
                        copy_shadow_from_host(point_light->shadow, payload.light);
                    }
                    else if (auto* spot_light = state_->scene.scene.try_get<scene::spot_light_component>(target))
                    {
                        spot_light->color = color;
                        spot_light->intensity = payload.light.intensity;
                        spot_light->range = payload.light.range;
                        spot_light->inner_angle = math::to_radians(payload.light.inner_angle_degrees);
                        spot_light->outer_angle = math::to_radians(payload.light.outer_angle_degrees);
                        spot_light->intensity_unit = unit;
                        spot_light->enabled = payload.light.enabled;
                        spot_light->casts_shadows = payload.light.casts_shadows;
                        spot_light->use_color_temperature = payload.light.use_color_temperature;
                        spot_light->temperature_kelvin = payload.light.temperature_kelvin;
                        copy_shadow_from_host(spot_light->shadow, payload.light);
                    }
                    else if (auto* area_light = state_->scene.scene.try_get<scene::area_light_component>(target))
                    {
                        area_light->shape = payload.light.kind == host_light_kind::disk
                                                ? render::area_light_shape::disk
                                                : render::area_light_shape::rectangle;
                        area_light->color = color;
                        area_light->intensity = payload.light.intensity;
                        area_light->width = payload.light.width;
                        area_light->height = payload.light.height;
                        area_light->two_sided = payload.light.two_sided;
                        area_light->intensity_unit = unit;
                        area_light->enabled = payload.light.enabled;
                        area_light->casts_shadows = payload.light.casts_shadows;
                        area_light->use_color_temperature = payload.light.use_color_temperature;
                        area_light->temperature_kelvin = payload.light.temperature_kelvin;
                        copy_shadow_from_host(area_light->shadow, payload.light);
                    }
                }

                push_event(state_->events, state_->event_sequence, host_event_type::component_changed,
                           "Entity light changed for " + std::to_string(targets.size()) + " entity(s)", entity);
                return success("{\"entity\":" + to_json(payload.entity) + '}');
            }
            else if constexpr (std::is_same_v<command_type, host_set_mesh_renderer_command>)
            {
                const auto entity = to_scene_entity(payload.entity);
                if (!valid_base_color_tint(payload.base_color_tint))
                    return fail("Mesh renderer tint channels must be finite and between 0 and 1", entity);
                if (payload.representation >
                    static_cast<std::uint8_t>(render::geometry_representation_policy::virtualized))
                    return fail("Mesh renderer geometry representation is invalid", entity);
                if (!std::isfinite(payload.shadow_lod_bias) || !std::isfinite(payload.maximum_shadow_distance) ||
                    payload.shadow_lod_bias < -4.0f || payload.shadow_lod_bias > 8.0f ||
                    payload.maximum_shadow_distance < 0.0f)
                    return fail("Mesh renderer shadow values are outside supported ranges", entity);
                const auto targets = edit_targets(state_->scene.scene, entity, payload.apply_to_selection);
                if (targets.empty()) return fail("Cannot edit a missing or unselected mesh renderer", entity);
                if (std::any_of(targets.begin(), targets.end(), [&](ecs::entity target)
                                { return !state_->scene.scene.has<scene::mesh_renderer_component>(target); }))
                    return fail("Every selected entity must have an editable mesh renderer component", entity);
                for (const auto target : targets)
                {
                    auto& mesh_renderer = state_->scene.scene.get<scene::mesh_renderer_component>(target);
                    mesh_renderer.representation =
                        static_cast<render::geometry_representation_policy>(payload.representation);
                    mesh_renderer.visible = payload.visible;
                    mesh_renderer.casts_shadows = payload.casts_shadows;
                    mesh_renderer.receives_shadows = payload.receives_shadows;
                    mesh_renderer.shadow_lod_bias = payload.shadow_lod_bias;
                    mesh_renderer.maximum_shadow_distance = payload.maximum_shadow_distance;
                    mesh_renderer.base_color_tint = to_math_vec4(payload.base_color_tint);
                }
                push_event(state_->events, state_->event_sequence, host_event_type::component_changed,
                           "Entity mesh renderer changed for " + std::to_string(targets.size()) + " entity(s)", entity);
                return success("{\"entity\":" + to_json(payload.entity) + '}');
            }
            else if constexpr (std::is_same_v<command_type, host_set_terrain_command>)
            {
                const auto entity = to_scene_entity(payload.entity);
                auto* terrain = state_->scene.scene.try_get<scene::terrain_component>(entity);
                if (!terrain) return fail("Entity does not have an editable terrain component", entity);
                if (!std::isfinite(payload.shadow_lod_bias) || !std::isfinite(payload.maximum_shadow_distance) ||
                    !std::isfinite(payload.geometric_error_multiplier) || payload.shadow_lod_bias < -4.0f ||
                    payload.shadow_lod_bias > 8.0f || payload.maximum_shadow_distance < 0.0f ||
                    payload.geometric_error_multiplier <= 0.0f ||
                    (payload.patch_quads != 16u && payload.patch_quads != 32u && payload.patch_quads != 64u))
                    return fail("Terrain LOD or shadow values are outside supported ranges", entity);
                terrain->enabled = payload.enabled;
                terrain->receive_shadows = payload.receive_shadows;
                terrain->cast_shadows = payload.cast_shadows;
                terrain->patch_quads = payload.patch_quads;
                terrain->maximum_hierarchy_depth = payload.maximum_hierarchy_depth;
                terrain->geometric_error_multiplier = payload.geometric_error_multiplier;
                terrain->shadow_lod_bias = payload.shadow_lod_bias;
                terrain->maximum_shadow_distance = payload.maximum_shadow_distance;
                ++terrain->content_revision;
                push_event(state_->events, state_->event_sequence, host_event_type::component_changed,
                           "Terrain settings changed", entity);
                return success("{\"entity\":" + to_json(payload.entity) + '}');
            }
            else if constexpr (std::is_same_v<command_type, host_set_terrain_brush_command>)
            {
                const auto entity = to_scene_entity(payload.entity);
                if (!state_->scene.scene.has<scene::terrain_component>(entity))
                    return fail("Terrain brush requires a terrain entity", entity);
                if (!std::isfinite(payload.radius) || !std::isfinite(payload.strength) ||
                    !std::isfinite(payload.falloff) || payload.radius < 0.25f || payload.radius > 128.0f ||
                    payload.strength <= 0.0f || payload.strength > 1.0f || payload.falloff < 0.0f ||
                    payload.falloff > 1.0f || payload.active_layer >= 4u)
                    return fail("Terrain brush values are outside supported ranges", entity);
                state_->terrain_brush.tool =
                    payload.tool == host_terrain_brush_tool::smooth    ? scene::terrain_brush_tool::smooth
                    : payload.tool == host_terrain_brush_tool::flatten ? scene::terrain_brush_tool::flatten
                    : payload.tool == host_terrain_brush_tool::paint   ? scene::terrain_brush_tool::paint
                                                                       : scene::terrain_brush_tool::sculpt;
                state_->terrain_brush.radius = payload.radius;
                state_->terrain_brush.strength = payload.strength;
                state_->terrain_brush.falloff = payload.falloff;
                state_->terrain_brush.active_layer = payload.active_layer;
                push_event(state_->events, state_->event_sequence, host_event_type::terrain_tool_changed,
                           "Terrain brush changed", entity, to_json(terrain_tool_snapshot()));
                return success(to_json(terrain_tool_snapshot()));
            }
            else if constexpr (std::is_same_v<command_type, host_create_terrain_command>)
            {
                scene::terrain_creation_descriptor descriptor;
                descriptor.size = payload.size;
                descriptor.minimum_elevation = payload.minimum_elevation;
                descriptor.maximum_elevation = payload.maximum_elevation;
                descriptor.sample_resolution = payload.resolution;
                descriptor.patch_quads = payload.patch_quads;
                descriptor.procedural_seed = payload.seed;
                descriptor.source = payload.source == "procedural" ? scene::terrain_initial_source::procedural
                                                                   : scene::terrain_initial_source::flat;
                const auto validation = scene::validate_terrain_creation(descriptor);
                if (!validation.succeeded) return fail(validation.message);
                const auto estimate = scene::estimate_terrain_memory(descriptor.sample_resolution);
                constexpr std::uint64_t history_budget = 64ull * 1024ull * 1024ull;
                if (estimate.history_bytes > history_budget)
                    return fail("Terrain creation needs " + std::to_string(estimate.history_bytes) +
                                " bytes of undo storage, exceeding the 64 MiB history budget");

                if (!state_->scene.terrain_material.valid())
                {
                    const auto asset_root =
                        !state_->assets.root.empty() ? state_->assets.root : state_->project.root / "Content";
                    if (!create_default_terrain_material(state_->scene, *state_->renderer, asset_root).valid())
                        return fail("Default terrain material could not be created");
                }

                scene::terrain_component staged;
                staged.size = descriptor.size;
                staged.subdivisions = descriptor.sample_resolution - 1u;
                staged.patch_quads = descriptor.patch_quads;
                staged.height_scale = descriptor.maximum_elevation - descriptor.minimum_elevation;
                staged.material = state_->scene.terrain_material;
                const scene::terrain_generation_descriptor generation{
                    .generator_id = descriptor.source == scene::terrain_initial_source::procedural
                                        ? "arc.terrain.domain_warped.v1"
                                        : "arc.terrain.flat.v1",
                    .seed = descriptor.procedural_seed,
                    .minimum_elevation = descriptor.minimum_elevation,
                    .maximum_elevation = descriptor.maximum_elevation};
                const auto generated = scene::generate_terrain(staged, generation);
                if (!generated.succeeded) return fail(generated.message);

                const auto created = state_->scene.scene.create();
                state_->scene.scene.emplace<scene::name_component>(created, "Terrain");
                state_->scene.scene.emplace<scene::tag_component>(created, "Environment");
                state_->scene.scene.emplace<scene::active_component>(created);
                state_->scene.scene.emplace<scene::transform_component>(created);
                state_->scene.scene.emplace<scene::selection_component>(created, false);
                state_->scene.scene.emplace<scene::terrain_component>(created, std::move(staged));
                ensure_scene_authoring_metadata(state_->scene);
                if (payload.parent.valid() &&
                    !scene::reparent(state_->scene.scene, created, to_scene_entity(payload.parent), {},
                                     scene::reparent_transform_policy::preserve_local))
                {
                    state_->scene.scene.destroy(created);
                    return fail("Terrain parent is missing or invalid");
                }
                if (!synchronize_terrain_render_resource(state_->scene, *state_->renderer, created))
                {
                    state_->scene.scene.destroy(created);
                    return fail("Terrain render resource could not be created");
                }
                select_entity(state_->scene.scene, created, state_->scene.selected_entity);
                state_->scene.terrain_entity = created;
                state_->terrain_operation = {.operation_id = state_->next_terrain_operation_id++,
                                             .state = host_terrain_operation_state::completed,
                                             .progress = 1.0f,
                                             .label = "Create Terrain",
                                             .entity = to_host_entity(created)};
                push_event(state_->events, state_->event_sequence, host_event_type::terrain_operation_changed,
                           "Terrain creation completed", created);
                push_event(state_->events, state_->event_sequence, host_event_type::entity_created, "Created terrain",
                           created);
                return success("{\"entity\":" + to_json(to_host_entity(created)) +
                               ",\"operationId\":" + std::to_string(state_->terrain_operation.operation_id) + '}');
            }
            else if constexpr (std::is_same_v<command_type, host_generate_terrain_command>)
            {
                const auto entity = to_scene_entity(payload.entity);
                auto* terrain = state_->scene.scene.try_get<scene::terrain_component>(entity);
                if (!terrain) return fail("Terrain generation requires a terrain entity", entity);
                const auto estimate = scene::estimate_terrain_memory(terrain->subdivisions + 1u);
                if (estimate.history_bytes > 64ull * 1024ull * 1024ull)
                    return fail("Terrain generation exceeds the configured undo budget", entity);
                auto staged = *terrain;
                const auto generated =
                    scene::generate_terrain(staged, {.generator_id = payload.generator_id,
                                                     .seed = payload.seed,
                                                     .minimum_elevation = payload.minimum_elevation,
                                                     .maximum_elevation = payload.maximum_elevation});
                if (!generated.succeeded) return fail(generated.message, entity);
                *terrain = std::move(staged);
                state_->scene.terrain_render_proxies.erase(entity_guid_of(state_->scene, entity), *state_->renderer);
                if (!synchronize_terrain_render_resource(state_->scene, *state_->renderer, entity))
                    return fail("Generated terrain could not be published to the renderer", entity);
                push_event(state_->events, state_->event_sequence, host_event_type::terrain_operation_changed,
                           "Terrain generation completed", entity);
                return success("{\"entity\":" + to_json(payload.entity) + '}');
            }
            else if constexpr (std::is_same_v<command_type, host_import_terrain_heightmap_command>)
            {
                const auto entity = to_scene_entity(payload.entity);
                auto* terrain = state_->scene.scene.try_get<scene::terrain_component>(entity);
                if (!terrain) return fail("Heightmap import requires a terrain entity", entity);
                const auto estimate = scene::estimate_terrain_memory(payload.target_resolution);
                if (estimate.history_bytes > 64ull * 1024ull * 1024ull)
                    return fail("Heightmap import exceeds the configured undo budget", entity);
                const auto path = payload.path.is_absolute() ? payload.path : state_->project.root / payload.path;
                scene::terrain_heightmap heightmap;
                const auto loaded = load_terrain_heightmap(path, payload.raw_width, payload.raw_height, heightmap);
                if (!loaded.succeeded) return fail(loaded.message, entity);
                auto staged = *terrain;
                const auto imported = scene::import_terrain_heightmap(staged, heightmap,
                                                                      {.target_resolution = payload.target_resolution,
                                                                       .physical_size = payload.physical_size,
                                                                       .minimum_elevation = payload.minimum_elevation,
                                                                       .maximum_elevation = payload.maximum_elevation,
                                                                       .flip_x = payload.flip_x,
                                                                       .flip_z = payload.flip_z,
                                                                       .normalize_source_range = payload.normalize});
                if (!imported.succeeded) return fail(imported.message, entity);
                *terrain = std::move(staged);
                state_->scene.terrain_render_proxies.erase(entity_guid_of(state_->scene, entity), *state_->renderer);
                if (!synchronize_terrain_render_resource(state_->scene, *state_->renderer, entity))
                    return fail("Imported terrain could not be published to the renderer", entity);
                push_event(state_->events, state_->event_sequence, host_event_type::terrain_operation_changed,
                           "Terrain heightmap import completed", entity);
                return success("{\"entity\":" + to_json(payload.entity) +
                               ",\"sourceWidth\":" + std::to_string(heightmap.width) +
                               ",\"sourceHeight\":" + std::to_string(heightmap.height) + '}');
            }
            else if constexpr (std::is_same_v<command_type, host_export_terrain_heightmap_command>)
            {
                const auto entity = to_scene_entity(payload.entity);
                const auto* terrain = state_->scene.scene.try_get<scene::terrain_component>(entity);
                if (!terrain) return fail("Heightmap export requires a terrain entity", entity);
                const auto heightmap = scene::export_terrain_heightmap(
                    *terrain,
                    {.format = payload.path.extension() == ".png" ? scene::terrain_heightmap_format::png16
                                                                  : scene::terrain_heightmap_format::raw_r16_le,
                     .minimum_elevation = payload.minimum_elevation,
                     .maximum_elevation = payload.maximum_elevation});
                const auto path = payload.path.is_absolute() ? payload.path : state_->project.root / payload.path;
                const auto saved = save_terrain_heightmap(path, heightmap);
                if (!saved.succeeded) return fail(saved.message, entity);
                push_event(state_->events, state_->event_sequence, host_event_type::terrain_operation_changed,
                           "Terrain heightmap export completed", entity);
                return success("{\"path\":" + to_json_string(path.generic_string()) + '}');
            }
            else if constexpr (std::is_same_v<command_type, host_resample_terrain_command>)
            {
                const auto entity = to_scene_entity(payload.entity);
                auto* terrain = state_->scene.scene.try_get<scene::terrain_component>(entity);
                if (!terrain) return fail("Terrain resampling requires a terrain entity", entity);
                const auto estimate = scene::estimate_terrain_memory(payload.resolution);
                if (estimate.history_bytes > 64ull * 1024ull * 1024ull)
                    return fail("Terrain resampling exceeds the configured undo budget", entity);
                auto staged = *terrain;
                const auto resampled = scene::resample_terrain(
                    staged, {.sample_resolution = payload.resolution, .physical_size = payload.physical_size});
                if (!resampled.succeeded) return fail(resampled.message, entity);
                *terrain = std::move(staged);
                state_->scene.terrain_render_proxies.erase(entity_guid_of(state_->scene, entity), *state_->renderer);
                if (!synchronize_terrain_render_resource(state_->scene, *state_->renderer, entity))
                    return fail("Resampled terrain could not be published to the renderer", entity);
                push_event(state_->events, state_->event_sequence, host_event_type::terrain_operation_changed,
                           "Terrain resampling completed", entity);
                return success("{\"entity\":" + to_json(payload.entity) + '}');
            }
            else if constexpr (std::is_same_v<command_type, host_cancel_terrain_operation_command>)
            {
                if (state_->terrain_operation.operation_id != payload.operation_id ||
                    (state_->terrain_operation.state != host_terrain_operation_state::queued &&
                     state_->terrain_operation.state != host_terrain_operation_state::running))
                    return fail("Terrain operation is not active");
                state_->terrain_operation.state = host_terrain_operation_state::cancelled;
                state_->terrain_operation.message = "Operation cancelled before publication";
                push_event(state_->events, state_->event_sequence, host_event_type::terrain_operation_changed,
                           state_->terrain_operation.message);
                return success("{\"operationId\":" + std::to_string(payload.operation_id) + '}');
            }
            else if constexpr (std::is_same_v<command_type, host_set_terrain_layer_command>)
            {
                const auto entity = to_scene_entity(payload.entity);
                auto* terrain = state_->scene.scene.try_get<scene::terrain_component>(entity);
                if (!terrain || payload.layer >= 4u)
                    return fail("Terrain layer assignment requires a valid terrain and layer 0-3", entity);
                render::texture_handle texture;
                std::filesystem::path resolved_path;
                if (!payload.path.empty())
                {
                    const auto path = resolve_project_asset(state_->assets.root, payload.path);
                    if (!path || !render::is_supported_texture_asset(*path))
                        return fail("Terrain layer must reference a supported project texture", entity);
                    auto loaded = render::load_texture_asset(*path);
                    if (!loaded.succeeded())
                        return fail("Terrain layer texture failed to load: " + loaded.message, entity);
                    texture = state_->renderer->create_texture(std::move(loaded.texture));
                    if (!texture.valid()) return fail("Terrain layer texture could not be uploaded", entity);
                    state_->scene.default_textures.push_back(texture);
                    resolved_path = *path;
                }
                state_->scene.terrain_material_descriptor.domain = render::material_domain::terrain;
                state_->scene.terrain_material_descriptor.terrain_layers[payload.layer].base_color_texture = texture;
                if (!state_->renderer->update_material(terrain->material, state_->scene.terrain_material_descriptor))
                    return fail("Terrain material could not be updated", entity);
                state_->scene.terrain_layer_paths[payload.layer] = std::move(resolved_path);
                ++terrain->content_revision;
                push_event(state_->events, state_->event_sequence, host_event_type::component_changed,
                           "Terrain layer assigned", entity);
                return success("{\"entity\":" + to_json(payload.entity) + '}');
            }
            else if constexpr (std::is_same_v<command_type, host_terrain_stroke_command>)
            {
                const auto entity = to_scene_entity(payload.entity);
                auto* terrain = state_->scene.scene.try_get<scene::terrain_component>(entity);
                auto* terrain_transform = state_->scene.scene.try_get<scene::transform_component>(entity);
                const auto* camera = state_->scene.scene.try_get<scene::camera_component>(state_->scene.camera_entity);
                const auto* camera_transform =
                    state_->scene.scene.try_get<scene::transform_component>(state_->scene.camera_entity);
                if (!terrain || !terrain_transform || !camera || !camera_transform)
                    return fail("Terrain stroke is missing terrain or viewport camera data", entity);
                if (state_->viewport_tool.tool != host_viewport_tool::terrain)
                    return fail("Terrain strokes require Terrain viewport mode", entity);
                editor_viewport viewport;
                viewport.set_size(static_cast<float>(state_->viewport_options.width),
                                  static_cast<float>(state_->viewport_options.height));
                if (!viewport.valid()) return fail("Terrain stroke requires a valid viewport", entity);
                scene::update_world_transforms(state_->scene.scene);
                const auto ray = screen_ray_from_camera(*camera, *camera_transform, viewport,
                                                        static_cast<float>(payload.x), static_cast<float>(payload.y));
                math::matrix4f inverse_terrain{};
                const auto terrain_world =
                    terrain_transform->dirty ? scene::local_matrix(*terrain_transform) : terrain_transform->world;
                if (!scene::inverse_affine(terrain_world, inverse_terrain))
                    return fail("Terrain transform cannot be inverted", entity);
                const auto local_origin = math::transform_point(inverse_terrain, ray.origin);
                const auto local_direction = math::normalize(math::transform_vector(inverse_terrain, ray.direction));
                const auto* proxy = state_->scene.terrain_render_proxies.find(entity_guid_of(state_->scene, entity));
                const auto* resource = proxy ? state_->renderer->terrain_data_for(proxy->handle) : nullptr;
                const auto hit =
                    resource ? scene::raycast_terrain(*terrain, resource->hierarchy, local_origin, local_direction)
                             : scene::raycast_terrain(*terrain, local_origin, local_direction);
                if (!hit.hit)
                {
                    state_->terrain_brush_local_position.reset();
                    state_->terrain_stroke_previous_position.reset();
                    return success("{\"hit\":false}");
                }
                state_->terrain_brush_local_position = hit.position;
                if (payload.phase == host_edit_phase::commit)
                {
                    state_->terrain_stroke_previous_position.reset();
                    push_event(state_->events, state_->event_sequence, host_event_type::terrain_stroke_committed,
                               "Terrain stroke committed", entity,
                               "{\"revision\":" + std::to_string(terrain->content_revision) + '}');
                    return success("{\"hit\":true,\"revision\":" + std::to_string(terrain->content_revision) + '}');
                }
                state_->terrain_brush.invert = payload.invert;
                if (payload.phase == host_edit_phase::begin)
                {
                    state_->terrain_flatten_height_captured = true;
                    state_->terrain_brush.flatten_height = hit.position[1];
                    state_->terrain_stroke_previous_position = hit.position;
                }
                scene::terrain_dirty_region dirty{};
                const auto merge_dirty = [&dirty](const scene::terrain_dirty_region& stamp)
                {
                    if (!stamp.valid) return;
                    if (!dirty.valid)
                    {
                        dirty = stamp;
                        return;
                    }
                    dirty.min_x = std::min(dirty.min_x, stamp.min_x);
                    dirty.min_z = std::min(dirty.min_z, stamp.min_z);
                    dirty.max_x = std::max(dirty.max_x, stamp.max_x);
                    dirty.max_z = std::max(dirty.max_z, stamp.max_z);
                    dirty.heights_changed = dirty.heights_changed || stamp.heights_changed;
                    dirty.weights_changed = dirty.weights_changed || stamp.weights_changed;
                };
                const auto previous = state_->terrain_stroke_previous_position.value_or(hit.position);
                const auto offset = hit.position - previous;
                const auto distance = math::length(offset);
                const auto sample_interval = terrain->subdivisions > 0
                                                 ? terrain->size / static_cast<float>(terrain->subdivisions)
                                                 : terrain->size;
                const auto spacing = std::max(0.0001f, std::min(state_->terrain_brush.radius * 0.25f, sample_interval));
                const auto stamp_count = std::max(1u, static_cast<std::uint32_t>(std::ceil(distance / spacing)));
                const auto stamp_seconds = payload.elapsed_seconds / static_cast<float>(stamp_count);
                for (std::uint32_t stamp_index = 1; stamp_index <= stamp_count; ++stamp_index)
                {
                    const auto alpha = static_cast<float>(stamp_index) / static_cast<float>(stamp_count);
                    merge_dirty(scene::apply_terrain_brush(*terrain, previous + offset * alpha, state_->terrain_brush,
                                                           stamp_seconds));
                }
                state_->terrain_stroke_previous_position = hit.position;
                if (dirty.valid &&
                    !synchronize_terrain_render_resource(state_->scene, *state_->renderer, entity, &dirty))
                    return fail("Terrain runtime chunks could not be updated", entity);
                return success("{\"hit\":true,\"revision\":" + std::to_string(terrain->content_revision) + '}');
            }
            else if constexpr (std::is_same_v<command_type, host_terrain_hover_command>)
            {
                const auto entity = to_scene_entity(payload.entity);
                if (payload.clear)
                {
                    state_->terrain_brush_local_position.reset();
                    return success(to_json(terrain_tool_snapshot()));
                }
                const auto* terrain = state_->scene.scene.try_get<scene::terrain_component>(entity);
                const auto* terrain_transform = state_->scene.scene.try_get<scene::transform_component>(entity);
                const auto* camera = state_->scene.scene.try_get<scene::camera_component>(state_->scene.camera_entity);
                const auto* camera_transform =
                    state_->scene.scene.try_get<scene::transform_component>(state_->scene.camera_entity);
                if (!terrain || !terrain_transform || !camera || !camera_transform)
                    return fail("Terrain hover is missing terrain or viewport camera data", entity);
                editor_viewport viewport;
                viewport.set_size(static_cast<float>(state_->viewport_options.width),
                                  static_cast<float>(state_->viewport_options.height));
                if (!viewport.valid()) return fail("Terrain hover requires a valid viewport", entity);
                scene::update_world_transforms(state_->scene.scene);
                const auto ray = screen_ray_from_camera(*camera, *camera_transform, viewport,
                                                        static_cast<float>(payload.x), static_cast<float>(payload.y));
                math::matrix4f inverse_terrain;
                const auto terrain_world =
                    terrain_transform->dirty ? scene::local_matrix(*terrain_transform) : terrain_transform->world;
                if (!scene::inverse_affine(terrain_world, inverse_terrain))
                    return fail("Terrain transform cannot be inverted", entity);
                const auto local_origin = math::transform_point(inverse_terrain, ray.origin);
                const auto local_direction = math::normalize(math::transform_vector(inverse_terrain, ray.direction));
                const auto* proxy = state_->scene.terrain_render_proxies.find(entity_guid_of(state_->scene, entity));
                const auto* resource = proxy ? state_->renderer->terrain_data_for(proxy->handle) : nullptr;
                const auto hit =
                    resource ? scene::raycast_terrain(*terrain, resource->hierarchy, local_origin, local_direction)
                             : scene::raycast_terrain(*terrain, local_origin, local_direction);
                if (hit.hit)
                    state_->terrain_brush_local_position = hit.position;
                else
                    state_->terrain_brush_local_position.reset();
                return success(to_json(terrain_tool_snapshot()));
            }
            else if constexpr (std::is_same_v<command_type, host_set_entity_material_command>)
            {
                const auto entity = to_scene_entity(payload.entity);
                const auto targets = edit_targets(state_->scene.scene, entity, payload.apply_to_selection);
                if (targets.empty()) return fail("Cannot edit a missing or unselected mesh renderer", entity);
                if (std::any_of(targets.begin(), targets.end(), [&](ecs::entity target)
                                { return !state_->scene.scene.has<scene::mesh_renderer_component>(target); }))
                    return fail("Every selected entity must have an editable mesh renderer component", entity);
                const auto resolved_material = resolve_editor_asset(state_->assets, state_->asset_registry.get(),
                                                                    state_->project.root, payload.path);
                if (!resolved_material || !is_material_asset_path(resolved_material->path))
                    return fail("Material must be an .arcmat project or built-in asset", entity);
                std::string message;
                for (const auto target : targets)
                {
                    if (!apply_material_asset_to_entity(state_->scene.material_library, *state_->renderer,
                                                        resolved_material->asset_root, resolved_material->path,
                                                        state_->scene.scene, target, &message))
                        return fail(message.empty() ? "Material assignment failed" : message, target);
                }
                ensure_scene_authoring_metadata(state_->scene);
                const auto registry_path = arc::assets::normalize_asset_path(payload.path);
                const auto material =
                    state_->asset_registry
                        ? state_->asset_registry->resolve(registry_path, arc::assets::asset_types::material)
                        : arc::assets::asset_reference{.expected_type = arc::assets::asset_types::material,
                                                       .path_hint = registry_path};
                for (const auto target : targets)
                {
                    const auto guid = entity_guid_of(state_->scene, target);
                    auto* binding = find_asset_binding(state_->scene, guid);
                    if (!binding)
                    {
                        state_->scene.asset_bindings.push_back({.entity = guid});
                        binding = &state_->scene.asset_bindings.back();
                    }
                    binding->material = material;
                }
                push_event(state_->events, state_->event_sequence, host_event_type::component_changed, message, entity);
                return success("{\"entity\":" + to_json(payload.entity) +
                               ",\"changed\":" + std::to_string(targets.size()) + '}');
            }
            else if constexpr (std::is_same_v<command_type, host_component_operation_command>)
            {
                std::vector<ecs::entity> targets;
                state_->scene.scene.view<scene::selection_component>().each(
                    [&](ecs::entity entity, const scene::selection_component& selection)
                    {
                        if (selection.selected) targets.push_back(entity);
                    });
                if (targets.empty() && state_->scene.scene.alive(state_->scene.selected_entity))
                    targets.push_back(state_->scene.selected_entity);
                if (targets.empty()) return fail("Component operation requires a selected entity");

                const bool known = payload.component == "transform" || payload.component == "camera" ||
                                   payload.component == "meshRenderer" || payload.component == "directionalLight" ||
                                   payload.component == "pointLight" || payload.component == "spotLight" ||
                                   payload.component == "areaLight" || payload.component == "renderLayer" ||
                                   payload.component == "mobility";
                const auto* project_schema = find_project_component(state_->project_module, payload.component);
                if (!known && !project_schema) return fail("Component is not supported by generic operations");
                if (payload.operation == host_component_operation::remove && payload.component == "transform")
                    return fail("Transform is required and cannot be removed");
                if (payload.operation == host_component_operation::remove && payload.component == "camera" &&
                    std::find(targets.begin(), targets.end(), state_->scene.camera_entity) != targets.end())
                    return fail("The editor viewport camera cannot be removed");

                const auto apply = [&]<class Component>(ecs::entity entity, Component reset_value)
                {
                    if (payload.operation == host_component_operation::remove)
                        return state_->scene.scene.remove<Component>(entity);
                    if (auto* current = state_->scene.scene.try_get<Component>(entity))
                    {
                        if (payload.operation == host_component_operation::add) return false;
                        *current = std::move(reset_value);
                        return true;
                    }
                    state_->scene.scene.emplace<Component>(entity, std::move(reset_value));
                    return true;
                };

                std::size_t changed = 0;
                for (const auto entity : targets)
                {
                    bool entity_changed = false;
                    if (project_schema)
                    {
                        const auto guid = entity_guid_of(state_->scene, entity);
                        auto records = project_component_records(state_->scene, guid);
                        const auto existing_key = project_component_record_key(records, *project_schema);
                        const bool exists = existing_key.has_value();
                        if (payload.operation == host_component_operation::remove && exists)
                        {
                            records.erase(*existing_key);
                            entity_changed = true;
                        }
                        else if (payload.operation == host_component_operation::add && !exists)
                        {
                            records[project_schema->canonical_name] = default_project_component(*project_schema);
                            entity_changed = true;
                        }
                        else if (payload.operation == host_component_operation::reset && exists)
                        {
                            if (*existing_key != project_schema->canonical_name) records.erase(*existing_key);
                            records[project_schema->canonical_name] = default_project_component(*project_schema);
                            entity_changed = true;
                        }
                        if (entity_changed) set_project_component_records(state_->scene, guid, records);
                    }
                    else if (payload.component == "transform")
                    {
                        scene::transform_component value{};
                        value.dirty = true;
                        entity_changed = apply(entity, std::move(value));
                    }
                    else if (payload.component == "camera")
                        entity_changed = apply(entity, scene::camera_component{});
                    else if (payload.component == "meshRenderer")
                    {
                        scene::mesh_renderer_component value;
                        value.mesh = state_->scene.default_mesh;
                        value.material = state_->scene.default_material;
                        if (const auto* current = state_->scene.scene.try_get<scene::mesh_renderer_component>(entity))
                        {
                            value.mesh = current->mesh;
                            value.material = current->material;
                        }
                        entity_changed = apply(entity, std::move(value));
                    }
                    else if (payload.component == "directionalLight")
                        entity_changed = apply(entity, scene::directional_light_component{});
                    else if (payload.component == "pointLight")
                        entity_changed = apply(entity, scene::point_light_component{});
                    else if (payload.component == "spotLight")
                        entity_changed = apply(entity, scene::spot_light_component{});
                    else if (payload.component == "areaLight")
                        entity_changed = apply(entity, scene::area_light_component{});
                    else if (payload.component == "renderLayer")
                        entity_changed = apply(entity, scene::render_layer_component{});
                    else if (payload.component == "mobility")
                        entity_changed = apply(entity, scene::mobility_component{});
                    if (entity_changed) ++changed;
                }
                if (!changed)
                    return fail(payload.operation == host_component_operation::add
                                    ? "Every selected entity already has this component"
                                    : "No selected entity has this component");
                scene::update_world_transforms(state_->scene.scene);
                push_event(state_->events, state_->event_sequence, host_event_type::component_changed,
                           "Component operation applied to " + std::to_string(changed) + " entity(s)",
                           state_->scene.selected_entity);
                return success("{\"changed\":" + std::to_string(changed) + '}');
            }
            else if constexpr (std::is_same_v<command_type, host_patch_project_component_command>)
            {
                const auto* schema = find_project_component(state_->project_module, payload.component);
                if (!schema) return fail("Project component schema is not loaded");
                const auto field = std::find_if(schema->fields.begin(), schema->fields.end(),
                                                [&](const auto& candidate) { return candidate.name == payload.field; });
                if (field == schema->fields.end()) return fail("Project component field is unknown");
                if (!has_flag(field->flags, project::game_field_flags_v1::editable) ||
                    has_flag(field->flags, project::game_field_flags_v1::read_only))
                    return fail("Project component field is read-only");
                nlohmann::json value = nlohmann::json::parse(payload.value_json, nullptr, false);
                std::string validation_error;
                if (value.is_discarded() || !validate_project_field_value(*field, value, validation_error))
                    return fail(validation_error.empty() ? "Project component field contains invalid JSON"
                                                         : validation_error);
                const auto reference = value.is_string() ? value.get<std::string>() : std::string{};
                if (field->kind == project::game_field_kind_v1::entity_reference && !reference.empty())
                {
                    const auto referenced_guid = ecs::parse_entity_guid(reference);
                    const auto referenced_entity =
                        referenced_guid ? find_entity_by_guid(state_->scene, *referenced_guid) : ecs::entity{};
                    if (!referenced_guid || !state_->scene.scene.alive(referenced_entity))
                        return fail("Project entity reference does not resolve in the open scene");
                    if (!field->entity_component_restriction.empty())
                    {
                        const auto* required =
                            find_project_component(state_->project_module, field->entity_component_restriction);
                        const auto records = project_component_records(state_->scene, *referenced_guid);
                        if (!required || !project_component_record_key(records, *required))
                            return fail("Project entity reference does not satisfy its component restriction");
                    }
                }
                if (field->kind == project::game_field_kind_v1::asset_reference && !reference.empty())
                {
                    const auto referenced_guid = assets::parse_asset_guid(reference);
                    const auto referenced_asset = referenced_guid && state_->asset_registry
                                                      ? state_->asset_registry->find(*referenced_guid)
                                                      : std::nullopt;
                    if (!referenced_guid || !referenced_asset)
                        return fail("Project asset reference does not resolve in the selected project");
                    if (!field->asset_type_restriction.empty())
                    {
                        const auto required_type = assets::parse_asset_type_id(field->asset_type_restriction);
                        if (!required_type || referenced_asset->type != *required_type)
                            return fail("Project asset reference has the wrong asset type");
                    }
                }

                std::vector<ecs::entity> targets;
                state_->scene.scene.view<scene::selection_component>().each(
                    [&](ecs::entity entity, const scene::selection_component& selection)
                    {
                        if (selection.selected) targets.push_back(entity);
                    });
                if (targets.empty() && state_->scene.scene.alive(state_->scene.selected_entity))
                    targets.push_back(state_->scene.selected_entity);
                std::size_t changed{};
                for (const auto entity : targets)
                {
                    const auto guid = entity_guid_of(state_->scene, entity);
                    auto records = project_component_records(state_->scene, guid);
                    const auto record_key = project_component_record_key(records, *schema);
                    if (!record_key) continue;
                    auto& component = records[*record_key];
                    component[field->name] = value;
                    component["typeId"] = schema->stable_id;
                    component["version"] = schema->schema_version;
                    set_project_component_records(state_->scene, guid, records);
                    ++changed;
                }
                if (!changed) return fail("No selected entity has this project component");
                push_event(state_->events, state_->event_sequence, host_event_type::component_changed,
                           "Project component field updated", state_->scene.selected_entity);
                return success("{\"changed\":" + std::to_string(changed) + '}');
            }
            else if constexpr (std::is_same_v<command_type, host_set_world_environment_command>)
            {
                const auto entity = to_scene_entity(payload.environment.entity);
                auto settings = scene::read_world_environment_settings(state_->scene.scene, entity);
                if (!settings) return fail("Cannot edit a missing or incomplete world environment", entity);

                const auto next_settings = apply_host_world_environment_snapshot(payload.environment, *settings);
                const auto validation = scene::validate_world_environment(next_settings);
                if (!validation.valid) return fail(validation_message(validation), entity);
                if (!scene::set_world_environment_settings(state_->scene.scene, entity, next_settings))
                    return fail("World environment update could not be applied", entity);
                push_event(state_->events, state_->event_sequence, host_event_type::component_changed,
                           "World environment changed", entity);
                return success("{\"entity\":" + to_json(payload.environment.entity) + '}');
            }
            else if constexpr (std::is_same_v<command_type, host_apply_world_environment_preset_command>)
            {
                const auto entity = to_scene_entity(payload.entity);
                auto settings = scene::read_world_environment_settings(state_->scene.scene, entity);
                if (!settings) return fail("Cannot apply a preset to a missing world environment", entity);

                scene::apply_world_environment_preset(to_scene_preset(payload.preset), *settings);
                if (!scene::set_world_environment_settings(state_->scene.scene, entity, *settings))
                    return fail("World environment preset could not be applied", entity);
                push_event(state_->events, state_->event_sequence, host_event_type::component_changed,
                           "World environment preset applied", entity);
                return success("{\"entity\":" + to_json(payload.entity) + '}');
            }
            else if constexpr (std::is_same_v<command_type, host_set_environment_hdri_command>)
            {
                const auto entity = to_scene_entity(payload.entity);
                auto* world = state_->scene.scene.try_get<scene::world_environment_component>(entity);
                auto* lighting = state_->scene.scene.try_get<scene::environment_lighting_component>(entity);
                if (!world || !lighting) return fail("Cannot assign an HDRI to a missing world environment", entity);

                const auto update_environment_resource = [&]
                {
                    if (!lighting->environment.valid()) return;
                    render::environment_descriptor environment;
                    environment.name = "World Environment HDRI";
                    environment.equirectangular_texture = world->hdri_texture;
                    environment.fallback_color = world->solid_color;
                    environment.intensity = world->radiance_intensity;
                    environment.diffuse_irradiance = lighting->constant_color;
                    environment.diffuse_intensity = lighting->diffuse_intensity;
                    state_->renderer->update_environment(lighting->environment, std::move(environment));
                };
                if (payload.path.empty())
                {
                    world->hdri_texture = {};
                    lighting->hdri_texture = {};
                    state_->scene.world_environment_hdri_path.clear();
                    update_environment_resource();
                    push_event(state_->events, state_->event_sequence, host_event_type::component_changed,
                               "World environment HDRI cleared", entity);
                    return success("{\"entity\":" + to_json(payload.entity) + '}');
                }

                const auto path = resolve_project_asset(state_->assets.root, payload.path);
                if (!path || !render::is_supported_texture_asset(*path))
                    return fail("Environment texture must be a supported project asset", entity);
                const auto loaded = render::load_texture_asset(*path);
                if (!loaded.succeeded())
                    return fail(loaded.message.empty() ? "Failed to load environment texture" : loaded.message, entity);

                auto texture = loaded.texture;
                texture.name = path->filename().string();
                if (world->hdri_texture.valid())
                    state_->renderer->update_texture(world->hdri_texture, std::move(texture));
                else
                    world->hdri_texture = state_->renderer->create_texture(std::move(texture));
                lighting->hdri_texture = world->hdri_texture;
                std::error_code relative_error;
                state_->scene.world_environment_hdri_path =
                    std::filesystem::relative(*path, state_->assets.root, relative_error);
                if (relative_error) state_->scene.world_environment_hdri_path = path->filename();
                update_environment_resource();
                push_event(state_->events, state_->event_sequence, host_event_type::component_changed,
                           "World environment HDRI changed", entity);
                return success("{\"entity\":" + to_json(payload.entity) + '}');
            }
            else if constexpr (std::is_same_v<command_type, host_set_camera_projection_command>)
            {
                if (auto* camera = state_->scene.scene.try_get<scene::camera_component>(state_->scene.camera_entity))
                {
                    camera->projection = to_scene_projection(payload.projection);
                    push_event(state_->events, state_->event_sequence, host_event_type::component_changed,
                               "Camera projection changed", state_->scene.camera_entity);
                    return success("{\"entity\":" + to_json(to_host_entity(state_->scene.camera_entity)) + '}');
                }
                return fail("No editor camera is available", state_->scene.camera_entity);
            }
            else if constexpr (std::is_same_v<command_type, host_viewport_attach_command>)
            {
                state_->active_viewport_id = payload.viewport_id.empty() ? "viewport-1" : payload.viewport_id;
                state_->viewport_options.viewport_id = state_->active_viewport_id;
                state_->viewport_options.width = payload.width;
                state_->viewport_options.height = payload.height;
                return success("{\"nativeHandle\":" + std::to_string(payload.native_handle) + '}');
            }
            else if constexpr (std::is_same_v<command_type, host_viewport_create_command>)
            {
                state_->active_viewport_id = payload.viewport_id.empty() ? "viewport-1" : payload.viewport_id;
                state_->viewport_options.viewport_id = state_->active_viewport_id;
                state_->viewport_options.width = payload.width;
                state_->viewport_options.height = payload.height;
                return success("{\"output\":\"" +
                               std::string(payload.output == host_viewport_output_type::shared_texture
                                               ? "sharedTexture"
                                               : "nativeWindow") +
                               "\"}");
            }
            else if constexpr (std::is_same_v<command_type, host_viewport_resize_command>)
            {
                if (payload.viewport_id != state_->active_viewport_id) return fail("Viewport is not attached");
                state_->viewport_options.width = payload.width;
                state_->viewport_options.height = payload.height;
                return success();
            }
            else if constexpr (std::is_same_v<command_type, host_revert_prefab_override_command>)
            {
                const ecs::entity entity = to_scene_entity(payload.entity);
                auto* instance = state_->scene.scene.try_get<scene::prefab_instance_component>(entity);
                if (!instance) return fail("Entity is not a prefab instance", entity);
                const auto source = ecs::parse_entity_guid(payload.source_entity);
                const auto component = ecs::parse_component_type_id(payload.component_id);
                if (!source || !component) return fail("Prefab override identity is invalid", entity);
                const auto kind =
                    payload.kind == "componentAdded"       ? ecs::prefab_override_kind::component_added
                    : payload.kind == "componentRemoved"   ? ecs::prefab_override_kind::component_removed
                    : payload.kind == "sourceChildRemoved" ? ecs::prefab_override_kind::source_child_removed
                    : payload.kind == "instanceChildAdded" ? ecs::prefab_override_kind::instance_child_added
                                                           : ecs::prefab_override_kind::field;
                const ecs::prefab_override_key key{
                    .source_entity = *source, .component = *component, .field = payload.field_id, .kind = kind};
                if (!ecs::revert_prefab_override(*instance, key)) return fail("Prefab override was not found", entity);
                push_event(state_->events, state_->event_sequence, host_event_type::component_changed,
                           "Prefab override reverted", entity);
                return success();
            }
            else if constexpr (std::is_same_v<command_type, host_viewport_detach_command>)
            {
                if (payload.viewport_id == state_->active_viewport_id)
                {
                    state_->viewport_options.width = 0;
                    state_->viewport_options.height = 0;
                    state_->viewport_submitted = false;
                }
                return success();
            }
            else if constexpr (std::is_same_v<command_type, host_viewport_frame_released_command> ||
                               std::is_same_v<command_type, host_viewport_set_visibility_command> ||
                               std::is_same_v<command_type, host_viewport_pointer_command> ||
                               std::is_same_v<command_type, host_viewport_key_command>)
            {
                return success();
            }
            else if constexpr (std::is_same_v<command_type, host_viewport_set_camera_mode_command>)
            {
                return execute(host_command_envelope{
                    .request_id = request_id,
                    .command_type = "camera.setProjection",
                    .payload = host_set_camera_projection_command{.projection = payload.projection}});
            }
            else if constexpr (std::is_same_v<command_type, host_viewport_set_render_options_command>)
            {
                if (payload.viewport_id != state_->active_viewport_id) return fail("Viewport is not attached");
                state_->viewport_options.render_mode = payload.render_mode;
                state_->viewport_options.visualization = payload.visualization;
                state_->viewport_options.overlay = payload.overlay;
                state_->viewport_options.shadows = payload.shadows;
                state_->viewport_options.grid = payload.grid;
                state_->viewport_options.realtime = payload.realtime;
                state_->viewport_options.camera_speed = payload.camera_speed;
                state_->viewport_options.anti_aliasing = payload.anti_aliasing;
                state_->viewport_options.environment = payload.environment;
                return success();
            }
            else if constexpr (std::is_same_v<command_type, host_viewport_camera_input_command>)
            {
                if (payload.viewport_id != state_->active_viewport_id) return fail("Viewport is not attached");
                if (payload.focus_selected)
                    focus_selected_entity(state_->scene.scene, state_->scene.selected_entity,
                                          state_->camera_controller);
                if (payload.orbit_x != 0.0f || payload.orbit_y != 0.0f)
                    state_->camera_controller.orbit(payload.orbit_x, payload.orbit_y);
                if (payload.look_x != 0.0f || payload.look_y != 0.0f)
                    state_->camera_controller.look(payload.look_x, payload.look_y);
                const float movement_scale = state_->viewport_options.camera_speed / 4.0f;
                if (payload.pan_x != 0.0f || payload.pan_y != 0.0f)
                    state_->camera_controller.pan(payload.pan_x * movement_scale, payload.pan_y * movement_scale);
                if (payload.forward != 0.0f) state_->camera_controller.move_forward(payload.forward * movement_scale);
                if (payload.move_right != 0.0f || payload.move_up != 0.0f || payload.move_forward != 0.0f)
                    state_->camera_controller.move(payload.move_right * movement_scale,
                                                   payload.move_up * movement_scale,
                                                   payload.move_forward * movement_scale);
                if (payload.zoom != 0.0f) state_->camera_controller.zoom(payload.zoom * movement_scale);
                if (auto* camera_transform =
                        state_->scene.scene.try_get<scene::transform_component>(state_->scene.camera_entity))
                {
                    state_->camera_controller.apply_to(*camera_transform);
                    return success("{\"entity\":" + to_json(to_host_entity(state_->scene.camera_entity)) + '}');
                }
                return fail("No editor camera is available", state_->scene.camera_entity);
            }
            else if constexpr (std::is_same_v<command_type, host_viewport_set_pose_command>)
            {
                const math::vector3f position{payload.position.x, payload.position.y, payload.position.z};
                const math::vector3f target{payload.target.x, payload.target.y, payload.target.z};
                if (!state_->camera_controller.place(position, target))
                    return fail("Viewport camera position and target must be finite and distinct");
                if (auto* camera_transform =
                        state_->scene.scene.try_get<scene::transform_component>(state_->scene.camera_entity))
                {
                    state_->camera_controller.apply_to(*camera_transform);
                    return success("{\"entity\":" + to_json(to_host_entity(state_->scene.camera_entity)) + '}');
                }
                return fail("No editor camera is available", state_->scene.camera_entity);
            }
            else if constexpr (std::is_same_v<command_type, host_history_undo_command>)
            {
                if (!state_->history.undo(state_->scene)) return fail("Nothing to undo");
                if (auto* camera_transform =
                        state_->scene.scene.try_get<scene::transform_component>(state_->scene.camera_entity))
                    state_->camera_controller.synchronize_from(*camera_transform);
                if (const auto& changed = state_->history.last_terrain_change(); changed)
                    synchronize_terrain_render_resource(state_->scene, *state_->renderer,
                                                        find_entity_by_guid(state_->scene, changed->entity),
                                                        &changed->region);
                else
                    synchronize_all_terrain_resources(state_->scene, *state_->renderer);
                push_event(state_->events, state_->event_sequence, host_event_type::scene_changed, "Undo completed",
                           state_->scene.selected_entity);
                return success();
            }
            else if constexpr (std::is_same_v<command_type, host_history_redo_command>)
            {
                if (!state_->history.redo(state_->scene)) return fail("Nothing to redo");
                if (auto* camera_transform =
                        state_->scene.scene.try_get<scene::transform_component>(state_->scene.camera_entity))
                    state_->camera_controller.synchronize_from(*camera_transform);
                if (const auto& changed = state_->history.last_terrain_change(); changed)
                    synchronize_terrain_render_resource(state_->scene, *state_->renderer,
                                                        find_entity_by_guid(state_->scene, changed->entity),
                                                        &changed->region);
                else
                    synchronize_all_terrain_resources(state_->scene, *state_->renderer);
                push_event(state_->events, state_->event_sequence, host_event_type::scene_changed, "Redo completed",
                           state_->scene.selected_entity);
                return success();
            }
            else if constexpr (std::is_same_v<command_type, host_history_begin_transaction_command>)
            {
                if (payload.id == 0 ||
                    !state_->history.begin(payload.id, payload.label.empty() ? "AI Scene Edit" : payload.label,
                                           state_->scene))
                    return fail("Could not begin edit transaction");
                return success("{\"transactionId\":" + std::to_string(payload.id) + '}');
            }
            else if constexpr (std::is_same_v<command_type, host_history_commit_transaction_command>)
            {
                if (!state_->history.transaction_matches(payload.id) ||
                    !state_->history.commit(payload.id, state_->scene))
                    return fail("Could not commit edit transaction");
                return success("{\"transactionId\":" + std::to_string(payload.id) + '}');
            }
            else if constexpr (std::is_same_v<command_type, host_history_cancel_transaction_command>)
            {
                if (!state_->history.cancel(payload.id, state_->scene))
                    return fail("Could not cancel edit transaction");
                synchronize_all_terrain_resources(state_->scene, *state_->renderer);
                state_->terrain_brush_local_position.reset();
                ++state_->scene_revision;
                push_event(state_->events, state_->event_sequence, host_event_type::scene_changed,
                           "Edit transaction cancelled");
                return success("{\"transactionId\":" + std::to_string(payload.id) + '}');
            }
            else if constexpr (std::is_same_v<command_type, host_viewport_set_tool_command>)
            {
                if (!(payload.translation_snap > 0.0f && payload.rotation_snap_degrees > 0.0f &&
                      payload.scale_snap > 0.0f))
                    return fail("Viewport snap values must be positive");
                state_->viewport_tool = payload;
                const char* tool = payload.tool == host_viewport_tool::translate ? "translate"
                                   : payload.tool == host_viewport_tool::rotate  ? "rotate"
                                   : payload.tool == host_viewport_tool::scale   ? "scale"
                                   : payload.tool == host_viewport_tool::terrain ? "terrain"
                                                                                 : "select";
                push_event(
                    state_->events, state_->event_sequence, host_event_type::component_changed, "Viewport tool changed",
                    {},
                    "{\"tool\":" + to_json_string(tool) + ",\"coordinateSpace\":" +
                        to_json_string(payload.coordinate_space == host_coordinate_space::local ? "local" : "world") +
                        ",\"snapping\":" + std::string(payload.snapping ? "true" : "false") +
                        ",\"translationSnap\":" + std::to_string(payload.translation_snap) +
                        ",\"rotationSnapDegrees\":" + std::to_string(payload.rotation_snap_degrees) +
                        ",\"scaleSnap\":" + std::to_string(payload.scale_snap) + '}');
                return success();
            }
            else if constexpr (std::is_same_v<command_type, host_viewport_pick_command>)
            {
                if (payload.viewport_id != state_->active_viewport_id) return fail("Viewport is not attached");
                const std::uint64_t pick_request_id = state_->next_pick_request_id++;
                scene::update_world_transforms(state_->scene.scene);
                editor_viewport viewport;
                viewport.set_size(static_cast<float>(state_->viewport_options.width),
                                  static_cast<float>(state_->viewport_options.height));
                const auto* camera = state_->scene.scene.try_get<scene::camera_component>(state_->scene.camera_entity);
                const auto* transform =
                    state_->scene.scene.try_get<scene::transform_component>(state_->scene.camera_entity);
                editor_pick_result cpu_fallback{};
                editor_ray pick_ray{};
                math::vector3f drop_position{};
                if (camera && transform && viewport.valid())
                {
                    pick_ray = screen_ray_from_camera(*camera, *transform, viewport, static_cast<float>(payload.x),
                                                      static_cast<float>(payload.y));
                    cpu_fallback = pick_scene_entity(state_->scene.scene, *state_->renderer, pick_ray);
                    if (cpu_fallback.entity.valid() && std::isfinite(cpu_fallback.distance) &&
                        cpu_fallback.distance > 0.0f)
                        drop_position = pick_ray.origin + pick_ray.direction * cpu_fallback.distance;
                    else
                    {
                        const float denominator = pick_ray.direction[1];
                        const float ground_distance =
                            std::abs(denominator) > 1.0e-5f ? -pick_ray.origin[1] / denominator : -1.0f;
                        drop_position = ground_distance > 0.0f ? pick_ray.origin + pick_ray.direction * ground_distance
                                                               : pick_ray.origin + pick_ray.direction * 10.0f;
                    }
                }
                state_->renderer->request_object_pick(pick_request_id, payload.x, payload.y);

                if (state_->renderer->backend())
                {
                    state_->pending_pick =
                        state::pending_viewport_pick{.request_id = pick_request_id,
                                                     .x = payload.x,
                                                     .y = payload.y,
                                                     .cpu_fallback = cpu_fallback,
                                                     .requested_after_frame = state_->viewport_frame_index};
                    return success("{\"pending\":true,\"worldPosition\":[" + std::to_string(drop_position[0]) + "," +
                                   std::to_string(drop_position[1]) + "," + std::to_string(drop_position[2]) + "]}");
                }

                const auto previous_selection = state_->scene.selected_entity;
                if (state_->scene.scene.alive(cpu_fallback.entity))
                    select_entity(state_->scene.scene, cpu_fallback.entity, state_->scene.selected_entity);
                else
                    clear_selection(state_->scene.scene, state_->scene.selected_entity);
                if (state_->scene.selected_entity != previous_selection)
                    push_event(state_->events, state_->event_sequence, host_event_type::entity_selected,
                               cpu_fallback.entity.valid() ? "Viewport entity selected" : "Viewport selection cleared",
                               cpu_fallback.entity);
                return success("{\"entity\":" + to_json(to_host_entity(cpu_fallback.entity)) + ",\"worldPosition\":[" +
                               std::to_string(drop_position[0]) + "," + std::to_string(drop_position[1]) + "," +
                               std::to_string(drop_position[2]) + "]}");
            }
            else if constexpr (std::is_same_v<command_type, host_viewport_capture_command>)
            {
                if (payload.capture_id == 0) return fail("Capture ID must be non-zero");
                render::render_frame_capture_request request{};
                request.capture_id = payload.capture_id;
                request.channels.clear();
                if (payload.color) request.channels.push_back(render::render_capture_channel::output_color);
                if (payload.scene_color) request.channels.push_back(render::render_capture_channel::scene_color);
                if (payload.depth) request.channels.push_back(render::render_capture_channel::linear_depth);
                if (payload.object_id) request.channels.push_back(render::render_capture_channel::object_id);
                if (payload.normals) request.channels.push_back(render::render_capture_channel::world_normal);
                if (payload.base_color) request.channels.push_back(render::render_capture_channel::base_color);
                if (payload.material_properties)
                    request.channels.push_back(render::render_capture_channel::material_properties);
                if (payload.emissive) request.channels.push_back(render::render_capture_channel::emissive);
                if (payload.indirect_diffuse)
                    request.channels.push_back(render::render_capture_channel::indirect_diffuse);
                if (payload.reflections) request.channels.push_back(render::render_capture_channel::reflections);
                if (payload.trace_source) request.channels.push_back(render::render_capture_channel::trace_source);
                if (payload.distance_field)
                    request.channels.push_back(render::render_capture_channel::mesh_distance_field);
                if (payload.temporal_confidence)
                    request.channels.push_back(render::render_capture_channel::temporal_confidence);
                state_->renderer->request_frame_capture(std::move(request));
                return success("{\"captureId\":" + std::to_string(payload.capture_id) + ",\"pending\":true}");
            }

            else
                return fail("Unsupported host command");
        },
        command.payload);

    if (authoring && !response.succeeded)
    {
        if (command.edit)
            state_->history.cancel(command.edit->id, state_->scene);
        else if (before)
        {
            state_->scene.terrain_render_proxies.clear(*state_->renderer);
            state_->scene = std::move(*before);
            synchronize_all_terrain_resources(state_->scene, *state_->renderer);
        }
    }
    else if (authoring && response.succeeded)
    {
        ++state_->scene_revision;
        if (command.edit && command.edit->phase == host_edit_phase::commit)
        {
            if (const auto* terrain_stroke = std::get_if<host_terrain_stroke_command>(&command.payload))
            {
                const auto terrain_entity = to_scene_entity(terrain_stroke->entity);
                state_->history.commit_terrain(command.edit->id, state_->scene,
                                               entity_guid_of(state_->scene, terrain_entity));
            }
            else
            {
                state_->history.commit(command.edit->id, state_->scene);
            }
        }
        else if (!command.edit && before)
        {
            state_->history.record(edit_label, std::move(*before), state_->scene);
        }
    }
    if (response.succeeded && std::holds_alternative<host_open_scene_command>(command.payload)) ++state_->world_epoch;
    if (response.succeeded && std::holds_alternative<host_new_scene_command>(command.payload)) ++state_->world_epoch;
    response.scene_revision = state_->scene_revision;
    response.world_epoch = state_->world_epoch;
    response.frame_revision = state_->viewport_frame_index;
    return response;
}

host_response arc_host::query(const host_query_envelope& query) const
{
    auto response = std::visit(
        [this, request_id = query.request_id](const auto& payload) -> host_response
        {
            using query_type = std::decay_t<decltype(payload)>;
            if constexpr (std::is_same_v<query_type, host_scene_hierarchy_query>)
            {
                return {.request_id = request_id, .succeeded = true, .payload_json = to_json(scene_snapshot())};
            }
            else if constexpr (std::is_same_v<query_type, host_selected_entity_query>)
            {
                return {
                    .request_id = request_id, .succeeded = true, .payload_json = to_json(selected_entity_snapshot())};
            }
            else if constexpr (std::is_same_v<query_type, host_workspace_documents_query>)
            {
                const auto history = state_->history.snapshot();
                const std::string guid =
                    state_->scene.scene_guid.valid() ? ecs::to_string(state_->scene.scene_guid) : std::string{};
                return {.request_id = request_id,
                        .succeeded = true,
                        .payload_json =
                            "{\"activeDocument\":" + to_json_string(guid) + ",\"documents\":[{\"guid\":" +
                            to_json_string(guid) + ",\"name\":" + to_json_string(state_->scene.scene_name) +
                            ",\"path\":" + to_json_string(state_->scene.active_scene_path.generic_string()) +
                            ",\"dirty\":" + std::string(history.dirty ? "true" : "false") +
                            ",\"recovered\":" + std::string(state_->scene.recovered_document ? "true" : "false") +
                            ",\"readOnly\":" + std::string(state_->project_read_only ? "true" : "false") +
                            ",\"active\":true,\"pinned\":false,\"entityCount\":" +
                            std::to_string(state_->scene.scene.live_count()) +
                            ",\"revision\":" + std::to_string(history.revision) + "}]}"};
            }
            else if constexpr (std::is_same_v<query_type, host_scene_entities_query>)
            {
                auto snapshot = scene_snapshot();
                if (!payload.search.empty())
                {
                    std::string search = payload.search;
                    std::transform(search.begin(), search.end(), search.begin(),
                                   [](unsigned char value) { return static_cast<char>(std::tolower(value)); });
                    std::erase_if(snapshot.entities,
                                  [&](const host_scene_entity_snapshot& entity)
                                  {
                                      std::string name = entity.name;
                                      std::transform(name.begin(), name.end(), name.begin(), [](unsigned char value)
                                                     { return static_cast<char>(std::tolower(value)); });
                                      return name.find(search) == std::string::npos &&
                                             entity.guid.find(search) == std::string::npos;
                                  });
                }
                snapshot.total_entity_count = snapshot.entities.size();
                snapshot.offset = std::min(payload.offset, snapshot.entities.size());
                const std::size_t end = std::min(snapshot.offset + payload.limit, snapshot.entities.size());
                snapshot.has_more = end < snapshot.entities.size();
                snapshot.entities = std::vector<host_scene_entity_snapshot>(
                    snapshot.entities.begin() + static_cast<std::ptrdiff_t>(snapshot.offset),
                    snapshot.entities.begin() + static_cast<std::ptrdiff_t>(end));
                return {.request_id = request_id, .succeeded = true, .payload_json = to_json(snapshot)};
            }
            else if constexpr (std::is_same_v<query_type, host_entity_by_guid_query>)
            {
                const auto snapshot = entity_snapshot(payload.guid);
                if (!snapshot.entity.valid())
                    return {.request_id = request_id, .succeeded = false, .error = "Entity GUID was not found"};
                return {.request_id = request_id, .succeeded = true, .payload_json = to_json(snapshot)};
            }
            else if constexpr (std::is_same_v<query_type, host_scene_spatial_query>)
            {
                // Spatial queries are editor-authoritative reads and may arrive before the
                // next render extraction after an edit. Refresh hierarchy-derived world
                // transforms so bounds and exact ray tests observe the latest scene state.
                scene::update_world_transforms(state_->scene.scene);
                if (payload.kind == host_spatial_query_kind::raycast)
                {
                    math::vector3f direction{payload.direction.x, payload.direction.y, payload.direction.z};
                    const float direction_length = math::length(direction);
                    if (!std::isfinite(direction_length) || direction_length <= 0.000001f)
                        return {.request_id = request_id,
                                .succeeded = false,
                                .error = "Ray direction must be finite and non-zero"};
                    direction = math::mul(direction, 1.0f / direction_length);
                    const auto hit =
                        pick_scene_entity(state_->scene.scene, *state_->renderer,
                                          editor_ray{.origin = {payload.origin.x, payload.origin.y, payload.origin.z},
                                                     .direction = direction});
                    if (!state_->scene.scene.alive(hit.entity))
                        return {.request_id = request_id, .succeeded = true, .payload_json = "{\"hits\":[]}"};
                    const std::string json = "{\"hits\":[{\"guid\":" +
                                             to_json_string(ecs::to_string(entity_guid_of(state_->scene, hit.entity))) +
                                             ",\"distance\":" + std::to_string(hit.distance) +
                                             ",\"exact\":" + std::string(hit.exact ? "true" : "false") +
                                             ",\"background\":" + std::string(hit.background ? "true" : "false") +
                                             "}]}";
                    return {.request_id = request_id, .succeeded = true, .payload_json = json};
                }

                struct spatial_match
                {
                    std::string guid;
                    float distance{};
                };
                std::vector<spatial_match> matches;
                const math::vector3f center{payload.center.x, payload.center.y, payload.center.z};
                const geometric::box3f query_bounds{
                    {center[0] - std::abs(payload.extent.x), center[1] - std::abs(payload.extent.y),
                     center[2] - std::abs(payload.extent.z)},
                    {center[0] + std::abs(payload.extent.x), center[1] + std::abs(payload.extent.y),
                     center[2] + std::abs(payload.extent.z)}};
                std::optional<render::view_frustum> viewport_frustum;
                if (payload.kind == host_spatial_query_kind::frustum)
                {
                    const auto* camera =
                        state_->scene.scene.try_get<scene::camera_component>(state_->scene.camera_entity);
                    const auto* transform =
                        state_->scene.scene.try_get<scene::transform_component>(state_->scene.camera_entity);
                    if (!camera || !transform)
                        return {.request_id = request_id,
                                .succeeded = false,
                                .error = "Viewport camera is unavailable for a frustum query"};
                    const float aspect = static_cast<float>(std::max(state_->viewport_options.width, 1u)) /
                                         static_cast<float>(std::max(state_->viewport_options.height, 1u));
                    viewport_frustum = render::make_view_frustum(scene::view_projection(*camera, *transform, aspect));
                }
                for (const auto entity : state_->scene.scene.entities())
                {
                    if (entity == state_->scene.camera_entity) continue;
                    const auto* bounds = state_->scene.scene.try_get<scene::bounds_component>(entity);
                    if (!bounds) continue;
                    const auto bounds_center = geometric::center(bounds->world_bounds);
                    const math::vector3f delta{bounds_center[0] - center[0], bounds_center[1] - center[1],
                                               bounds_center[2] - center[2]};
                    const float distance = math::length(delta);
                    const bool included = payload.kind == host_spatial_query_kind::bounds
                                              ? geometric::intersects(bounds->world_bounds, query_bounds)
                                          : payload.kind == host_spatial_query_kind::frustum
                                              ? render::intersects(*viewport_frustum, bounds->world_bounds)
                                              : distance <= std::max(payload.radius, 0.0f);
                    if (included) matches.push_back({ecs::to_string(entity_guid_of(state_->scene, entity)), distance});
                }
                std::sort(matches.begin(), matches.end(),
                          [](const spatial_match& left, const spatial_match& right)
                          {
                              if (left.distance != right.distance) return left.distance < right.distance;
                              return left.guid < right.guid;
                          });
                if (matches.size() > payload.limit) matches.resize(payload.limit);
                std::string json = "{\"hits\":[";
                for (std::size_t index = 0; index < matches.size(); ++index)
                {
                    if (index != 0) json += ',';
                    json += "{\"guid\":" + to_json_string(matches[index].guid) +
                            ",\"distance\":" + std::to_string(matches[index].distance) + '}';
                }
                json += "]}";
                return {.request_id = request_id, .succeeded = true, .payload_json = std::move(json)};
            }
            else if constexpr (std::is_same_v<query_type, host_component_schema_query>)
            {
                return {.request_id = request_id,
                        .succeeded = true,
                        .payload_json = component_schema_json(state_->project_module.component_schemas())};
            }
            else if constexpr (std::is_same_v<query_type, host_gateway_diagnostics_query>)
            {
                const auto profile = state_->renderer->last_frame_profile();
                const auto texture_resources = state_->renderer->texture_residency().resource_snapshots();
                const auto usage_name = [](render::render_resource_usage usage)
                {
                    switch (usage)
                    {
                        case render::render_resource_usage::color_attachment:
                            return "colorAttachment";
                        case render::render_resource_usage::depth_attachment:
                            return "depthAttachment";
                        case render::render_resource_usage::sampled:
                            return "sampled";
                        case render::render_resource_usage::storage:
                            return "storage";
                        case render::render_resource_usage::transfer_src:
                            return "transferSource";
                        case render::render_resource_usage::transfer_dst:
                            return "transferDestination";
                        case render::render_resource_usage::vertex_buffer:
                            return "vertexBuffer";
                        case render::render_resource_usage::index_buffer:
                            return "indexBuffer";
                        case render::render_resource_usage::uniform_buffer:
                            return "uniformBuffer";
                        case render::render_resource_usage::storage_buffer:
                            return "storageBuffer";
                        case render::render_resource_usage::indirect_buffer:
                            return "indirectBuffer";
                        case render::render_resource_usage::present:
                            return "present";
                        case render::render_resource_usage::unknown:
                            break;
                    }
                    return "unknown";
                };
                const auto queue_name = [](render::render_queue_type queue)
                {
                    switch (queue)
                    {
                        case render::render_queue_type::graphics:
                            return "graphics";
                        case render::render_queue_type::compute:
                            return "compute";
                        case render::render_queue_type::transfer:
                            return "transfer";
                    }
                    return "graphics";
                };
                std::string json = "{\"frameIndex\":" + std::to_string(profile.frame_index) +
                                   ",\"summary\":" + to_json_string(profile.summary) + ",\"passes\":[";
                for (std::size_t index = 0; index < profile.pass_timings.size(); ++index)
                {
                    if (index != 0) json += ',';
                    json += "{\"name\":" + to_json_string(profile.pass_timings[index].name) +
                            ",\"milliseconds\":" + std::to_string(profile.pass_timings[index].milliseconds) + '}';
                }
                json += "],\"timingsAvailable\":" + std::string(profile.pass_timings.empty() ? "false" : "true") +
                        ",\"graph\":{\"executedPasses\":[";
                for (std::size_t index = 0; index < profile.graph.passes.size(); ++index)
                {
                    if (index != 0) json += ',';
                    json += to_json_string(profile.graph.passes[index].name);
                }
                std::uint64_t transient_bytes{};
                std::uint32_t alias_count{};
                for (const auto& physical : profile.graph.physical_resources)
                    transient_bytes += physical.estimated_bytes;
                for (const auto& lifetime : profile.graph.lifetimes)
                    alias_count += lifetime.aliased ? 1u : 0u;
                json += "],\"culledPasses\":[";
                for (std::size_t index = 0; index < profile.graph.culled_passes.size(); ++index)
                {
                    if (index != 0) json += ',';
                    json += to_json_string(profile.graph.culled_passes[index].name);
                }
                json += "],\"resources\":[";
                bool first_resource = true;
                for (std::size_t index = 0; index < profile.graph.lifetimes.size(); ++index)
                {
                    const auto& lifetime = profile.graph.lifetimes[index];
                    if (lifetime.first_pass == render::render_graph_resource_handle::invalid_index) continue;
                    if (!first_resource) json += ',';
                    first_resource = false;
                    const auto* resource =
                        lifetime.handle.valid() && lifetime.handle.index < profile.graph.resources.size()
                            ? &profile.graph.resources[lifetime.handle.index]
                            : nullptr;
                    json += "{\"name\":" + to_json_string(resource ? resource->name : std::string{}) + ",\"format\":" +
                            to_json_string(resource ? render::render_format_name(resource->format) : "unknown") +
                            ",\"firstPass\":" + std::to_string(lifetime.first_pass) +
                            ",\"lastPass\":" + std::to_string(lifetime.last_pass) +
                            ",\"physicalResource\":" + std::to_string(lifetime.physical_resource) +
                            ",\"estimatedBytes\":" + std::to_string(lifetime.estimated_bytes) +
                            ",\"aliased\":" + std::string(lifetime.aliased ? "true" : "false") + '}';
                }
                json += "],\"transitions\":[";
                for (std::size_t index = 0; index < profile.graph.transitions.size(); ++index)
                {
                    if (index != 0) json += ',';
                    const auto& transition = profile.graph.transitions[index];
                    json += "{\"resource\":" + to_json_string(transition.resource) +
                            ",\"before\":" + to_json_string(usage_name(transition.before)) +
                            ",\"after\":" + to_json_string(usage_name(transition.after)) +
                            ",\"beforePass\":" + std::to_string(transition.before_pass) +
                            ",\"afterPass\":" + std::to_string(transition.after_pass) +
                            ",\"beforeQueue\":" + to_json_string(queue_name(transition.before_queue)) +
                            ",\"afterQueue\":" + to_json_string(queue_name(transition.after_queue)) + '}';
                }
                json += "],\"submissions\":[";
                for (std::size_t index = 0; index < profile.graph.submissions.size(); ++index)
                {
                    if (index != 0) json += ',';
                    const auto& submission = profile.graph.submissions[index];
                    json += "{\"queue\":" + to_json_string(queue_name(submission.queue)) +
                            ",\"passCount\":" + std::to_string(submission.passes.size()) +
                            ",\"waitCount\":" + std::to_string(submission.waits.size()) +
                            ",\"signalValue\":" + std::to_string(submission.signal_value) + '}';
                }
                json += "],\"histories\":[";
                for (std::size_t index = 0; index < profile.graph.history_rotations.size(); ++index)
                {
                    if (index != 0) json += ',';
                    const auto& history = profile.graph.history_rotations[index];
                    json += "{\"key\":" + to_json_string(history.persistent_key) +
                            ",\"length\":" + std::to_string(history.history_length) +
                            ",\"invalidated\":" + std::string(history.invalidated ? "true" : "false") + '}';
                }
                json +=
                    "],\"resourceCount\":" + std::to_string(profile.graph.resources.size()) +
                    ",\"physicalResourceCount\":" + std::to_string(profile.graph.physical_resources.size()) +
                    ",\"aliasCount\":" + std::to_string(alias_count) +
                    ",\"barrierCount\":" + std::to_string(profile.graph.transitions.size()) +
                    ",\"estimatedTransientBytes\":" + std::to_string(transient_bytes) + "},\"renderer\":{\"path\":" +
                    to_json_string(profile.configuration.path == render::render_path::forward_plus ? "forwardPlus"
                                                                                                   : "deferred") +
                    ",\"renderScale\":" + std::to_string(profile.configuration.render_scale) + ",\"qualityTier\":" +
                    to_json_string(profile.configuration.quality == render::render_quality_tier::low     ? "low"
                                   : profile.configuration.quality == render::render_quality_tier::ultra ? "ultra"
                                   : profile.configuration.quality == render::render_quality_tier::high  ? "high"
                                                                                                         : "standard") +
                    ",\"targetFrameMilliseconds\":" + std::to_string(profile.configuration.target_frame_time_ms) +
                    ",\"geometryErrorThreshold\":" + std::to_string(profile.configuration.geometry_error_threshold) +
                    ",\"shadowResolutionScale\":" + std::to_string(profile.configuration.shadow_resolution_scale) +
                    ",\"volumetricResolutionScale\":" +
                    std::to_string(profile.configuration.volumetric_resolution_scale) +
                    ",\"giTraceBudget\":" + std::to_string(profile.configuration.gi_trace_budget) +
                    ",\"reflectionRayBudget\":" + std::to_string(profile.configuration.reflection_ray_budget) +
                    ",\"lightingTraceScale\":" + std::to_string(profile.configuration.lighting_trace_scale) +
                    ",\"surfaceCacheUpdateBudget\":" +
                    std::to_string(profile.configuration.surface_cache_update_budget) +
                    ",\"radianceProbeUpdateBudget\":" +
                    std::to_string(profile.configuration.radiance_probe_update_budget) + ",\"fallbackReasons\":[";
                for (std::size_t index = 0; index < profile.configuration.fallback_reasons.size(); ++index)
                {
                    if (index != 0) json += ',';
                    json += to_json_string(profile.configuration.fallback_reasons[index]);
                }
                json +=
                    "]},\"environment\":{\"enabled\":" + std::string(profile.environment.enabled ? "true" : "false") +
                    ",\"skyVisible\":" + std::string(profile.environment.sky_visible ? "true" : "false") +
                    ",\"affectsLighting\":" + std::string(profile.environment.affects_lighting ? "true" : "false") +
                    ",\"source\":" + to_json_string(profile.environment.source) +
                    ",\"qualityPath\":" + to_json_string(profile.environment.quality_path) +
                    ",\"atmosphereLutState\":" + to_json_string(profile.environment.atmosphere_lut_state) +
                    ",\"lightingState\":" + to_json_string(profile.environment.environment_lighting_state) +
                    ",\"cloudShadowResolution\":" + std::to_string(profile.environment.cloud_shadow_resolution) +
                    ",\"fallback\":" + to_json_string(profile.environment.fallback_reason) +
                    "},\"shadows\":{\"cascades\":" + std::to_string(profile.shadows.directional_cascade_count) +
                    ",\"directionalResolution\":" + std::to_string(profile.shadows.directional_resolution) +
                    ",\"localAtlasResolution\":" + std::to_string(profile.shadows.local_atlas_resolution) +
                    ",\"localAllocations\":" + std::to_string(profile.shadows.local_allocation_count) +
                    ",\"localOccupiedTexels\":" + std::to_string(profile.shadows.local_occupied_texels) +
                    ",\"localEvictions\":" + std::to_string(profile.shadows.local_eviction_count) +
                    ",\"localResolutionReductions\":" + std::to_string(profile.shadows.local_resolution_reductions) +
                    ",\"shadowedPointLights\":" + std::to_string(profile.shadows.shadowed_point_lights) +
                    ",\"shadowedSpotLights\":" + std::to_string(profile.shadows.shadowed_spot_lights) +
                    ",\"staticCasters\":" + std::to_string(profile.shadows.static_caster_count) +
                    ",\"dynamicCasters\":" + std::to_string(profile.shadows.dynamic_caster_count) +
                    ",\"localCacheHits\":" + std::to_string(profile.shadows.local_cache_hits) +
                    ",\"localCacheMisses\":" + std::to_string(profile.shadows.local_cache_misses) +
                    ",\"staticCacheHit\":" + std::string(profile.shadows.static_cache_hit ? "true" : "false") +
                    ",\"screenSpaceShadows\":" + std::string(profile.shadows.screen_space_shadows ? "true" : "false") +
                    ",\"virtualShadowMaps\":" + std::string(profile.shadows.virtual_shadow_maps ? "true" : "false") +
                    ",\"virtualAddressSpaces\":" + std::to_string(profile.shadows.virtual_address_space_count) +
                    ",\"virtualPageCapacity\":" + std::to_string(profile.shadows.virtual_page_capacity) +
                    ",\"virtualResidentPages\":" + std::to_string(profile.shadows.virtual_resident_pages) +
                    ",\"virtualDirtyPages\":" + std::to_string(profile.shadows.virtual_dirty_pages) +
                    ",\"virtualRenderedPages\":" + std::to_string(profile.shadows.virtual_rendered_pages) +
                    ",\"virtualReusedPages\":" + std::to_string(profile.shadows.virtual_reused_pages) +
                    ",\"virtualEvictions\":" + std::to_string(profile.shadows.virtual_evictions) +
                    ",\"virtualParentFallbacks\":" + std::to_string(profile.shadows.virtual_parent_fallbacks) +
                    ",\"virtualFailedRequests\":" + std::to_string(profile.shadows.virtual_failed_requests) +
                    ",\"virtualMemoryBytes\":" + std::to_string(profile.shadows.virtual_memory_bytes) +
                    ",\"fallback\":" + to_json_string(profile.shadows.fallback_reason) +
                    "},\"gpuScene\":{\"enabled\":" + std::string(profile.gpu_scene.enabled ? "true" : "false") +
                    ",\"hzbOcclusion\":" + std::string(profile.gpu_scene.hzb_occlusion ? "true" : "false") +
                    ",\"historyValid\":" + std::string(profile.gpu_scene.history_valid ? "true" : "false") +
                    ",\"capacity\":" + std::to_string(profile.gpu_scene.capacity) +
                    ",\"activeInstances\":" + std::to_string(profile.gpu_scene.active_instances) +
                    ",\"uploadedInstances\":" + std::to_string(profile.gpu_scene.uploaded_instances) +
                    ",\"uploadedRanges\":" + std::to_string(profile.gpu_scene.uploaded_ranges) +
                    ",\"uploadedBytes\":" + std::to_string(profile.gpu_scene.uploaded_bytes) +
                    ",\"geometryTableEntries\":" + std::to_string(profile.gpu_scene.geometry_table_entries) +
                    ",\"materialTableEntries\":" + std::to_string(profile.gpu_scene.material_table_entries) +
                    ",\"textureTableEntries\":" + std::to_string(profile.gpu_scene.texture_table_entries) +
                    ",\"samplerTableEntries\":" + std::to_string(profile.gpu_scene.sampler_table_entries) +
                    ",\"skinPaletteTableEntries\":" +
                    std::to_string(profile.gpu_scene.skin_palette_table_entries) +
                    ",\"sharedVertexHeapBytes\":" + std::to_string(profile.gpu_scene.shared_vertex_heap_bytes) +
                    ",\"sharedIndexHeapBytes\":" + std::to_string(profile.gpu_scene.shared_index_heap_bytes) +
                    ",\"candidateInstances\":" + std::to_string(profile.gpu_scene.candidate_instances) +
                    ",\"visibleInstances\":" + std::to_string(profile.gpu_scene.visible_instances) +
                    ",\"frustumRejected\":" + std::to_string(profile.gpu_scene.frustum_rejected) +
                    ",\"distanceRejected\":" + std::to_string(profile.gpu_scene.distance_rejected) +
                    ",\"occlusionRejected\":" + std::to_string(profile.gpu_scene.occlusion_rejected) +
                    ",\"activePipelineBins\":" + std::to_string(profile.gpu_scene.active_pipeline_bins) +
                    ",\"indirectCommands\":" + std::to_string(profile.gpu_scene.indirect_commands) +
                    ",\"overflowRecords\":" + std::to_string(profile.gpu_scene.overflow_records) +
                    ",\"cpuSubmissions\":" + std::to_string(profile.gpu_scene.cpu_submissions) +
                    ",\"fallback\":" + to_json_string(profile.gpu_scene.fallback_reason) +
                    "},\"textureStreaming\":{\"gpuBudgetBytes\":" +
                    std::to_string(profile.texture_streaming.gpu_budget_bytes) +
                    ",\"gpuResidentBytes\":" + std::to_string(profile.texture_streaming.gpu_resident_bytes) +
                    ",\"cpuBudgetBytes\":" + std::to_string(profile.texture_streaming.cpu_budget_bytes) +
                    ",\"cpuCachedBytes\":" + std::to_string(profile.texture_streaming.cpu_cached_bytes) +
                    ",\"uploadBudgetBytes\":" + std::to_string(profile.texture_streaming.upload_budget_bytes) +
                    ",\"uploadedBytes\":" + std::to_string(profile.texture_streaming.uploaded_bytes) +
                    ",\"ioReadBytes\":" + std::to_string(profile.texture_streaming.io_read_bytes) +
                    ",\"ioFailedBytes\":" + std::to_string(profile.texture_streaming.io_failed_bytes) +
                    ",\"ioCompletedReads\":" + std::to_string(profile.texture_streaming.io_completed_reads) +
                    ",\"ioFailedReads\":" + std::to_string(profile.texture_streaming.io_failed_reads) +
                    ",\"ioInFlightReads\":" + std::to_string(profile.texture_streaming.io_in_flight_reads) +
                    ",\"ioLatencyMilliseconds\":" +
                    std::to_string(profile.texture_streaming.io_latency_milliseconds) +
                    ",\"uploadLatencyMilliseconds\":" +
                    std::to_string(profile.texture_streaming.upload_latency_milliseconds) +
                    ",\"streamedTextures\":" + std::to_string(profile.texture_streaming.streamed_textures) +
                    ",\"virtualTextures\":" + std::to_string(profile.texture_streaming.virtual_textures) +
                    ",\"residentMips\":" + std::to_string(profile.texture_streaming.resident_mips) +
                    ",\"residentPages\":" + std::to_string(profile.texture_streaming.resident_pages) +
                    ",\"requestedMips\":" + std::to_string(profile.texture_streaming.requested_mips) +
                    ",\"requestedPages\":" + std::to_string(profile.texture_streaming.requested_pages) +
                    ",\"failedMips\":" + std::to_string(profile.texture_streaming.failed_mips) +
                    ",\"failedPages\":" + std::to_string(profile.texture_streaming.failed_pages) +
                    ",\"evictions\":" + std::to_string(profile.texture_streaming.evictions) +
                    ",\"feedbackOverflow\":" + std::to_string(profile.texture_streaming.feedback_overflow) +
                    ",\"parentFallbacks\":" + std::to_string(profile.texture_streaming.parent_fallbacks) +
                    ",\"cacheHitRate\":" + std::to_string(profile.texture_streaming.cache_hit_rate) +
                    ",\"overBudget\":" + std::string(profile.texture_streaming.over_budget ? "true" : "false") +
                    ",\"fallback\":" + to_json_string(profile.texture_streaming.fallback_reason) +
                    ",\"resources\":[";
                for (std::size_t index = 0; index < texture_resources.size(); ++index)
                {
                    if (index != 0) json += ',';
                    const auto& resource = texture_resources[index];
                    const auto mode_name = [](render::texture_streaming_mode mode)
                    {
                        return mode == render::texture_streaming_mode::virtual_tiles       ? "virtual_tiles"
                               : mode == render::texture_streaming_mode::streamed_mips ? "streamed_mips"
                                                                                      : "resident";
                    };
                    json += "{\"resourceIndex\":" + std::to_string(resource.resource.index) +
                            ",\"resourceGeneration\":" + std::to_string(resource.resource.generation) +
                            ",\"contentGeneration\":" + std::to_string(resource.content_generation) +
                            ",\"authoredMode\":" + to_json_string(mode_name(resource.authored_mode)) +
                            ",\"resolvedMode\":" + to_json_string(mode_name(resource.resolved_mode)) +
                            ",\"desiredMip\":" + std::to_string(resource.requested_mip) +
                            ",\"residentMip\":" + std::to_string(resource.resident_first_mip) +
                            ",\"tailFirstMip\":" + std::to_string(resource.tail_first_mip) +
                            ",\"residentBytes\":" + std::to_string(resource.resident_bytes) +
                            ",\"forcedMip\":" +
                            (resource.forced_mip ? std::to_string(*resource.forced_mip) : "null") + '}';
                }
                json += "]},\"virtualGeometry\":{\"enabled\":" +
                    std::string(profile.virtual_geometry.enabled ? "true" : "false") + ",\"rasterPath\":" +
                    to_json_string(profile.virtual_geometry.raster_path == render::virtual_geometry_raster_path::compute
                                       ? "compute"
                                   : profile.virtual_geometry.raster_path ==
                                           render::virtual_geometry_raster_path::mesh_shader
                                       ? "meshShader"
                                       : "unavailable") +
                    ",\"visibleClusters\":" + std::to_string(profile.virtual_geometry.visible_clusters) +
                    ",\"visibleTriangles\":" + std::to_string(profile.virtual_geometry.visible_triangles) +
                    ",\"frustumRejected\":" + std::to_string(profile.virtual_geometry.frustum_rejected) +
                    ",\"coneRejected\":" + std::to_string(profile.virtual_geometry.cone_rejected) +
                    ",\"hzbRejected\":" + std::to_string(profile.virtual_geometry.hzb_rejected) +
                    ",\"projectedSizeRejected\":" + std::to_string(profile.virtual_geometry.projected_size_rejected) +
                    ",\"requestedPages\":" + std::to_string(profile.virtual_geometry.requested_pages) +
                    ",\"loadedPages\":" + std::to_string(profile.virtual_geometry.loaded_pages) +
                    ",\"failedPages\":" + std::to_string(profile.virtual_geometry.failed_pages) +
                    ",\"parentFallbacks\":" + std::to_string(profile.virtual_geometry.parent_fallbacks) +
                    ",\"residentBytes\":" + std::to_string(profile.virtual_geometry.resident_bytes) +
                    ",\"residencyBudgetBytes\":" + std::to_string(profile.virtual_geometry.residency_budget_bytes) +
                    ",\"fallback\":" + to_json_string(profile.virtual_geometry.fallback_reason) +
                    "},\"indirectLighting\":{\"enabled\":" +
                    std::string(profile.indirect_lighting.enabled ? "true" : "false") + ",\"tracePath\":" +
                    to_json_string(profile.indirect_lighting.trace_path == render::lighting_trace_path::hybrid_hardware
                                       ? "hybridHardware"
                                   : profile.indirect_lighting.trace_path ==
                                           render::lighting_trace_path::software_distance_field
                                       ? "softwareDistanceField"
                                   : profile.indirect_lighting.trace_path == render::lighting_trace_path::screen_space
                                       ? "screenSpace"
                                       : "bakedProbe") +
                    ",\"traceScale\":" + std::to_string(profile.indirect_lighting.trace_scale) +
                    ",\"giRays\":" + std::to_string(profile.indirect_lighting.gi_rays) +
                    ",\"reflectionRays\":" + std::to_string(profile.indirect_lighting.reflection_rays) +
                    ",\"surfaceCards\":" + std::to_string(profile.indirect_lighting.surface_cards) +
                    ",\"residentSurfacePages\":" + std::to_string(profile.indirect_lighting.resident_surface_pages) +
                    ",\"residentDistanceFieldPages\":" +
                    std::to_string(profile.indirect_lighting.resident_distance_field_pages) +
                    ",\"dirtyRegions\":" + std::to_string(profile.indirect_lighting.dirty_regions) +
                    ",\"surfaceUpdates\":" + std::to_string(profile.indirect_lighting.surface_updates) +
                    ",\"radianceProbeUpdates\":" + std::to_string(profile.indirect_lighting.radiance_probe_updates) +
                    ",\"residentBytes\":" + std::to_string(profile.indirect_lighting.resident_bytes) +
                    ",\"budgetBytes\":" + std::to_string(profile.indirect_lighting.budget_bytes) +
                    ",\"screenHitRate\":" + std::to_string(profile.indirect_lighting.screen_hit_rate) +
                    ",\"softwareHitRate\":" + std::to_string(profile.indirect_lighting.software_hit_rate) +
                    ",\"hardwareHitRate\":" + std::to_string(profile.indirect_lighting.hardware_hit_rate) +
                    ",\"fallback\":" + to_json_string(profile.indirect_lighting.fallback_reason) +
                    "},\"temporal\":{\"enabled\":" + std::string(profile.temporal.enabled ? "true" : "false") +
                    ",\"upscaling\":" + std::string(profile.temporal.upscaling ? "true" : "false") +
                    ",\"historyValid\":" + std::string(profile.temporal.history_valid ? "true" : "false") +
                    ",\"cameraCut\":" + std::string(profile.temporal.camera_cut ? "true" : "false") +
                    ",\"jitterX\":" + std::to_string(profile.temporal.jitter[0]) +
                    ",\"jitterY\":" + std::to_string(profile.temporal.jitter[1]) +
                    ",\"resetReason\":" + to_json_string(profile.temporal.reset_reason) + "}}";
                return {.request_id = request_id, .succeeded = true, .payload_json = std::move(json)};
            }
            else if constexpr (std::is_same_v<query_type, host_viewport_capture_query>)
            {
                auto capture = state_->renderer->last_frame_capture();
                if (!capture.available || capture.capture_id != payload.capture_id)
                {
                    return {.request_id = request_id,
                            .succeeded = true,
                            .payload_json =
                                "{\"captureId\":" + std::to_string(payload.capture_id) + ",\"pending\":true}"};
                }
                const auto channel_name = [](render::render_capture_channel channel)
                {
                    switch (channel)
                    {
                        case render::render_capture_channel::output_color:
                            return "color";
                        case render::render_capture_channel::scene_color:
                            return "sceneColor";
                        case render::render_capture_channel::linear_depth:
                            return "depth";
                        case render::render_capture_channel::object_id:
                            return "objectId";
                        case render::render_capture_channel::world_normal:
                            return "normals";
                        case render::render_capture_channel::base_color:
                            return "baseColor";
                        case render::render_capture_channel::material_properties:
                            return "materialProperties";
                        case render::render_capture_channel::emissive:
                            return "emissive";
                        case render::render_capture_channel::indirect_diffuse:
                            return "indirectDiffuse";
                        case render::render_capture_channel::reflections:
                            return "reflections";
                        case render::render_capture_channel::trace_source:
                            return "traceSource";
                        case render::render_capture_channel::mesh_distance_field:
                            return "distanceField";
                        case render::render_capture_channel::temporal_confidence:
                            return "temporalConfidence";
                    }
                    return "unknown";
                };
                const auto format_name = [](render::render_capture_format format)
                {
                    switch (format)
                    {
                        case render::render_capture_format::rgba8_unorm:
                            return "rgba8";
                        case render::render_capture_format::bgra8_unorm:
                            return "bgra8";
                        case render::render_capture_format::rgba16_float:
                            return "rgba16f";
                        case render::render_capture_format::r32_float:
                            return "r32f";
                        case render::render_capture_format::r32_uint:
                            return "r32ui";
                    }
                    return "unknown";
                };
                const auto matrix_json = [](const math::matrix4f& matrix)
                {
                    std::string json = "[";
                    for (std::size_t index = 0; index < 16; ++index)
                    {
                        if (index != 0) json += ',';
                        json += std::to_string(matrix.data()[index]);
                    }
                    json += ']';
                    return json;
                };
                const auto& capture_camera = capture.camera;
                const bool orthographic = std::abs(capture_camera.projection(3, 3) - 1.0f) <= 0.000001f;
                if (capture_camera.near_plane > 0.0f && capture_camera.far_plane > capture_camera.near_plane)
                {
                    for (auto& image : capture.images)
                    {
                        if (image.channel != render::render_capture_channel::linear_depth ||
                            image.format != render::render_capture_format::r32_float)
                            continue;
                        const float near_plane = capture_camera.near_plane;
                        const float far_plane = capture_camera.far_plane;
                        for (std::size_t offset = 0; offset + sizeof(float) <= image.data.size();
                             offset += sizeof(float))
                        {
                            float device_depth{};
                            std::memcpy(&device_depth, image.data.data() + offset, sizeof(device_depth));
                            const float linear_depth =
                                orthographic
                                    ? near_plane + device_depth * (far_plane - near_plane)
                                    : (near_plane * far_plane) /
                                          std::max(far_plane - device_depth * (far_plane - near_plane), 0.000001f);
                            std::memcpy(image.data.data() + offset, &linear_depth, sizeof(linear_depth));
                        }
                    }
                }
                std::string json =
                    "{\"captureId\":" + std::to_string(capture.capture_id) +
                    ",\"pending\":false,\"succeeded\":" + std::string(capture.succeeded ? "true" : "false") +
                    ",\"frameIndex\":" + std::to_string(capture.frame_index) + ",\"camera\":{\"position\":[" +
                    std::to_string(capture_camera.position[0]) + ',' + std::to_string(capture_camera.position[1]) +
                    ',' + std::to_string(capture_camera.position[2]) + "],\"forward\":[" +
                    std::to_string(capture_camera.forward[0]) + ',' + std::to_string(capture_camera.forward[1]) + ',' +
                    std::to_string(capture_camera.forward[2]) + "],\"up\":[" + std::to_string(capture_camera.up[0]) +
                    ',' + std::to_string(capture_camera.up[1]) + ',' + std::to_string(capture_camera.up[2]) +
                    "],\"projectionType\":" + to_json_string(orthographic ? "orthographic" : "perspective") +
                    ",\"nearPlane\":" + std::to_string(capture_camera.near_plane) +
                    ",\"farPlane\":" + std::to_string(capture_camera.far_plane) + ",\"renderExtent\":[" +
                    std::to_string(capture_camera.render_width) + ',' + std::to_string(capture_camera.render_height) +
                    "],\"outputExtent\":[" + std::to_string(capture_camera.output_width) + ',' +
                    std::to_string(capture_camera.output_height) +
                    "],\"viewProjection\":" + matrix_json(capture_camera.view_projection) +
                    ",\"inverseViewProjection\":" + matrix_json(capture_camera.inverse_view_projection) +
                    ",\"projection\":" + matrix_json(capture_camera.projection) + "},\"images\":[";
                for (std::size_t index = 0; index < capture.images.size(); ++index)
                {
                    if (index != 0) json += ',';
                    const auto& image = capture.images[index];
                    json += "{\"channel\":" + to_json_string(channel_name(image.channel)) +
                            ",\"format\":" + to_json_string(format_name(image.format)) +
                            ",\"width\":" + std::to_string(image.width) +
                            ",\"height\":" + std::to_string(image.height) +
                            ",\"data\":" + to_json_string(base64_encode(image.data)) + '}';
                }
                json += "],\"objects\":[";
                bool first_object = true;
                for (const auto& object : capture.objects)
                {
                    const ecs::entity entity{object.object.index, object.object.generation};
                    if (!state_->scene.scene.alive(entity)) continue;
                    if (!first_object) json += ',';
                    first_object = false;
                    json += "{\"id\":" + std::to_string(object.encoded_id) +
                            ",\"guid\":" + to_json_string(ecs::to_string(entity_guid_of(state_->scene, entity))) + '}';
                }
                json += "],\"diagnostics\":[";
                for (std::size_t index = 0; index < capture.diagnostics.size(); ++index)
                {
                    if (index != 0) json += ',';
                    json += to_json_string(capture.diagnostics[index]);
                }
                json += "]}";
                return {.request_id = request_id, .succeeded = true, .payload_json = std::move(json)};
            }
            else if constexpr (std::is_same_v<query_type, host_project_assets_query>)
            {
                return {
                    .request_id = request_id, .succeeded = true, .payload_json = to_json(project_assets_snapshot())};
            }
            else if constexpr (std::is_same_v<query_type, host_texture_settings_query>)
            {
                if (!state_->asset_registry)
                    return {
                        .request_id = request_id, .succeeded = false, .error = "No project asset registry is active"};
                const auto guid = assets::parse_asset_guid(payload.guid);
                if (!guid) return {.request_id = request_id, .succeeded = false, .error = "Invalid texture GUID"};
                const auto asset = state_->asset_registry->find(*guid);
                if (!asset || asset->type != assets::asset_types::texture_2d)
                    return {.request_id = request_id, .succeeded = false, .error = "Texture asset was not found"};
                const auto metadata =
                    assets::load_asset_metadata(assets::metadata_path_for(state_->project.root / asset->source_path));
                if (!metadata) return {.request_id = request_id, .succeeded = false, .error = metadata.error().message};
                const auto settings = render::tools::parse_texture_import_settings(metadata.value().canonical_settings,
                                                                                   metadata.value().settings_version);
                if (!settings) return {.request_id = request_id, .succeeded = false, .error = settings.error()};
                const auto& value = settings.value();
                const auto json =
                    "{\"settingsVersion\":" + std::to_string(metadata.value().settings_version) +
                    ",\"preset\":" + to_json_string(render::tools::texture_import_preset_name(value.preset)) +
                    ",\"semantic\":" + to_json_string(render::tools::texture_semantic_name(value.semantic)) +
                    ",\"colorSpace\":" + to_json_string(render::tools::texture_color_space_name(value.color_space)) +
                    ",\"streamingMode\":" +
                    to_json_string(render::tools::texture_streaming_mode_name(value.streaming_mode)) +
                    ",\"compression\":" +
                    to_json_string(render::tools::texture_compression_policy_name(value.compression)) +
                    ",\"powerOfTwo\":" +
                    to_json_string(render::tools::texture_power_of_two_policy_name(value.power_of_two)) +
                    ",\"minFilter\":" + to_json_string(render::tools::texture_filter_mode_name(value.min_filter)) +
                    ",\"magFilter\":" + to_json_string(render::tools::texture_filter_mode_name(value.mag_filter)) +
                    ",\"mipFilter\":" + to_json_string(render::tools::texture_mip_filter_mode_name(value.mip_filter)) +
                    ",\"wrapU\":" + to_json_string(render::tools::texture_address_mode_name(value.wrap_u)) +
                    ",\"wrapV\":" + to_json_string(render::tools::texture_address_mode_name(value.wrap_v)) +
                    ",\"mipGenerationFilter\":" +
                    to_json_string(render::tools::texture_mip_generation_filter_name(value.mip_generation_filter)) +
                    ",\"maxSize\":" + std::to_string(value.max_size) +
                    ",\"anisotropy\":" + std::to_string(value.anisotropy) +
                    ",\"lodBias\":" + std::to_string(value.lod_bias) +
                    ",\"minimumLod\":" + std::to_string(value.minimum_lod) +
                    ",\"maximumLod\":" + std::to_string(value.maximum_lod) +
                    ",\"alphaCoverageThreshold\":" + std::to_string(value.alpha_coverage_threshold) +
                    ",\"mipSharpen\":" + std::to_string(value.mip_sharpen) +
                    ",\"ditherMips\":" + std::string(value.dither_mips ? "true" : "false") +
                    ",\"debandMips\":" + std::string(value.deband_mips ? "true" : "false") +
                    ",\"debandStrength\":" + std::to_string(value.deband_strength) +
                    ",\"curvesEnabled\":" + std::string(value.curves_enabled ? "true" : "false") +
                    ",\"curveMaster\":" + render::tools::serialize_texture_curve(value.curve_master) +
                    ",\"curveR\":" + render::tools::serialize_texture_curve(value.curve_r) +
                    ",\"curveG\":" + render::tools::serialize_texture_curve(value.curve_g) +
                    ",\"curveB\":" + render::tools::serialize_texture_curve(value.curve_b) +
                    ",\"curveA\":" + render::tools::serialize_texture_curve(value.curve_a) +
                    ",\"brightness\":" + std::to_string(value.brightness) +
                    ",\"gamma\":" + std::to_string(value.gamma) + ",\"contrast\":" + std::to_string(value.contrast) +
                    ",\"saturation\":" + std::to_string(value.saturation) +
                    ",\"vibrance\":" + std::to_string(value.vibrance) + ",\"tintR\":" + std::to_string(value.tint_r) +
                    ",\"tintG\":" + std::to_string(value.tint_g) + ",\"tintB\":" + std::to_string(value.tint_b) +
                    ",\"inputBlack\":" + std::to_string(value.input_black) +
                    ",\"inputWhite\":" + std::to_string(value.input_white) +
                    ",\"outputBlack\":" + std::to_string(value.output_black) +
                    ",\"outputWhite\":" + std::to_string(value.output_white) +
                    ",\"channelR\":" + to_json_string(render::tools::texture_channel_source_name(value.channel_r)) +
                    ",\"channelG\":" + to_json_string(render::tools::texture_channel_source_name(value.channel_g)) +
                    ",\"channelB\":" + to_json_string(render::tools::texture_channel_source_name(value.channel_b)) +
                    ",\"channelA\":" + to_json_string(render::tools::texture_channel_source_name(value.channel_a)) +
                    ",\"invertR\":" + std::string(value.invert_r ? "true" : "false") +
                    ",\"invertG\":" + std::string(value.invert_g ? "true" : "false") +
                    ",\"invertB\":" + std::string(value.invert_b ? "true" : "false") +
                    ",\"invertA\":" + std::string(value.invert_a ? "true" : "false") +
                    ",\"generateMips\":" + std::string(value.generate_mips ? "true" : "false") +
                    ",\"preserveAlphaCoverage\":" + std::string(value.preserve_alpha_coverage ? "true" : "false") + '}';
                return {.request_id = request_id, .succeeded = true, .payload_json = json};
            }
            else if constexpr (std::is_same_v<query_type, host_asset_thumbnail_query>)
            {
                const auto thumbnail = asset_thumbnail(payload.path, payload.max_size);
                if (!thumbnail)
                    return {.request_id = request_id,
                            .succeeded = false,
                            .error = "Texture thumbnail could not be generated"};
                return {.request_id = request_id, .succeeded = true, .payload_json = to_json(*thumbnail)};
            }
            else if constexpr (std::is_same_v<query_type, host_viewport_state_query>)
            {
                if (payload.viewport_id != state_->active_viewport_id)
                    return {.request_id = request_id, .succeeded = false, .error = "Viewport is not attached"};
                std::string camera_json = "null";
                if (const auto* transform =
                        state_->scene.scene.try_get<scene::transform_component>(state_->scene.camera_entity))
                {
                    if (const auto* camera =
                            state_->scene.scene.try_get<scene::camera_component>(state_->scene.camera_entity))
                    {
                        camera_json =
                            "{\"transform\":" + to_json(to_host_transform(*transform)) +
                            ",\"settings\":" + to_json(to_host_camera(*camera)) + ",\"focus\":" +
                            to_json(host_transform{.position = {state_->camera_controller.focus_point()[0],
                                                                state_->camera_controller.focus_point()[1],
                                                                state_->camera_controller.focus_point()[2]}}) +
                            ",\"distance\":" + std::to_string(state_->camera_controller.distance()) + '}';
                    }
                }
                return {
                    .request_id = request_id,
                    .succeeded = true,
                    .payload_json =
                        "{\"viewportId\":" + to_json_string(state_->active_viewport_id) +
                        ",\"width\":" + std::to_string(state_->viewport_options.width) +
                        ",\"height\":" + std::to_string(state_->viewport_options.height) +
                        ",\"fps\":" + std::to_string(state_->viewport_fps) +
                        ",\"frameTimeMs\":" + std::to_string(state_->viewport_frame_ms) +
                        ",\"drawCalls\":" + std::to_string(state_->viewport_draw_calls) +
                        ",\"frameIndex\":" + std::to_string(state_->viewport_frame_index) +
                        ",\"submitted\":" + std::string(state_->viewport_submitted ? "true" : "false") +
                        ",\"renderOptions\":{\"renderMode\":" +
                        to_json_string(to_string(state_->viewport_options.render_mode)) +
                        ",\"visualization\":" + to_json_string(to_string(state_->viewport_options.visualization)) +
                        ",\"overlay\":" + to_json_string(to_string(state_->viewport_options.overlay)) +
                        ",\"shadows\":" + std::string(state_->viewport_options.shadows ? "true" : "false") +
                        ",\"grid\":" + std::string(state_->viewport_options.grid ? "true" : "false") +
                        ",\"realtime\":" + std::string(state_->viewport_options.realtime ? "true" : "false") +
                        ",\"cameraSpeed\":" + std::to_string(state_->viewport_options.camera_speed) +
                        ",\"antiAliasing\":" +
                        to_json_string(
                            state_->viewport_options.anti_aliasing == host_camera_anti_aliasing::disabled ? "disabled"
                            : state_->viewport_options.anti_aliasing == host_camera_anti_aliasing::fxaa   ? "fxaa"
                            : state_->viewport_options.anti_aliasing == host_camera_anti_aliasing::taa    ? "taa"
                            : state_->viewport_options.anti_aliasing == host_camera_anti_aliasing::taau   ? "taau"
                                                                                                          : "inherit") +
                        ",\"environment\":{\"sky\":" +
                        std::string(state_->viewport_options.environment.sky ? "true" : "false") +
                        ",\"fog\":" + std::string(state_->viewport_options.environment.fog ? "true" : "false") +
                        ",\"terrain\":" + std::string(state_->viewport_options.environment.terrain ? "true" : "false") +
                        ",\"water\":" + std::string(state_->viewport_options.environment.water ? "true" : "false") +
                        ",\"vegetation\":" +
                        std::string(state_->viewport_options.environment.vegetation ? "true" : "false") +
                        ",\"decals\":" + std::string(state_->viewport_options.environment.decals ? "true" : "false") +
                        "}}" + ",\"camera\":" + camera_json + '}'};
            }
            else if constexpr (std::is_same_v<query_type, host_world_environment_query>)
            {
                const auto snapshot = world_environment_snapshot(payload.entity);
                if (!snapshot)
                    return {.request_id = request_id, .succeeded = false, .error = "World environment is missing"};
                return {.request_id = request_id, .succeeded = true, .payload_json = to_json(*snapshot)};
            }
            else if constexpr (std::is_same_v<query_type, host_history_state_query>)
            {
                const auto history = state_->history.snapshot();
                return {.request_id = request_id,
                        .succeeded = true,
                        .payload_json =
                            "{\"canUndo\":" + std::string(history.can_undo ? "true" : "false") +
                            ",\"canRedo\":" + std::string(history.can_redo ? "true" : "false") +
                            ",\"dirty\":" + std::string(history.dirty ? "true" : "false") +
                            ",\"transactionActive\":" + std::string(history.transaction_active ? "true" : "false") +
                            ",\"undoLabel\":" + to_json_string(history.undo_label) +
                            ",\"redoLabel\":" + to_json_string(history.redo_label) +
                            ",\"revision\":" + std::to_string(history.revision) + '}'};
            }
            else if constexpr (std::is_same_v<query_type, host_runtime_state_query>)
            {
                return {.request_id = request_id, .succeeded = true, .payload_json = to_json(runtime_snapshot())};
            }
            else if constexpr (std::is_same_v<query_type, host_terrain_tool_state_query>)
            {
                return {.request_id = request_id, .succeeded = true, .payload_json = to_json(terrain_tool_snapshot())};
            }

            else
                return {.request_id = request_id, .succeeded = false, .error = "Unsupported host query"};
        },
        query.payload);
    response.scene_revision = state_->scene_revision;
    response.world_epoch = state_->world_epoch;
    response.frame_revision = state_->viewport_frame_index;
    return response;
}

host_runtime_snapshot arc_host::runtime_snapshot() const
{
    host_runtime_snapshot snapshot;
    snapshot.state = state_->preview_stopped        ? host_runtime_state::stopped
                     : state_->simulation.paused()  ? host_runtime_state::paused
                     : state_->simulation.running() ? host_runtime_state::running
                                                    : host_runtime_state::stopped;
    for (const framework::runtime_world_id id : state_->simulation.worlds().ordered_worlds())
    {
        const framework::runtime_world* world = state_->simulation.worlds().find(id);
        if (world && world->state() == framework::runtime_world_state::faulted)
        {
            snapshot.state = host_runtime_state::faulted;
            break;
        }
    }
    snapshot.tick_id = state_->simulation.current_tick().id.value;
    snapshot.revision = state_->runtime_revision;
    snapshot.discarded_ticks = state_->simulation.discarded_ticks();
    snapshot.time_scale = state_->simulation.time_scale();
    snapshot.world_count = static_cast<std::uint32_t>(state_->simulation.worlds().size());
    return snapshot;
}

host_terrain_tool_snapshot arc_host::terrain_tool_snapshot() const
{
    host_terrain_tool_snapshot snapshot;
    snapshot.entity = to_host_entity(state_->scene.selected_entity);
    snapshot.active = state_->viewport_tool.tool == host_viewport_tool::terrain &&
                      state_->scene.scene.has<scene::terrain_component>(state_->scene.selected_entity);
    snapshot.hover_visible = snapshot.active && state_->terrain_brush_local_position.has_value();
    snapshot.tool = state_->terrain_brush.tool == scene::terrain_brush_tool::smooth ? host_terrain_brush_tool::smooth
                    : state_->terrain_brush.tool == scene::terrain_brush_tool::flatten
                        ? host_terrain_brush_tool::flatten
                    : state_->terrain_brush.tool == scene::terrain_brush_tool::paint ? host_terrain_brush_tool::paint
                                                                                     : host_terrain_brush_tool::sculpt;
    snapshot.radius = state_->terrain_brush.radius;
    snapshot.strength = state_->terrain_brush.strength;
    snapshot.falloff = state_->terrain_brush.falloff;
    snapshot.active_layer = state_->terrain_brush.active_layer;
    return snapshot;
}

host_scene_snapshot arc_host::scene_snapshot() const
{
    host_scene_snapshot snapshot;
    const auto history = state_->history.snapshot();
    snapshot.scene_guid = ecs::to_string(state_->scene.scene_guid);
    snapshot.scene_name = state_->scene.scene_name;
    snapshot.active_scene_path = state_->scene.active_scene_path.generic_string();
    snapshot.scene_revision = state_->scene_revision;
    snapshot.world_epoch = state_->world_epoch;
    snapshot.frame_revision = state_->viewport_frame_index;
    snapshot.dirty = history.dirty;
    snapshot.can_undo = history.can_undo;
    snapshot.can_redo = history.can_redo;
    snapshot.undo_label = history.undo_label;
    snapshot.redo_label = history.redo_label;
    const auto root_entities = scene::roots(state_->scene.scene);
    std::uint32_t maximum_root_index{};
    for (const auto root : root_entities)
        maximum_root_index = std::max(maximum_root_index, root.index);
    std::vector<std::uint32_t> root_orders(static_cast<std::size_t>(maximum_root_index) + 1u, 0u);
    std::uint32_t root_order{};
    for (const auto root : root_entities)
        if (root != state_->scene.camera_entity) root_orders[root.index] = root_order++;
    const auto add = [&](ecs::entity entity)
    {
        if (!entity.valid() || !state_->scene.scene.alive(entity)) return;
        host_entity_kind kind = host_entity_kind::unknown;
        if (state_->scene.scene.has<scene::world_environment_component>(entity) ||
            state_->scene.scene.has<scene::terrain_component>(entity) ||
            state_->scene.scene.has<scene::water_component>(entity) ||
            state_->scene.scene.has<scene::vegetation_component>(entity) ||
            state_->scene.scene.has<scene::decal_component>(entity))
            kind = host_entity_kind::environment;
        else if (state_->scene.scene.has<scene::directional_light_component>(entity) ||
                 state_->scene.scene.has<scene::point_light_component>(entity) ||
                 state_->scene.scene.has<scene::spot_light_component>(entity) ||
                 state_->scene.scene.has<scene::area_light_component>(entity))
            kind = host_entity_kind::light;
        else if (state_->scene.scene.has<scene::camera_component>(entity))
            kind = host_entity_kind::camera;
        else if (state_->scene.scene.has<scene::mesh_renderer_component>(entity))
            kind = host_entity_kind::mesh;
        std::string parent_guid;
        std::uint32_t sibling_order{};
        if (const auto* hierarchy = state_->scene.scene.try_get<scene::hierarchy_component>(entity))
        {
            if (state_->scene.scene.alive(hierarchy->parent))
            {
                parent_guid = ecs::to_string(entity_guid_of(state_->scene, hierarchy->parent));
                auto sibling = hierarchy->previous_sibling;
                std::unordered_set<ecs::entity, ecs::entity_hash> visited_siblings;
                while (state_->scene.scene.alive(sibling) && visited_siblings.insert(sibling).second)
                {
                    ++sibling_order;
                    sibling = state_->scene.scene.get<scene::hierarchy_component>(sibling).previous_sibling;
                }
            }
            else if (entity.index < root_orders.size())
            {
                sibling_order = root_orders[entity.index];
            }
        }
        snapshot.entities.push_back(
            {.entity = to_host_entity(entity),
             .guid = ecs::to_string(entity_guid_of(state_->scene, entity)),
             .parent_guid = std::move(parent_guid),
             .sibling_order = sibling_order,
             .name = entity_name(state_->scene, entity, "Entity"),
             .kind = kind,
             .document_guid = snapshot.scene_guid,
             .layer =
                 [&]
             {
                 const auto* layer = state_->scene.scene.try_get<scene::render_layer_component>(entity);
                 return layer && layer->mask == host_environment_render_layer ? std::string{"Environment"}
                                                                              : std::string{"Default"};
             }(),
             .active = entity_active(state_->scene, entity),
             .visible = entity_active(state_->scene, entity),
             .transform = [&]() -> std::optional<host_transform>
             {
                 if (const auto* transform = state_->scene.scene.try_get<scene::transform_component>(entity))
                     return to_host_transform(*transform);
                 return std::nullopt;
             }(),
             .prefab_override_count =
                 [&]
             {
                 const auto* prefab = state_->scene.scene.try_get<ecs::prefab_instance_component>(entity);
                 return prefab ? prefab->overrides.size() : std::size_t{};
             }(),
             .selected =
                 [&]
             {
                 const auto* selection = state_->scene.scene.try_get<scene::selection_component>(entity);
                 return selection && selection->selected;
             }()});
    };

    for (const auto entity : state_->scene.scene.entities())
        if (entity != state_->scene.camera_entity) add(entity);
    snapshot.total_entity_count = snapshot.entities.size();
    return snapshot;
}

host_selected_entity_snapshot arc_host::selected_entity_snapshot() const
{
    return entity_snapshot(to_host_entity(state_->scene.selected_entity));
}

host_selected_entity_snapshot arc_host::entity_snapshot(std::string_view guid_text) const
{
    const auto guid = ecs::parse_entity_guid(guid_text);
    if (!guid) return {};
    return entity_snapshot(to_host_entity(find_entity_by_guid(state_->scene, *guid)));
}

host_selected_entity_snapshot arc_host::entity_snapshot(host_entity_id host_entity) const
{
    host_selected_entity_snapshot snapshot;
    const auto selected = to_scene_entity(host_entity);
    if (!state_->scene.scene.alive(selected)) return snapshot;

    snapshot.entity = to_host_entity(selected);
    for (const auto candidate : state_->scene.scene.entities())
    {
        const auto* selection = state_->scene.scene.try_get<scene::selection_component>(candidate);
        if (!selection || !selection->selected) continue;
        ++snapshot.selection_count;
        snapshot.selected_guids.push_back(ecs::to_string(entity_guid_of(state_->scene, candidate)));
    }
    snapshot.guid = ecs::to_string(entity_guid_of(state_->scene, selected));
    snapshot.name = entity_name(state_->scene, selected, "Unnamed Entity");
    if (const auto* tag = state_->scene.scene.try_get<scene::tag_component>(selected)) snapshot.tag = tag->value;
    snapshot.active = entity_active(state_->scene, selected);
    if (const auto* mobility = state_->scene.scene.try_get<scene::mobility_component>(selected))
        snapshot.mobility = to_host_mobility(mobility->value);
    if (const auto* layer = state_->scene.scene.try_get<scene::render_layer_component>(selected))
        snapshot.render_layer_mask = layer->mask;
    if (const auto* transform = state_->scene.scene.try_get<scene::transform_component>(selected))
    {
        snapshot.transform = to_host_transform(*transform);
        add_component_snapshot<scene::transform_component>(state_->scene.scene, selected, snapshot.components,
                                                           host_component_kind::transform, "Transform");
    }
    if (const auto* bounds = state_->scene.scene.try_get<scene::bounds_component>(selected))
    {
        snapshot.bounds = host_bounds_snapshot{
            .minimum = {bounds->world_bounds.min[0], bounds->world_bounds.min[1], bounds->world_bounds.min[2]},
            .maximum = {bounds->world_bounds.max[0], bounds->world_bounds.max[1], bounds->world_bounds.max[2]}};
    }
    if (const auto* camera = state_->scene.scene.try_get<scene::camera_component>(selected))
    {
        snapshot.camera = to_host_camera(*camera);
        add_component_snapshot<scene::camera_component>(state_->scene.scene, selected, snapshot.components,
                                                        host_component_kind::camera, "Camera");
    }
    if (const auto* mesh_renderer = state_->scene.scene.try_get<scene::mesh_renderer_component>(selected))
    {
        snapshot.mesh_renderer =
            mesh_renderer_snapshot(state_->scene, state_->assets, state_->project.root, selected, *mesh_renderer);
        add_component_snapshot<scene::mesh_renderer_component>(state_->scene.scene, selected, snapshot.components,
                                                               host_component_kind::mesh_renderer, "Mesh Renderer");
    }
    if (const auto* light = state_->scene.scene.try_get<scene::directional_light_component>(selected))
    {
        snapshot.light = to_host_light(*light);
        add_component_snapshot<scene::directional_light_component>(state_->scene.scene, selected, snapshot.components,
                                                                   host_component_kind::directional_light,
                                                                   "Directional Light");
    }
    if (const auto* light = state_->scene.scene.try_get<scene::point_light_component>(selected))
    {
        snapshot.light = to_host_light(*light);
        add_component_snapshot<scene::point_light_component>(state_->scene.scene, selected, snapshot.components,
                                                             host_component_kind::point_light, "Point Light");
    }
    if (const auto* light = state_->scene.scene.try_get<scene::spot_light_component>(selected))
    {
        snapshot.light = to_host_light(*light);
        add_component_snapshot<scene::spot_light_component>(state_->scene.scene, selected, snapshot.components,
                                                            host_component_kind::spot_light, "Spot Light");
    }
    if (const auto* light = state_->scene.scene.try_get<scene::area_light_component>(selected))
    {
        snapshot.light = to_host_light(*light);
        add_component_snapshot<scene::area_light_component>(state_->scene.scene, selected, snapshot.components,
                                                            host_component_kind::area_light, "Area Light");
    }
    if (state_->scene.scene.has<scene::sky_atmosphere_component>(selected))
        add_component_snapshot<scene::sky_atmosphere_component>(state_->scene.scene, selected, snapshot.components,
                                                                host_component_kind::sky_atmosphere, "Sky Atmosphere");
    if (state_->scene.scene.has<scene::world_environment_component>(selected))
        add_component_snapshot<scene::world_environment_component>(state_->scene.scene, selected, snapshot.components,
                                                                   host_component_kind::world_environment,
                                                                   "World Environment");
    if (state_->scene.scene.has<scene::celestial_sky_component>(selected))
        add_component_snapshot<scene::celestial_sky_component>(state_->scene.scene, selected, snapshot.components,
                                                               host_component_kind::celestial_sky, "Sun, Moon & Time");
    if (state_->scene.scene.has<scene::cloud_layers_component>(selected))
        add_component_snapshot<scene::cloud_layers_component>(state_->scene.scene, selected, snapshot.components,
                                                              host_component_kind::cloud_layers, "Cloud Layers");
    if (state_->scene.scene.has<scene::environment_lighting_component>(selected))
        add_component_snapshot<scene::environment_lighting_component>(
            state_->scene.scene, selected, snapshot.components, host_component_kind::environment_lighting,
            "Environment Lighting");
    if (state_->scene.scene.has<scene::height_fog_component>(selected))
        add_component_snapshot<scene::height_fog_component>(state_->scene.scene, selected, snapshot.components,
                                                            host_component_kind::height_fog, "Height Fog");
    if (const auto* terrain = state_->scene.scene.try_get<scene::terrain_component>(selected))
    {
        host_terrain_snapshot terrain_snapshot;
        terrain_snapshot.enabled = terrain->enabled;
        terrain_snapshot.size = terrain->size;
        if (!terrain->heights.empty())
        {
            const auto [minimum, maximum] = std::minmax_element(terrain->heights.begin(), terrain->heights.end());
            terrain_snapshot.minimum_elevation = *minimum;
            terrain_snapshot.maximum_elevation = *maximum;
        }
        terrain_snapshot.resolution = terrain->subdivisions + 1u;
        terrain_snapshot.chunk_quads = terrain->chunk_quads;
        terrain_snapshot.patch_quads = terrain->patch_quads;
        terrain_snapshot.maximum_hierarchy_depth = terrain->maximum_hierarchy_depth;
        terrain_snapshot.geometric_error_multiplier = terrain->geometric_error_multiplier;
        terrain_snapshot.receive_shadows = terrain->receive_shadows;
        terrain_snapshot.cast_shadows = terrain->cast_shadows;
        terrain_snapshot.shadow_lod_bias = terrain->shadow_lod_bias;
        terrain_snapshot.maximum_shadow_distance = terrain->maximum_shadow_distance;
        terrain_snapshot.content_revision = terrain->content_revision;
        terrain_snapshot.cpu_memory_bytes =
            terrain->heights.size() * sizeof(float) + terrain->layer_weights.size() * sizeof(terrain->layer_weights[0]);
        if (const auto* binding = find_asset_binding(state_->scene, entity_guid_of(state_->scene, selected)))
        {
            terrain_snapshot.material_guid = assets::to_string(binding->material.guid);
            terrain_snapshot.material_path = binding->material.path_hint;
        }
        if (const auto* proxy = state_->scene.terrain_render_proxies.find(entity_guid_of(state_->scene, selected)))
        {
            const auto resource = state_->renderer->terrain_snapshot(proxy->handle);
            terrain_snapshot.hierarchy_nodes = resource.hierarchy_nodes;
            terrain_snapshot.source_patches = resource.hierarchy_leaves;
            terrain_snapshot.gpu_memory_bytes = resource.height_bytes + resource.weight_bytes;
            terrain_snapshot.uploaded_bytes = resource.uploaded_height_bytes + resource.uploaded_weight_bytes;
            if (const auto* data = state_->renderer->terrain_data_for(proxy->handle))
                terrain_snapshot.hierarchy_depth = data->hierarchy.maximum_depth;
        }
        const auto& profile = state_->renderer->last_frame_profile().terrain;
        terrain_snapshot.visible_patches = profile.selected_patches;
        terrain_snapshot.rendered_triangles = profile.rendered_triangles;
        terrain_snapshot.brush_tool =
            state_->terrain_brush.tool == scene::terrain_brush_tool::smooth    ? host_terrain_brush_tool::smooth
            : state_->terrain_brush.tool == scene::terrain_brush_tool::flatten ? host_terrain_brush_tool::flatten
            : state_->terrain_brush.tool == scene::terrain_brush_tool::paint   ? host_terrain_brush_tool::paint
                                                                               : host_terrain_brush_tool::sculpt;
        terrain_snapshot.brush_radius = state_->terrain_brush.radius;
        terrain_snapshot.brush_strength = state_->terrain_brush.strength;
        terrain_snapshot.brush_falloff = state_->terrain_brush.falloff;
        terrain_snapshot.active_layer = state_->terrain_brush.active_layer;
        for (std::size_t layer = 0; layer < state_->scene.terrain_layer_paths.size(); ++layer)
            terrain_snapshot.layer_base_color_paths[layer] =
                asset_relative_path(state_->assets.root, state_->scene.terrain_layer_paths[layer]);
        snapshot.terrain = std::move(terrain_snapshot);
        add_component_snapshot<scene::terrain_component>(state_->scene.scene, selected, snapshot.components,
                                                         host_component_kind::terrain, "Terrain");
    }
    if (state_->scene.scene.has<scene::water_component>(selected))
        add_component_snapshot<scene::water_component>(state_->scene.scene, selected, snapshot.components,
                                                       host_component_kind::water, "Water");
    if (state_->scene.scene.has<scene::vegetation_component>(selected))
        add_component_snapshot<scene::vegetation_component>(state_->scene.scene, selected, snapshot.components,
                                                            host_component_kind::vegetation, "Vegetation");
    if (state_->scene.scene.has<scene::decal_component>(selected))
        add_component_snapshot<scene::decal_component>(state_->scene.scene, selected, snapshot.components,
                                                       host_component_kind::decal, "Decal");
    if (const auto* prefab = state_->scene.scene.try_get<scene::prefab_instance_component>(selected))
    {
        snapshot.prefab = host_prefab_snapshot{.prefab_guid = ecs::to_string(prefab->prefab_guid),
                                               .prefab_path = prefab->prefab_path,
                                               .override_count = prefab->overrides.size(),
                                               .source_missing = prefab->source_missing};
        for (const auto& value : prefab->overrides)
        {
            const char* kind =
                value.key.kind == ecs::prefab_override_kind::component_added        ? "componentAdded"
                : value.key.kind == ecs::prefab_override_kind::component_removed    ? "componentRemoved"
                : value.key.kind == ecs::prefab_override_kind::source_child_removed ? "sourceChildRemoved"
                : value.key.kind == ecs::prefab_override_kind::instance_child_added ? "instanceChildAdded"
                                                                                    : "field";
            snapshot.prefab->overrides.push_back({.source_entity = ecs::to_string(value.key.source_entity),
                                                  .component_id = ecs::to_string(value.key.component),
                                                  .field_id = value.key.field,
                                                  .kind = kind});
        }
        add_component_snapshot<scene::prefab_instance_component>(state_->scene.scene, selected, snapshot.components,
                                                                 host_component_kind::prefab_instance,
                                                                 "Prefab Instance");
    }
    const auto project_records = project_component_records(state_->scene, entity_guid_of(state_->scene, selected));
    for (const auto& schema : state_->project_module.component_schemas())
    {
        const auto record_key = project_component_record_key(project_records, schema);
        if (!record_key) continue;
        auto values = default_project_component(schema);
        values.update(project_records.at(*record_key));
        values.erase("typeId");
        values.erase("version");
        values.erase("_arcFieldIds");
        snapshot.project_components.push_back({.type_id = schema.stable_id,
                                               .canonical_name = schema.canonical_name,
                                               .display_name = schema.display_name,
                                               .schema_version = schema.schema_version,
                                               .values_json = values.dump()});
    }
    return snapshot;
}

std::optional<host_world_environment_snapshot> arc_host::world_environment_snapshot(host_entity_id host_entity) const
{
    const auto settings = scene::read_world_environment_settings(state_->scene.scene, to_scene_entity(host_entity));
    if (!settings) return std::nullopt;
    return to_host_world_environment_snapshot(host_entity, *settings, state_->scene.world_environment_hdri_path);
}

host_project_assets_snapshot arc_host::project_assets_snapshot() const
{
    host_project_assets_snapshot snapshot;
    snapshot.project_name = state_->project.name;
    snapshot.project_root = state_->project.root;
    snapshot.asset_root = state_->assets.root;
    snapshot.default_mesh_path = state_->assets.default_mesh_path.generic_string();
    snapshot.default_mesh_loaded = state_->assets.default_mesh_loaded;
    snapshot.default_mesh_message = state_->assets.default_mesh_message;
    if (state_->derived_cache)
    {
        const auto cache = state_->derived_cache->statistics();
        snapshot.cache_root = state_->derived_cache->config().root;
        snapshot.cache_local_bytes = cache.local_bytes;
        snapshot.cache_local_hits = cache.local_hits;
        snapshot.cache_local_misses = cache.local_misses;
        snapshot.cache_shared_hits = cache.shared_hits;
        snapshot.cache_shared_misses = cache.shared_misses;
        snapshot.cache_corrupt_entries = cache.corrupt_entries;
        snapshot.cache_evictions = cache.evictions;
        snapshot.cache_hit_rate = cache.hit_rate();
    }
    if (state_->asset_registry)
    {
        const auto state_name = [](arc::assets::asset_state value)
        {
            switch (value)
            {
                case arc::assets::asset_state::unknown:
                    return "unknown";
                case arc::assets::asset_state::queued:
                    return "queued";
                case arc::assets::asset_state::importing:
                    return "importing";
                case arc::assets::asset_state::ready:
                    return "ready";
                case arc::assets::asset_state::stale:
                    return "stale";
                case arc::assets::asset_state::failed:
                    return "failed";
            }
            return "unknown";
        };
        const auto residency_name = [](arc::assets::asset_residency value)
        {
            switch (value)
            {
                case arc::assets::asset_residency::metadata_only:
                    return "metadata";
                case arc::assets::asset_residency::source:
                    return "source";
                case arc::assets::asset_residency::derived:
                    return "derived";
                case arc::assets::asset_residency::cpu:
                    return "cpu";
                case arc::assets::asset_residency::device:
                    return "device";
            }
            return "metadata";
        };
        const auto kind_name = [](arc::assets::asset_type_id type)
        {
            if (type == arc::assets::asset_types::scene) return "scene";
            if (type == arc::assets::asset_types::prefab) return "prefab";
            if (type == arc::assets::asset_types::material) return "material";
            if (type == arc::assets::asset_types::shader) return "shader";
            if (type == arc::assets::asset_types::texture_2d) return "texture";
            if (type == arc::assets::asset_types::environment) return "environment";
            if (type == arc::assets::asset_types::imported_scene) return "scene";
            if (type == arc::assets::asset_types::static_mesh) return "mesh";
            return "unknown";
        };
        const auto texture_format_name = [](render::texture_format format)
        {
            switch (format)
            {
                case render::texture_format::rgba8_unorm:
                    return "RGBA8 UNorm";
                case render::texture_format::rgba8_srgb:
                    return "RGBA8 sRGB";
                case render::texture_format::rgba16f:
                    return "RGBA16F";
                case render::texture_format::rgba32f:
                    return "RGBA32F";
                case render::texture_format::bc1_rgba_unorm:
                    return "BC1 UNorm";
                case render::texture_format::bc1_rgba_srgb:
                    return "BC1 sRGB";
                case render::texture_format::bc2_rgba_unorm:
                    return "BC2 UNorm";
                case render::texture_format::bc2_rgba_srgb:
                    return "BC2 sRGB";
                case render::texture_format::bc3_rgba_unorm:
                    return "BC3 UNorm";
                case render::texture_format::bc3_rgba_srgb:
                    return "BC3 sRGB";
                case render::texture_format::bc4_r_unorm:
                    return "BC4";
                case render::texture_format::bc5_rg_unorm:
                    return "BC5";
                case render::texture_format::bc6h_rgb_ufloat:
                    return "BC6H";
                case render::texture_format::bc7_rgba_unorm:
                    return "BC7 UNorm";
                case render::texture_format::bc7_rgba_srgb:
                    return "BC7 sRGB";
            }
            return "Unknown";
        };
        for (const auto& asset : state_->asset_registry->search())
        {
            const bool running =
                asset.state == arc::assets::asset_state::queued || asset.state == arc::assets::asset_state::importing;
            const auto absolute_path = state_->project.root / asset.source_path;
            auto display_path = absolute_path.lexically_relative(state_->assets.root);
            if (display_path.empty() || display_path.native().starts_with(std::filesystem::path("..").native()))
                display_path = asset.source_path;
            host_asset_snapshot host_asset{.guid = arc::assets::to_string(asset.guid),
                                           .path = arc::assets::normalize_asset_path(display_path),
                                           .scope = asset.read_only ? "builtin" : "project",
                                           .kind = kind_name(asset.type),
                                           .type_id = arc::assets::to_string(asset.type),
                                           .importer_id = arc::assets::to_string(asset.importer),
                                           .state = state_name(asset.state),
                                           .residency = residency_name(asset.residency),
                                           .generation = asset.generation,
                                           .strong_references = asset.strong_references,
                                           .pins = asset.pins,
                                           .diagnostic = asset.diagnostics.empty() ? std::string{}
                                                                                   : asset.diagnostics.back().message};
            for (const auto& artifact : asset.artifacts)
                host_asset.artifact_size += artifact.size;
            if (asset.type == arc::assets::asset_types::texture_2d)
            {
                const auto metadata_path = assets::metadata_path_for(absolute_path);
                if (const auto metadata = assets::load_asset_metadata(metadata_path); metadata)
                {
                    host_asset.settings_version = metadata.value().settings_version;
                    const auto settings = render::tools::parse_texture_import_settings(
                        metadata.value().canonical_settings, metadata.value().settings_version);
                    if (settings)
                        host_asset.streaming_mode =
                            settings.value().streaming_mode == render::texture_streaming_mode::streamed_mips
                                ? "streamed_mips"
                            : settings.value().streaming_mode == render::texture_streaming_mode::virtual_tiles
                                ? "virtual_tiles"
                                : "resident";
                    else
                        host_asset.streaming_eligibility_error = settings.error();
                }
                if (!render::is_supported_texture_asset(absolute_path) || absolute_path.extension() == ".hdr")
                    host_asset.streaming_eligibility_error =
                        "Streaming supports DDS, PNG, JPEG, TGA, and BMP ordinary 2D textures";
                else
                {
                    const auto texture = render::inspect_texture_asset(absolute_path);
                    if (texture.succeeded())
                    {
                        host_asset.width = texture.width;
                        host_asset.height = texture.height;
                        host_asset.texture_format = texture_format_name(texture.format);
                        host_asset.mip_count = texture.mip_count;
                        if (host_asset.streaming_mode == "virtual_tiles")
                        {
                            auto mip_width = texture.width;
                            auto mip_height = texture.height;
                            for (std::uint32_t mip = 0; mip < texture.mip_count; ++mip)
                            {
                                if (mip_width <= render::virtual_texture_tile_size &&
                                    mip_height <= render::virtual_texture_tile_size)
                                    break;
                                host_asset.tile_count += ((mip_width + render::virtual_texture_tile_size - 1u) /
                                                          render::virtual_texture_tile_size) *
                                                         ((mip_height + render::virtual_texture_tile_size - 1u) /
                                                          render::virtual_texture_tile_size);
                                mip_width = std::max(1u, mip_width / 2u);
                                mip_height = std::max(1u, mip_height / 2u);
                            }
                        }
                    }
                    else
                        host_asset.streaming_eligibility_error = texture.message;
                }
            }
            host_asset.dependencies.reserve(asset.dependencies.size());
            for (const auto dependency : asset.dependencies)
                host_asset.dependencies.push_back(arc::assets::to_string(dependency));
            host_asset.reverse_dependencies.reserve(asset.reverse_dependencies.size());
            for (const auto dependency : asset.reverse_dependencies)
                host_asset.reverse_dependencies.push_back(arc::assets::to_string(dependency));
            host_asset.read_only = asset.read_only;
            host_asset.imported = asset.state == arc::assets::asset_state::ready;
            host_asset.import_running = running;
            host_asset.has_last_good = asset.has_last_good;
            snapshot.assets.push_back(std::move(host_asset));
        }
        return snapshot;
    }
    if (!state_->assets.default_mesh_path.empty())
    {
        snapshot.assets.push_back({.path = state_->assets.default_mesh_path.generic_string(),
                                   .kind = "scene",
                                   .imported = state_->assets.default_mesh_loaded,
                                   .import_running = false});
    }
    for (const auto& material : state_->scene.material_library.materials)
    {
        snapshot.assets.push_back({.path = asset_relative_path(state_->assets.root, material.path),
                                   .kind = "material",
                                   .imported = true,
                                   .import_running = false});
    }
    if (!state_->assets.root.empty())
    {
        std::error_code error;
        for (std::filesystem::recursive_directory_iterator
                 iterator(state_->assets.root, std::filesystem::directory_options::skip_permission_denied, error),
             end;
             iterator != end && !error; iterator.increment(error))
        {
            if (!iterator->is_regular_file(error)) continue;
            auto extension = iterator->path().extension().string();
            std::transform(extension.begin(), extension.end(), extension.begin(),
                           [](unsigned char value) { return static_cast<char>(std::tolower(value)); });
            const bool texture_asset = render::is_supported_texture_asset(iterator->path());
            const bool material_asset_path = is_material_asset_path(iterator->path());
            const bool prefab_asset_path = extension == ".arcprefab";
            if (!texture_asset && !material_asset_path && !prefab_asset_path) continue;
            const auto relative_path =
                std::filesystem::relative(iterator->path(), state_->assets.root, error).generic_string();
            if (std::any_of(snapshot.assets.begin(), snapshot.assets.end(),
                            [&](const auto& asset) { return asset.path == relative_path; }))
                continue;
            snapshot.assets.push_back({.path = relative_path,
                                       .kind = prefab_asset_path     ? "prefab"
                                               : material_asset_path ? "material"
                                               : extension == ".hdr" ? "environment"
                                                                     : "texture",
                                       .imported = true,
                                       .import_running = false});
        }
    }
    return snapshot;
}

std::optional<host_asset_thumbnail_snapshot> arc_host::asset_thumbnail(std::string_view path,
                                                                       std::uint32_t max_size) const
{
    const auto resolved = resolve_editor_asset(state_->assets, state_->asset_registry.get(), state_->project.root,
                                               std::filesystem::path{path});
    if (!resolved) return std::nullopt;

    const auto requested_size = std::clamp(max_size, 32u, 256u);
    auto extension = resolved->path.extension().string();
    std::transform(extension.begin(), extension.end(), extension.begin(),
                   [](unsigned char value) { return static_cast<char>(std::tolower(value)); });
    const bool model_asset = extension == ".fbx" || extension == ".obj" || extension == ".glb" || extension == ".gltf";

    render::texture_data preview;
    if (model_asset)
    {
        // Model thumbnails are expensive enough that both the React thumbnail provider
        // and the native host keep them cached. The file timestamp makes reimport/source
        // edits invalidate this cache without coupling the renderer to asset generations.
        struct model_thumbnail_cache_entry
        {
            std::filesystem::file_time_type source_time{};
            host_asset_thumbnail_snapshot snapshot;
        };
        static std::unordered_map<std::string, model_thumbnail_cache_entry> model_thumbnail_cache;

        std::error_code timestamp_error;
        const auto source_time = std::filesystem::last_write_time(resolved->path, timestamp_error);
        const auto cache_key =
            resolved->path.lexically_normal().generic_string() + ":" + std::to_string(requested_size);
        if (!timestamp_error)
            if (const auto cached = model_thumbnail_cache.find(cache_key);
                cached != model_thumbnail_cache.end() && cached->second.source_time == source_time)
                return cached->second.snapshot;

        render::scene_import_options import_options;
        import_options.asset_root = resolved->asset_root;
        import_options.import_directory = resolved->path.parent_path();
        import_options.copy_assets = false;
        const auto imported = render::load_scene_asset(resolved->path, import_options);
        if (!imported.succeeded()) return std::nullopt;

        model_preview_options preview_options{.size = requested_size};
        // V1 intentionally uses the neutral/default Phong-style material. The
        // model_preview_options contract already accepts a material override so
        // model/material binding resolution can opt into an imported or assigned
        // material later without changing asset.thumbnail or the Content Browser.
        auto rendered = render_model_preview(imported, preview_options);
        if (!rendered.succeeded()) return std::nullopt;
        preview = std::move(rendered.texture);

        const auto png = texture_preview_png_rgba(preview);
        if (png.empty()) return std::nullopt;
        const float scale = std::min(1.0f, static_cast<float>(requested_size) /
                                               static_cast<float>(std::max(preview.width, preview.height)));
        host_asset_thumbnail_snapshot snapshot{
            .path = std::string(path),
            .width = std::max(1u, static_cast<std::uint32_t>(std::lround(preview.width * scale))),
            .height = std::max(1u, static_cast<std::uint32_t>(std::lround(preview.height * scale))),
            .data_url = "data:image/png;base64," + base64_encode(png)};
        if (!timestamp_error) model_thumbnail_cache[cache_key] = {.source_time = source_time, .snapshot = snapshot};
        return snapshot;
    }

    if (is_material_asset_path(resolved->path))
    {
        material_asset material;
        std::string message;
        if (!load_material_asset(resolved->path, resolved->asset_root, material, message)) return std::nullopt;
        auto rendered = render_material_preview(material, resolved->asset_root, requested_size);
        if (!rendered.succeeded()) return std::nullopt;
        preview = std::move(rendered.texture);
    }
    else
    {
        if (!render::is_supported_texture_asset(resolved->path)) return std::nullopt;
        const auto loaded = render::load_texture_asset(resolved->path);
        if (!loaded.succeeded()) return std::nullopt;
        preview = loaded.texture;
    }
    const auto bmp = texture_preview_bmp(preview, requested_size);
    if (bmp.empty()) return std::nullopt;
    const float scale = std::min(1.0f, static_cast<float>(requested_size) /
                                           static_cast<float>(std::max(preview.width, preview.height)));
    return host_asset_thumbnail_snapshot{
        .path = std::string(path),
        .width = std::max(1u, static_cast<std::uint32_t>(std::lround(preview.width * scale))),
        .height = std::max(1u, static_cast<std::uint32_t>(std::lround(preview.height * scale))),
        .data_url = "data:image/bmp;base64," + base64_encode(bmp)};
}

std::vector<host_event> arc_host::poll_events()
{
    auto events = std::move(state_->events);
    state_->events.clear();
    return events;
}

host_viewport_frame arc_host::request_viewport(const host_viewport_request& request)
{
    if (request.viewport_id != state_->active_viewport_id)
        return {.message = "Viewport render skipped: viewport is not attached"};
    if (state_->asset_registry)
    {
        state_->asset_registry->poll();
        for (const auto& event : state_->asset_registry->events_since(state_->asset_event_cursor))
        {
            state_->asset_event_cursor = std::max(state_->asset_event_cursor, event.sequence);
            push_event(state_->events, state_->event_sequence, host_event_type::asset_changed, event.message, {},
                       "{\"guid\":\"" + arc::assets::to_string(event.guid) +
                           "\",\"registryRevision\":" + std::to_string(event.registry_revision) +
                           ",\"state\":" + std::to_string(static_cast<unsigned>(event.state)) +
                           ",\"progress\":" + std::to_string(event.progress) + "}");
        }
    }
    const auto frame_start = std::chrono::steady_clock::now();
    float delta_seconds = 0.0f;
    if (state_->last_viewport_frame_time.time_since_epoch().count() != 0)
        delta_seconds = std::chrono::duration<float>(frame_start - state_->last_viewport_frame_time).count();
    const framework::frame_time runtime_frame = state_->simulation.advance(delta_seconds);
    if (runtime_frame.last_completed_tick.value != state_->last_runtime_tick_event)
    {
        state_->last_runtime_tick_event = runtime_frame.last_completed_tick.value;
        ++state_->runtime_revision;
        host_runtime_snapshot snapshot = runtime_snapshot();
        snapshot.interpolation_alpha = runtime_frame.interpolation_alpha;
        push_event(state_->events, state_->event_sequence,
                   snapshot.state == host_runtime_state::faulted ? host_event_type::runtime_fault
                                                                 : host_event_type::runtime_tick_completed,
                   snapshot.state == host_runtime_state::faulted ? "Preview runtime faulted"
                                                                 : "Preview runtime tick completed",
                   {}, to_json(snapshot));
    }
    // Native presentation submits a lightweight request every frame. Preserve
    // the render/debug options selected through viewport.setRenderOptions
    // instead of resetting them to host_viewport_request defaults at 60 Hz.
    state_->viewport_options.frame_index = request.frame_index;
    state_->viewport_options.width = request.width;
    state_->viewport_options.height = request.height;
    state_->viewport_frame_index = request.frame_index;
    if (!state_->renderer->backend())
    {
        const std::string message = "Viewport render skipped: no render backend attached";
        state_->viewport_submitted = false;
        push_event(state_->events, state_->event_sequence, host_event_type::viewport_error, message);
        return {.message = message};
    }

    if (state_->pending_pick)
    {
        const auto gpu = state_->renderer->last_object_pick();
        const bool gpu_ready = gpu.available && gpu.request_id == state_->pending_pick->request_id &&
                               gpu.x == state_->pending_pick->x && gpu.y == state_->pending_pick->y &&
                               gpu.frame_index > state_->pending_pick->requested_after_frame;
        const bool fallback_due = request.frame_index >= state_->pending_pick->requested_after_frame +
                                                             defaults::viewport_pick_fallback_frame_count;
        if (gpu_ready || fallback_due)
        {
            ecs::entity picked = gpu_ready && gpu.hit ? ecs::entity{gpu.object.index, gpu.object.generation}
                                                      : state_->pending_pick->cpu_fallback.entity;
            if (gpu_ready && gpu.hit && state_->scene.scene.alive(picked))
            {
                const bool gpu_background = state_->scene.scene.has<scene::terrain_component>(picked) ||
                                            state_->scene.scene.has<scene::water_component>(picked) ||
                                            state_->scene.scene.has<scene::world_environment_component>(picked);
                if (gpu_background && state_->pending_pick->cpu_fallback.entity.valid() &&
                    !state_->pending_pick->cpu_fallback.background && state_->pending_pick->cpu_fallback.exact)
                    picked = state_->pending_pick->cpu_fallback.entity;
            }
            if (gpu_ready && gpu.hit && !state_->scene.scene.alive(picked))
                picked = state_->pending_pick->cpu_fallback.entity;
            arc::diagnostics::debug("editor.pick", "pick request=" + std::to_string(state_->pending_pick->request_id) +
                                                       " source=" + std::string(gpu_ready && gpu.hit ? "gpu" : "cpu") +
                                                       " x=" + std::to_string(state_->pending_pick->x) +
                                                       " y=" + std::to_string(state_->pending_pick->y) +
                                                       " entity=" + std::to_string(picked.index) + " cpuDistance=" +
                                                       std::to_string(state_->pending_pick->cpu_fallback.distance));
            const auto previous_selection = state_->scene.selected_entity;
            if (state_->scene.scene.alive(picked))
                select_entity(state_->scene.scene, picked, state_->scene.selected_entity);
            else
                clear_selection(state_->scene.scene, state_->scene.selected_entity);
            if (state_->scene.selected_entity != previous_selection)
                push_event(state_->events, state_->event_sequence, host_event_type::entity_selected,
                           picked.valid() ? "Viewport entity selected" : "Viewport selection cleared", picked);
            state_->pending_pick.reset();
        }
    }

    if (state_->scene.focus_imported_scene_requested)
    {
        focus_selected_entity(state_->scene.scene, state_->scene.selected_entity, state_->camera_controller);
        if (auto* camera_transform =
                state_->scene.scene.try_get<scene::transform_component>(state_->scene.camera_entity))
            state_->camera_controller.apply_to(*camera_transform);
        state_->scene.focus_imported_scene_requested = false;
    }

    auto debug_overlay = build_editor_gizmo_overlay(
        state_->scene.scene, state_->scene.selected_entity, state_->scene.camera_entity,
        editor_gizmo_context{.tool = to_editor_tool(state_->viewport_tool.tool),
                             .coordinate_space = state_->viewport_tool.coordinate_space == host_coordinate_space::local
                                                     ? gizmo_coordinate_space::local
                                                     : gizmo_coordinate_space::world,
                             .highlighted_axis = state_->gizmo_highlight,
                             .viewport_width = request.width,
                             .viewport_height = request.height});
    if (state_->viewport_options.grid)
    {
        const auto* camera = state_->scene.scene.try_get<scene::camera_component>(state_->scene.camera_entity);
        const auto* camera_transform =
            state_->scene.scene.try_get<scene::transform_component>(state_->scene.camera_entity);
        if (camera && camera_transform)
            append_editor_grid_overlay(debug_overlay, *camera, *camera_transform, request.height);
    }
    if (state_->viewport_tool.tool == host_viewport_tool::terrain && state_->terrain_brush_local_position &&
        state_->scene.scene.alive(state_->scene.terrain_entity))
    {
        const auto* terrain = state_->scene.scene.try_get<scene::terrain_component>(state_->scene.terrain_entity);
        const auto* transform = state_->scene.scene.try_get<scene::transform_component>(state_->scene.terrain_entity);
        if (terrain && transform)
        {
            constexpr std::uint32_t segment_count = 64u;
            const auto world = transform->dirty ? scene::local_matrix(*transform) : transform->world;
            for (std::uint32_t segment = 0; segment < segment_count; ++segment)
            {
                const float angle0 = math::tau<float> * static_cast<float>(segment) / segment_count;
                const float angle1 = math::tau<float> * static_cast<float>(segment + 1u) / segment_count;
                auto point0 = math::vector3f{
                    (*state_->terrain_brush_local_position)[0] + std::cos(angle0) * state_->terrain_brush.radius, 0.0f,
                    (*state_->terrain_brush_local_position)[2] + std::sin(angle0) * state_->terrain_brush.radius};
                auto point1 = math::vector3f{
                    (*state_->terrain_brush_local_position)[0] + std::cos(angle1) * state_->terrain_brush.radius, 0.0f,
                    (*state_->terrain_brush_local_position)[2] + std::sin(angle1) * state_->terrain_brush.radius};
                point0[1] = scene::sample_terrain_height(*terrain, point0[0], point0[2]) + 0.04f;
                point1[1] = scene::sample_terrain_height(*terrain, point1[0], point1[2]) + 0.04f;
                debug_overlay.lines.push_back({.start = math::transform_point(world, point0),
                                               .end = math::transform_point(world, point1),
                                               .color = {0.95f, 0.68f, 0.16f, 1.0f},
                                               .depth = render::debug_overlay_depth_mode::tested});
            }
        }
    }
    state_->scene.last_render = scene::render_scene(
        state_->scene.scene, *state_->renderer, request.width, request.height,
        to_render_mode(state_->viewport_options.render_mode), to_visualization(state_->viewport_options.visualization),
        to_overlay(state_->viewport_options.overlay), state_->viewport_options.shadows,
        to_scene_visibility(state_->viewport_options.environment), delta_seconds, std::move(debug_overlay),
        state_->scene.camera_entity, &state_->scene.terrain_render_proxies);

    auto view_config = state_->renderer->resolved_config();
    auto requested_anti_aliasing = to_render_anti_aliasing(state_->viewport_options.anti_aliasing);
    if (state_->viewport_options.anti_aliasing == host_camera_anti_aliasing::inherit)
    {
        if (const auto* camera = state_->scene.scene.try_get<scene::camera_component>(state_->scene.camera_entity);
            camera && camera->anti_aliasing != render::camera_anti_aliasing_override::inherit)
            requested_anti_aliasing = static_cast<render::anti_aliasing_method>(camera->anti_aliasing);
        else
            requested_anti_aliasing = state_->renderer->config().anti_aliasing;
    }
    view_config.requested_anti_aliasing = requested_anti_aliasing;
    view_config.anti_aliasing = state_->renderer->resolve_view_anti_aliasing(requested_anti_aliasing);
    view_config.features.anti_aliasing = view_config.anti_aliasing;
    view_config.features.fxaa = view_config.anti_aliasing == render::anti_aliasing_method::fxaa;
    view_config.features.temporal_antialiasing = view_config.anti_aliasing == render::anti_aliasing_method::taa ||
                                                 view_config.anti_aliasing == render::anti_aliasing_method::taau;
    view_config.features.temporal_upscaling = view_config.anti_aliasing == render::anti_aliasing_method::taau;
    const auto submit_result = state_->renderer->render_frame(
        request.frame_index,
        render::make_scene_draw_graph("viewport", view_config, true, state_->scene.last_render.environment));
    const auto frame_end = std::chrono::steady_clock::now();
    state_->viewport_frame_ms = std::chrono::duration<double, std::milli>(frame_end - frame_start).count();
    if (state_->last_viewport_frame_time.time_since_epoch().count() != 0)
    {
        const double frame_interval_seconds =
            std::chrono::duration<double>(frame_end - state_->last_viewport_frame_time).count();
        if (frame_interval_seconds > 0.0) state_->viewport_fps = 1.0 / frame_interval_seconds;
    }
    state_->last_viewport_frame_time = frame_end;
    state_->viewport_draw_calls = static_cast<std::uint32_t>(std::min<std::size_t>(
        state_->scene.last_render.submitted_draw_count, std::numeric_limits<std::uint32_t>::max()));
    state_->viewport_submitted = submit_result.has_value();
    if (!submit_result && !submit_result.error().message.empty())
    {
        arc::diagnostics::error("editor.host", submit_result.error().message);
        push_event(state_->events, state_->event_sequence, host_event_type::viewport_error,
                   submit_result.error().message);
    }

    return {.submitted = submit_result.has_value(),
            .message = submit_result ? std::string{} : submit_result.error().message,
            .payload_json = "{\"drawCalls\":" + std::to_string(state_->scene.last_render.submitted_draw_count) + '}'};
}

render::renderer& arc_host::renderer_service() noexcept
{
    return *state_->renderer;
}

const render::renderer& arc_host::renderer_service() const noexcept
{
    return *state_->renderer;
}

editor_scene_state& arc_host::scene_state() noexcept
{
    return state_->scene;
}

const editor_scene_state& arc_host::scene_state() const noexcept
{
    return state_->scene;
}

const host_viewport_set_tool_command& arc_host::viewport_tool_state() const noexcept
{
    return state_->viewport_tool;
}

void arc_host::set_viewport_gizmo_highlight(gizmo_axis axis) noexcept
{
    state_->gizmo_highlight = axis;
}

in_process_host_session::in_process_host_session(std::shared_ptr<arc_host> host) : host_(std::move(host)) {}

host_response in_process_host_session::execute(const host_command_envelope& command)
{
    return host_->execute(command);
}

host_response in_process_host_session::query(const host_query_envelope& query)
{
    return host_->query(query);
}

std::vector<host_event> in_process_host_session::poll_events()
{
    return host_->poll_events();
}

host_viewport_frame in_process_host_session::request_viewport(const host_viewport_request& request)
{
    return host_->request_viewport(request);
}

std::string stdio_host_session::command_line(const host_command_envelope& command)
{
    return to_json(command) + '\n';
}

std::string stdio_host_session::query_line(const host_query_envelope& query)
{
    return to_json(query) + '\n';
}

std::shared_ptr<arc_host> arc_host_manager::acquire(std::unique_ptr<render::renderer> renderer)
{
    if (auto existing = host_.lock())
    {
        arc::diagnostics::info("editor.host", "Acquired existing Arc Host");
        return existing;
    }

    auto created = std::make_shared<arc_host>(std::move(renderer));
    host_ = created;
    return created;
}

} // namespace arc::editor
