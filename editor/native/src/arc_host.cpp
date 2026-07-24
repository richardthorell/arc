#include <arc/editor/arc_host.h>

#include <arc/diagnostics/diagnostics.h>
#include <arc/editor/editor_defaults.h>
#include <arc/editor/editor_gizmo.h>
#include <arc/editor/editor_history.h>
#include <arc/editor/editor_interaction.h>
#include <arc/editor/editor_state.h>
#include <arc/editor/material_preview.h>
#include <arc/editor/prefab_document.h>
#include <arc/editor/scene_document.h>
#include <arc/editor/world_environment_host.h>
#include <arc/geometric/box.h>
#include <arc/framework/framework.h>
#include <arc/render/render.h>
#include <arc/scene/scene.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cctype>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <memory>
#include <optional>
#include <sstream>
#include <string>
#include <type_traits>
#include <unordered_set>
#include <utility>
#include <vector>

namespace arc::editor
{
namespace
{

constexpr std::string_view base64_alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

template <class Component>
void copy_component(const scene::registry& source, scene::registry& target, scene::entity from, scene::entity to)
{
    if (const auto* value = source.try_get<Component>(from))
        target.emplace<Component>(to, *value);
}

scene::entity duplicate_entity_subtree(editor_scene_state& state, scene::entity source, scene::entity parent = {})
{
    if (!state.scene.alive(source))
        return {};
    const auto duplicate = state.scene.create();
    copy_component<scene::name_component>(state.scene, state.scene, source, duplicate);
    if (auto* name = state.scene.try_get<scene::name_component>(duplicate)) name->value += " Copy";
    copy_component<scene::tag_component>(state.scene, state.scene, source, duplicate);
    copy_component<scene::active_component>(state.scene, state.scene, source, duplicate);
    copy_component<scene::transform_component>(state.scene, state.scene, source, duplicate);
    copy_component<scene::bounds_component>(state.scene, state.scene, source, duplicate);
    copy_component<scene::camera_component>(state.scene, state.scene, source, duplicate);
    copy_component<scene::mesh_renderer_component>(state.scene, state.scene, source, duplicate);
    copy_component<scene::virtual_mesh_renderer_component>(state.scene, state.scene, source, duplicate);
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
    state.scene.emplace<scene::persistent_id_component>(duplicate, scene::generate_entity_guid());
    state.scene.emplace<scene::hierarchy_component>(duplicate);
    state.scene.emplace<scene::selection_component>(duplicate, false);
    const auto new_guid = entity_guid_of(state, duplicate);
    if (const auto* binding = find_asset_binding(state, entity_guid_of(state, source)))
    {
        auto copied = *binding;
        copied.entity = new_guid;
        state.asset_bindings.push_back(std::move(copied));
    }
    if (state.scene.alive(parent))
        scene::reparent(state.scene, duplicate, parent, {}, scene::reparent_transform_policy::preserve_local);
    for (const auto child : scene::children(state.scene, source))
        duplicate_entity_subtree(state, child, duplicate);
    return duplicate;
}

template <class Command>
constexpr bool is_authoring_command() noexcept
{
    return std::is_same_v<Command, host_create_entity_command> || std::is_same_v<Command, host_delete_entity_command> ||
        std::is_same_v<Command, host_duplicate_entity_command> || std::is_same_v<Command, host_reparent_entity_command> ||
        std::is_same_v<Command, host_create_prefab_command> || std::is_same_v<Command, host_instantiate_prefab_command> ||
        std::is_same_v<Command, host_revert_prefab_command> ||
        std::is_same_v<Command, host_unpack_prefab_command> ||
        std::is_same_v<Command, host_reorder_entity_command> || std::is_same_v<Command, host_rename_entity_command> ||
        std::is_same_v<Command, host_set_active_command> || std::is_same_v<Command, host_set_tag_command> ||
        std::is_same_v<Command, host_set_transform_command> || std::is_same_v<Command, host_set_render_layer_command> ||
        std::is_same_v<Command, host_set_camera_command> || std::is_same_v<Command, host_set_light_command> ||
        std::is_same_v<Command, host_set_mesh_renderer_command> ||
        std::is_same_v<Command, host_set_terrain_command> || std::is_same_v<Command, host_terrain_stroke_command> ||
        std::is_same_v<Command, host_set_terrain_layer_command> ||
        std::is_same_v<Command, host_set_entity_material_command> || std::is_same_v<Command, host_set_world_environment_command> ||
        std::is_same_v<Command, host_apply_world_environment_preset_command> || std::is_same_v<Command, host_set_environment_hdri_command>;
}

std::string history_label(const host_command_payload& command)
{
    return std::visit([](const auto& value) -> std::string {
        using type = std::decay_t<decltype(value)>;
        if constexpr (std::is_same_v<type, host_create_entity_command>) return "Create Entity";
        if constexpr (std::is_same_v<type, host_delete_entity_command>) return "Delete Entity";
        if constexpr (std::is_same_v<type, host_duplicate_entity_command>) return "Duplicate Entity";
        if constexpr (std::is_same_v<type, host_create_prefab_command>) return "Create Prefab";
        if constexpr (std::is_same_v<type, host_instantiate_prefab_command>) return "Instantiate Prefab";
        if constexpr (std::is_same_v<type, host_apply_prefab_command>) return "Apply Prefab";
        if constexpr (std::is_same_v<type, host_revert_prefab_command>) return "Revert Prefab";
        if constexpr (std::is_same_v<type, host_unpack_prefab_command>) return "Unpack Prefab";
        if constexpr (std::is_same_v<type, host_reparent_entity_command>) return "Reparent Entity";
        if constexpr (std::is_same_v<type, host_reorder_entity_command>) return "Reorder Entity";
        if constexpr (std::is_same_v<type, host_rename_entity_command>) return "Rename Entity";
        if constexpr (std::is_same_v<type, host_set_transform_command>) return "Transform Entity";
        if constexpr (std::is_same_v<type, host_set_entity_material_command>) return "Assign Material";
        if constexpr (std::is_same_v<type, host_set_terrain_command>) return "Edit Terrain";
        if constexpr (std::is_same_v<type, host_set_terrain_layer_command>) return "Assign Terrain Layer";
        if constexpr (std::is_same_v<type, host_terrain_stroke_command>) return "Terrain Stroke";
        if constexpr (std::is_same_v<type, host_set_world_environment_command> ||
            std::is_same_v<type, host_apply_world_environment_preset_command> ||
            std::is_same_v<type, host_set_environment_hdri_command>) return "Edit World Environment";
        return "Edit Component";
    }, command);
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
    const float srgb = mapped <= 0.0031308f
        ? mapped * 12.92f
        : 1.055f * std::pow(mapped, 1.0f / 2.4f) - 0.055f;
    return static_cast<std::uint8_t>(std::clamp(std::lround(srgb * 255.0f), 0l, 255l));
}

std::vector<std::byte> texture_preview_bmp(const render::texture_data& texture, std::uint32_t max_size)
{
    if (!texture.has_pixels() || texture.width == 0u || texture.height == 0u)
        return {};
    const bool float_pixels = texture.format == render::texture_format::rgba32f;
    const bool byte_pixels = texture.format == render::texture_format::rgba8_unorm ||
        texture.format == render::texture_format::rgba8_srgb;
    const std::size_t bytes_per_pixel = float_pixels ? sizeof(float) * 4u : 4u;
    const std::size_t base_level_size = static_cast<std::size_t>(texture.width) * texture.height * bytes_per_pixel;
    if ((!float_pixels && !byte_pixels) || texture.pixels.size() < base_level_size)
        return {};

    const float scale = std::min(
        1.0f,
        static_cast<float>(max_size) / static_cast<float>(std::max(texture.width, texture.height)));
    const auto width = std::max(1u, static_cast<std::uint32_t>(std::lround(texture.width * scale)));
    const auto height = std::max(1u, static_cast<std::uint32_t>(std::lround(texture.height * scale)));
    constexpr std::size_t header_size = 54u;
    std::vector<std::byte> bmp(header_size + static_cast<std::size_t>(width) * height * 4u);
    bmp[0] = std::byte{ 'B' };
    bmp[1] = std::byte{ 'M' };
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
                rgba = { preview_channel(linear[0]), preview_channel(linear[1]), preview_channel(linear[2]), 255u };
            }
            else
            {
                const auto offset = source_pixel * 4u;
                rgba = {
                    std::to_integer<std::uint8_t>(texture.pixels[offset]),
                    std::to_integer<std::uint8_t>(texture.pixels[offset + 1u]),
                    std::to_integer<std::uint8_t>(texture.pixels[offset + 2u]),
                    std::to_integer<std::uint8_t>(texture.pixels[offset + 3u])
                };
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

std::optional<std::filesystem::path> resolve_project_asset(
    const std::filesystem::path& asset_root,
    const std::filesystem::path& relative_path)
{
    if (asset_root.empty() || relative_path.empty())
        return std::nullopt;
    std::error_code error;
    const auto root = std::filesystem::weakly_canonical(asset_root, error);
    if (error)
        return std::nullopt;
    auto candidate = std::filesystem::path(relative_path);
    if (candidate.is_relative())
        candidate = root / candidate;
    candidate = std::filesystem::weakly_canonical(candidate, error);
    if (error || !std::filesystem::is_regular_file(candidate, error))
        return std::nullopt;
    const auto relative = std::filesystem::relative(candidate, root, error);
    if (error || relative.empty())
        return std::nullopt;
    for (const auto& part : relative)
    {
        if (part == "..")
            return std::nullopt;
    }
    return candidate;
}

std::optional<std::filesystem::path> resolve_project_document(
    const std::filesystem::path& project_root,
    const std::filesystem::path& authored,
    bool must_exist)
{
    if (project_root.empty() || authored.empty())
        return std::nullopt;
    std::error_code error;
    const auto root = std::filesystem::weakly_canonical(project_root, error);
    if (error)
        return std::nullopt;
    auto candidate = authored.is_absolute() ? authored : root / authored;
    candidate = must_exist ? std::filesystem::weakly_canonical(candidate, error) : candidate.lexically_normal();
    if (error || (must_exist && !std::filesystem::is_regular_file(candidate, error)))
        return std::nullopt;
    const auto relative = std::filesystem::relative(candidate, root, error);
    if (error || relative.empty() || relative.is_absolute())
        return std::nullopt;
    for (const auto& part : relative)
        if (part == "..")
            return std::nullopt;
    return candidate;
}

scene::entity to_scene_entity(host_entity_id entity) noexcept
{
    return { .index = entity.index, .generation = entity.generation };
}

host_entity_id to_host_entity(scene::entity entity) noexcept
{
    return { .index = entity.index, .generation = entity.generation };
}

host_vec3 to_host_vec3(const math::vector3f& value) noexcept
{
    return { value[0], value[1], value[2] };
}

math::vector3f to_math_vec3(host_vec3 value) noexcept
{
    return { value.x, value.y, value.z };
}

host_quat to_host_quat(const math::quatf& value) noexcept
{
    return { value[0], value[1], value[2], value[3] };
}

math::quatf to_math_quat(host_quat value) noexcept
{
    return { value.x, value.y, value.z, value.w };
}

host_vec4 to_host_vec4(const math::vector4f& value) noexcept
{
    return { value[0], value[1], value[2], value[3] };
}

math::vector4f to_math_vec4(host_vec4 value) noexcept
{
    return { value.x, value.y, value.z, value.w };
}

host_transform to_host_transform(const scene::transform_component& transform) noexcept
{
    return {
        .position = to_host_vec3(transform.position),
        .rotation = to_host_quat(transform.rotation),
        .scale = to_host_vec3(transform.scale)
    };
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
    return projection == scene::camera_projection::orthographic
        ? host_camera_projection::orthographic
        : host_camera_projection::perspective;
}

host_camera_snapshot to_host_camera(const scene::camera_component& camera) noexcept
{
    return {
        .projection = to_host_projection(camera.projection),
        .fov_y_degrees = math::to_degrees(camera.fov_y_radians),
        .orthographic_height = camera.orthographic_height,
        .near_plane = camera.near_plane,
        .far_plane = camera.far_plane,
        .active = camera.active,
        .clear_color = to_host_vec4(camera.clear_color),
        .exposure_mode = camera.exposure.mode == render::exposure_mode::manual
            ? host_exposure_mode::manual
            : host_exposure_mode::automatic,
        .exposure_metering = camera.exposure.metering == render::exposure_metering_mode::center_weighted
            ? host_exposure_metering_mode::center_weighted
            : host_exposure_metering_mode::average,
        .manual_ev100 = camera.exposure.manual_ev100,
        .exposure_compensation = camera.exposure.compensation_ev,
        .minimum_ev100 = camera.exposure.minimum_ev100,
        .maximum_ev100 = camera.exposure.maximum_ev100,
        .brighten_speed = camera.exposure.brighten_speed,
        .darken_speed = camera.exposure.darken_speed
    };
}

bool valid_camera(const host_camera_snapshot& camera) noexcept
{
    const auto finite = [](float value) { return std::isfinite(value); };
    return finite(camera.fov_y_degrees) && camera.fov_y_degrees > 1.0f && camera.fov_y_degrees < 179.0f &&
        finite(camera.orthographic_height) && camera.orthographic_height > 0.0f &&
        finite(camera.near_plane) && camera.near_plane > 0.0f &&
        finite(camera.far_plane) && camera.far_plane > camera.near_plane &&
        finite(camera.manual_ev100) &&
        finite(camera.exposure_compensation) &&
        finite(camera.minimum_ev100) &&
        finite(camera.maximum_ev100) && camera.maximum_ev100 > camera.minimum_ev100 &&
        finite(camera.brighten_speed) && camera.brighten_speed >= 0.0f &&
        finite(camera.darken_speed) && camera.darken_speed >= 0.0f &&
        finite(camera.clear_color.x) && camera.clear_color.x >= 0.0f && camera.clear_color.x <= 1.0f &&
        finite(camera.clear_color.y) && camera.clear_color.y >= 0.0f && camera.clear_color.y <= 1.0f &&
        finite(camera.clear_color.z) && camera.clear_color.z >= 0.0f && camera.clear_color.z <= 1.0f &&
        finite(camera.clear_color.w) && camera.clear_color.w >= 0.0f && camera.clear_color.w <= 1.0f;
}

bool valid_base_color_tint(host_vec4 tint) noexcept
{
    const auto valid_channel = [](float value) { return std::isfinite(value) && value >= 0.0f && value <= 1.0f; };
    return valid_channel(tint.x) && valid_channel(tint.y) && valid_channel(tint.z) && valid_channel(tint.w);
}

scene::camera_component to_scene_camera(const host_camera_snapshot& camera) noexcept
{
    return {
        .projection = camera.projection == host_camera_projection::orthographic
            ? scene::camera_projection::orthographic
            : scene::camera_projection::perspective,
        .fov_y_radians = math::to_radians(camera.fov_y_degrees),
        .near_plane = camera.near_plane,
        .far_plane = camera.far_plane,
        .orthographic_height = camera.orthographic_height,
        .active = camera.active,
        .clear_color = to_math_vec4(camera.clear_color),
        .exposure = {
            .mode = camera.exposure_mode == host_exposure_mode::manual
                ? render::exposure_mode::manual
                : render::exposure_mode::automatic,
            .metering = camera.exposure_metering == host_exposure_metering_mode::center_weighted
                ? render::exposure_metering_mode::center_weighted
                : render::exposure_metering_mode::average,
            .manual_ev100 = camera.manual_ev100,
            .compensation_ev = camera.exposure_compensation,
            .minimum_ev100 = camera.minimum_ev100,
            .maximum_ev100 = camera.maximum_ev100,
            .brighten_speed = camera.brighten_speed,
            .darken_speed = camera.darken_speed
        }
    };
}

host_light_unit to_host_light_unit(render::light_intensity_unit unit) noexcept
{
    switch (unit)
    {
    case render::light_intensity_unit::lumen: return host_light_unit::lumen;
    case render::light_intensity_unit::candela: return host_light_unit::candela;
    case render::light_intensity_unit::lux: return host_light_unit::lux;
    case render::light_intensity_unit::nit: return host_light_unit::nit;
    case render::light_intensity_unit::unitless: return host_light_unit::unitless;
    }
    return host_light_unit::unitless;
}

render::light_intensity_unit to_render_light_unit(host_light_unit unit) noexcept
{
    switch (unit)
    {
    case host_light_unit::lumen: return render::light_intensity_unit::lumen;
    case host_light_unit::candela: return render::light_intensity_unit::candela;
    case host_light_unit::lux: return render::light_intensity_unit::lux;
    case host_light_unit::nit: return render::light_intensity_unit::nit;
    case host_light_unit::unitless: return render::light_intensity_unit::unitless;
    }
    return render::light_intensity_unit::unitless;
}

host_light_snapshot to_host_light(const scene::directional_light_component& light) noexcept
{
    return {
        .kind = host_light_kind::directional,
        .unit = to_host_light_unit(light.intensity_unit),
        .color = to_host_vec3(light.color),
        .intensity = light.intensity,
        .enabled = light.enabled,
        .casts_shadows = light.casts_shadows,
        .use_color_temperature = light.use_color_temperature,
        .temperature_kelvin = light.temperature_kelvin
    };
}

host_light_snapshot to_host_light(const scene::point_light_component& light) noexcept
{
    return {
        .kind = host_light_kind::point,
        .unit = to_host_light_unit(light.intensity_unit),
        .color = to_host_vec3(light.color),
        .intensity = light.intensity,
        .range = light.range,
        .enabled = light.enabled,
        .casts_shadows = light.casts_shadows,
        .use_color_temperature = light.use_color_temperature,
        .temperature_kelvin = light.temperature_kelvin
    };
}

host_light_snapshot to_host_light(const scene::spot_light_component& light) noexcept
{
    return {
        .kind = host_light_kind::spot,
        .unit = to_host_light_unit(light.intensity_unit),
        .color = to_host_vec3(light.color),
        .intensity = light.intensity,
        .range = light.range,
        .inner_angle_degrees = math::to_degrees(light.inner_angle),
        .outer_angle_degrees = math::to_degrees(light.outer_angle),
        .enabled = light.enabled,
        .casts_shadows = light.casts_shadows,
        .use_color_temperature = light.use_color_temperature,
        .temperature_kelvin = light.temperature_kelvin
    };
}

host_light_snapshot to_host_light(const scene::area_light_component& light) noexcept
{
    return {
        .kind = light.shape == render::area_light_shape::disk
            ? host_light_kind::disk
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
        .temperature_kelvin = light.temperature_kelvin
    };
}

bool valid_light(const host_light_snapshot& light) noexcept
{
    const auto finite_nonnegative = [](float value) {
        return std::isfinite(value) && value >= 0.0f;
    };
    const bool color_valid = finite_nonnegative(light.color.x) &&
        finite_nonnegative(light.color.y) && finite_nonnegative(light.color.z);
    const bool common_valid = color_valid && finite_nonnegative(light.intensity) &&
        std::isfinite(light.temperature_kelvin) &&
        light.temperature_kelvin >= 1000.0f && light.temperature_kelvin <= 40000.0f;
    if (!common_valid)
        return false;
    const bool unit_valid =
        (light.kind == host_light_kind::directional &&
            (light.unit == host_light_unit::lux || light.unit == host_light_unit::unitless)) ||
        ((light.kind == host_light_kind::point || light.kind == host_light_kind::spot) &&
            (light.unit == host_light_unit::lumen || light.unit == host_light_unit::candela ||
                light.unit == host_light_unit::unitless)) ||
        ((light.kind == host_light_kind::rectangle || light.kind == host_light_kind::disk) &&
            (light.unit == host_light_unit::lumen || light.unit == host_light_unit::nit ||
                light.unit == host_light_unit::unitless));
    if (!unit_valid)
        return false;
    if ((light.kind == host_light_kind::point || light.kind == host_light_kind::spot) &&
        (!std::isfinite(light.range) || light.range <= 0.0f))
        return false;
    if (light.kind == host_light_kind::spot &&
        (!std::isfinite(light.inner_angle_degrees) || !std::isfinite(light.outer_angle_degrees) ||
            light.inner_angle_degrees < 0.0f || light.outer_angle_degrees <= light.inner_angle_degrees ||
            light.outer_angle_degrees >= 179.0f))
        return false;
    return (light.kind != host_light_kind::rectangle && light.kind != host_light_kind::disk) ||
        (std::isfinite(light.width) && light.width > 0.0f &&
            std::isfinite(light.height) && light.height > 0.0f);
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
    return projection == host_camera_projection::orthographic
        ? scene::camera_projection::orthographic
        : scene::camera_projection::perspective;
}

render::render_mode to_render_mode(host_render_mode mode) noexcept
{
    return mode == host_render_mode::wireframe ? render::render_mode::wireframe : render::render_mode::shaded;
}

render::mesh_visualization_mode to_visualization(host_visualization_mode mode) noexcept
{
    switch (mode)
    {
    case host_visualization_mode::albedo: return render::mesh_visualization_mode::albedo;
    case host_visualization_mode::opacity: return render::mesh_visualization_mode::opacity;
    case host_visualization_mode::world_normal: return render::mesh_visualization_mode::world_normal;
    case host_visualization_mode::specularity: return render::mesh_visualization_mode::specularity;
    case host_visualization_mode::gloss: return render::mesh_visualization_mode::gloss;
    case host_visualization_mode::metalness: return render::mesh_visualization_mode::metalness;
    case host_visualization_mode::ao: return render::mesh_visualization_mode::ao;
    case host_visualization_mode::emission: return render::mesh_visualization_mode::emission;
    case host_visualization_mode::lighting: return render::mesh_visualization_mode::lighting;
    case host_visualization_mode::uv0: return render::mesh_visualization_mode::uv0;
    case host_visualization_mode::cascade_debug: return render::mesh_visualization_mode::cascade_debug;
    case host_visualization_mode::shadow_mask: return render::mesh_visualization_mode::shadow_mask;
    case host_visualization_mode::light_complexity: return render::mesh_visualization_mode::light_complexity;
    case host_visualization_mode::cluster_debug: return render::mesh_visualization_mode::cluster_debug;
    case host_visualization_mode::standard:
        break;
    }
    return render::mesh_visualization_mode::standard;
}

render::editor_overlay_mode to_overlay(host_overlay_mode mode) noexcept
{
    switch (mode)
    {
    case host_overlay_mode::none: return render::editor_overlay_mode::none;
    case host_overlay_mode::all_wireframe: return render::editor_overlay_mode::all_wireframe;
    case host_overlay_mode::selected_wireframe:
        break;
    }
    return render::editor_overlay_mode::selected_wireframe;
}

editor_tool to_editor_tool(host_viewport_tool tool) noexcept
{
    switch (tool)
    {
    case host_viewport_tool::translate: return editor_tool::translate;
    case host_viewport_tool::rotate: return editor_tool::rotate;
    case host_viewport_tool::scale: return editor_tool::scale;
    case host_viewport_tool::select: return editor_tool::select;
    case host_viewport_tool::terrain: return editor_tool::select;
    }
    return editor_tool::select;
}

scene::scene_render_visibility to_scene_visibility(host_environment_visibility visibility) noexcept
{
    return {
        .sky = visibility.sky,
        .fog = visibility.fog,
        .terrain = visibility.terrain,
        .water = visibility.water,
        .vegetation = visibility.vegetation,
        .decals = visibility.decals
    };
}

scene::world_environment_preset to_scene_preset(host_world_environment_preset preset) noexcept
{
    switch (preset)
    {
    case host_world_environment_preset::clear_day: return scene::world_environment_preset::clear_day;
    case host_world_environment_preset::golden_hour: return scene::world_environment_preset::golden_hour;
    case host_world_environment_preset::overcast: return scene::world_environment_preset::overcast;
    case host_world_environment_preset::night: return scene::world_environment_preset::night;
    case host_world_environment_preset::indoor_neutral: return scene::world_environment_preset::indoor_neutral;
    case host_world_environment_preset::alpine_late_morning: break;
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

void remove_entity_ref(std::vector<scene::entity>& entities, scene::entity entity)
{
    entities.erase(
        std::remove(entities.begin(), entities.end(), entity),
        entities.end());
}

void forget_entity(editor_scene_state& scene, scene::entity entity)
{
    if (scene.camera_entity == entity)
        scene.camera_entity = {};
    if (scene.game_camera_entity == entity)
        scene.game_camera_entity = {};
    if (scene.sun_entity == entity)
        scene.sun_entity = {};
    if (scene.world_environment_entity == entity)
        scene.world_environment_entity = {};
    if (scene.mesh_entity == entity)
        scene.mesh_entity = {};
    if (scene.terrain_entity == entity)
        scene.terrain_entity = {};
    if (scene.water_entity == entity)
        scene.water_entity = {};
    if (scene.vegetation_entity == entity)
        scene.vegetation_entity = {};
    if (scene.selected_entity == entity)
        scene.selected_entity = {};
    remove_entity_ref(scene.primitive_entities, entity);
    remove_entity_ref(scene.imported_scene_entities, entity);
    remove_entity_ref(scene.world_feature_entities, entity);
}

void rebuild_all_terrain_chunks(editor_scene_state& state, render::renderer& renderer)
{
    for (const auto entity : state.scene.entities())
        if (state.scene.has<scene::terrain_component>(entity))
            rebuild_terrain_chunks(state, renderer, entity);
}

std::string entity_name(const editor_scene_state& state, scene::entity entity, const char* fallback)
{
    if (const auto* name = state.scene.try_get<scene::name_component>(entity))
        return name->value;
    return fallback;
}

bool entity_active(const editor_scene_state& state, scene::entity entity)
{
    const auto* active = state.scene.try_get<scene::active_component>(entity);
    return !active || active->active;
}

void push_event(
    std::vector<host_event>& events,
    std::uint64_t& sequence,
    host_event_type type,
    std::string message,
    scene::entity entity = {},
    std::string payload_json = {})
{
    events.push_back({
        .sequence = ++sequence,
        .event_type = type,
        .entity = to_host_entity(entity),
        .message = std::move(message),
        .payload_json = std::move(payload_json)
    });
}

void add_component_snapshot(std::vector<host_component_snapshot>& components, host_component_kind kind, const char* label)
{
    components.push_back({ .kind = kind, .label = label, .editable = true });
}

std::string asset_relative_path(const std::filesystem::path& root, const std::filesystem::path& path)
{
    if (path.empty())
        return {};
    std::error_code error;
    const auto relative = std::filesystem::relative(path, root, error);
    return (error ? path.lexically_normal() : relative.lexically_normal()).generic_string();
}

host_mesh_renderer_snapshot mesh_renderer_snapshot(
    const editor_scene_state& state,
    const editor_asset_state& assets,
    const scene::mesh_renderer_component& mesh_renderer)
{
    host_mesh_renderer_snapshot snapshot;
    snapshot.visible = mesh_renderer.visible;
    snapshot.base_color_tint = to_host_vec4(mesh_renderer.base_color_tint);
    snapshot.has_material = mesh_renderer.material.valid();
    for (const auto& record : state.material_library.materials)
    {
        if (record.material != mesh_renderer.material)
            continue;
        snapshot.asset_backed_material = true;
        snapshot.material_name = record.asset.name;
        snapshot.material_path = asset_relative_path(assets.root, record.path);
        return snapshot;
    }
    if (mesh_renderer.material == state.default_material) snapshot.material_name = "Default Mesh Material";
    else if (mesh_renderer.material == state.primitive_material) snapshot.material_name = "Primitive Material";
    else if (mesh_renderer.material == state.terrain_material) snapshot.material_name = "Terrain Material";
    else if (mesh_renderer.material == state.water_material) snapshot.material_name = "Water Material";
    else if (mesh_renderer.material == state.vegetation_material) snapshot.material_name = "Vegetation Material";
    else if (mesh_renderer.material.valid()) snapshot.material_name = "Embedded Material";
    return snapshot;
}

editor_scene_state create_default_scene(const editor_asset_state& assets, render::renderer& renderer)
{
    editor_scene_state state;

    math::vector3f center{};
    math::vector3f local_min = defaults::fallback_mesh_bounds_min;
    math::vector3f local_max = defaults::fallback_mesh_bounds_max;
    float radius = defaults::fallback_mesh_radius;
    if (assets.default_mesh_loaded && !assets.default_mesh.vertices.empty())
    {
        local_min = math::vector3f{
            assets.default_mesh.vertices[0].position[0],
            assets.default_mesh.vertices[0].position[1],
            assets.default_mesh.vertices[0].position[2]
        };
        local_max = local_min;
        for (const auto& vertex : assets.default_mesh.vertices)
        {
            for (std::size_t axis = 0; axis < 3; ++axis)
            {
                local_min[axis] = std::min(local_min[axis], vertex.position[axis]);
                local_max[axis] = std::max(local_max[axis], vertex.position[axis]);
            }
        }

        center = math::mul(math::add(local_min, local_max), 0.5f);
        const auto span = math::sub(local_max, local_min);
        radius = std::max({ span[0], span[1], span[2], defaults::fallback_mesh_radius }) * 0.5f;
    }

    if (assets.default_mesh_loaded)
    {
        state.default_textures.reserve(assets.default_textures.size());
        for (const auto& texture : assets.default_textures)
            state.default_textures.push_back(renderer.create_texture(texture));

        render::material_desc material;
        material.name = assets.default_mesh.name + " Material";
        if (!assets.default_materials.empty())
        {
            const auto material_index = assets.default_mesh.material_index < assets.default_materials.size()
                ? assets.default_mesh.material_index
                : std::size_t{ 0 };
            const auto& imported = assets.default_materials[material_index];
            material = imported.material;

            const auto assign_texture = [&](std::size_t index, render::texture_handle& handle) {
                if (index != render::material_texture_indices::invalid && index < state.default_textures.size())
                    handle = state.default_textures[index];
            };

            assign_texture(imported.textures.base_color, material.base_color_texture);
            assign_texture(imported.textures.metallic_roughness, material.metallic_roughness_texture);
            assign_texture(imported.textures.normal, material.normal_texture);
            assign_texture(imported.textures.occlusion, material.occlusion_texture);
            assign_texture(imported.textures.emissive, material.emissive_texture);
            assign_texture(imported.textures.clear_coat, material.clear_coat_texture);
            assign_texture(imported.textures.clear_coat_roughness, material.clear_coat_roughness_texture);
            assign_texture(imported.textures.clear_coat_normal, material.clear_coat_normal_texture);
            assign_texture(imported.textures.anisotropy, material.anisotropy_texture);
            assign_texture(imported.textures.thickness, material.thickness_texture);
            assign_texture(imported.textures.transmission, material.transmission_texture);
        }
        state.default_material = renderer.create_material(material);
        state.default_mesh = renderer.create_mesh(assets.default_mesh);
        state.mesh_uploaded = state.default_mesh.valid();
    }

    const auto camera = state.scene.create();
    state.camera_entity = camera;
    scene::transform_component camera_transform;
    camera_transform.position = defaults::default_camera_position;
    state.scene.emplace<scene::name_component>(camera, "Editor Camera");
    state.scene.emplace<scene::tag_component>(camera, "Editor");
    state.scene.emplace<scene::active_component>(camera);
    state.scene.emplace<scene::transform_component>(camera, camera_transform);
    scene::camera_component editor_camera;
    editor_camera.near_plane = 0.1f;
    editor_camera.far_plane = 2000.0f;
    editor_camera.clear_color = math::vector4f{ 0.055f, 0.12f, 0.22f, 1.0f };
    state.scene.emplace<scene::camera_component>(camera, editor_camera);

    const auto game_camera = state.scene.create();
    state.game_camera_entity = game_camera;
    scene::transform_component game_camera_transform;
    game_camera_transform.position = defaults::default_camera_position;
    state.scene.emplace<scene::name_component>(game_camera, "Main Camera");
    state.scene.emplace<scene::tag_component>(game_camera, "Camera");
    state.scene.emplace<scene::active_component>(game_camera);
    state.scene.emplace<scene::transform_component>(game_camera, game_camera_transform);
    scene::camera_component main_camera;
    main_camera.active = false;
    main_camera.near_plane = 0.1f;
    main_camera.far_plane = 2000.0f;
    state.scene.emplace<scene::camera_component>(game_camera, main_camera);

    const auto sun = state.scene.create();
    state.sun_entity = sun;
    scene::transform_component sun_transform;
    sun_transform.rotation = quaternion_from_euler_degrees(defaults::default_sun_rotation_degrees);
    state.scene.emplace<scene::name_component>(sun, "Sun");
    state.scene.emplace<scene::tag_component>(sun, "Light");
    state.scene.emplace<scene::active_component>(sun);
    state.scene.emplace<scene::transform_component>(sun, sun_transform);
    state.scene.emplace<scene::directional_light_component>(
        sun,
        defaults::default_sun_color,
        defaults::default_sun_intensity,
        true);
    auto& sun_light = state.scene.get<scene::directional_light_component>(sun);
    sun_light.intensity_unit = render::light_intensity_unit::lux;
    sun_light.shadow.resolution = defaults::default_sun_shadow_resolution;
    sun_light.shadow.filter = defaults::default_sun_shadow_filter;
    sun_light.shadow.bias = defaults::default_sun_shadow_bias;
    sun_light.shadow.normal_bias = defaults::default_sun_shadow_normal_bias;

    add_world_environment_to_scene(state);

    render::environment_desc environment_lighting;
    environment_lighting.name = "Default Mountain Daylight";
    environment_lighting.fallback_color = math::vector3f{ 0.16f, 0.22f, 0.30f };
    environment_lighting.intensity = 1.1f;
    environment_lighting.diffuse_irradiance = math::vector3f{ 0.18f, 0.23f, 0.29f };
    environment_lighting.diffuse_intensity = 1.0f;
    state.environment_lighting_resource = renderer.create_environment(environment_lighting);
    if (auto* lighting = state.scene.try_get<scene::environment_lighting_component>(state.world_environment_entity))
        lighting->environment = state.environment_lighting_resource;

    const auto terrain_material = create_default_terrain_material(state, renderer, assets.root);
    const auto terrain = add_terrain_to_scene(state, renderer, terrain_material);
    add_water_to_scene(state, renderer);
    add_grass_patch_to_scene(state, renderer);

    if (state.default_mesh.valid())
    {
        const auto* terrain_data = state.scene.try_get<scene::terrain_component>(terrain);
        const auto create_rock = [&](const char* name, float x, float z, float scale_factor, float yaw) {
            const auto mesh = state.scene.create();
            const float scale = (defaults::imported_mesh_fit_size / radius) * scale_factor;
            const float terrain_height = terrain_data != nullptr
                ? scene::sample_terrain_height(*terrain_data, x, z)
                : 0.0f;
            scene::transform_component transform;
            transform.position = math::vector3f{
                x - center[0] * scale,
                terrain_height - local_min[1] * scale,
                z - center[2] * scale
            };
            transform.rotation = quaternion_from_euler_degrees({ 0.0f, yaw, 0.0f });
            transform.scale = math::vector3f{ scale, scale, scale };
            state.scene.emplace<scene::name_component>(mesh, name);
            state.scene.emplace<scene::tag_component>(mesh, "Mesh");
            state.scene.emplace<scene::active_component>(mesh);
            state.scene.emplace<scene::selection_component>(mesh, false);
            state.scene.emplace<scene::bounds_component>(
                mesh,
                geometric::box3f{ geometric::point3f(local_min), geometric::point3f(local_max) },
                geometric::box3f{ geometric::point3f(local_min), geometric::point3f(local_max) },
                true);
            state.scene.emplace<scene::transform_component>(mesh, transform);
            state.scene.emplace<scene::mesh_renderer_component>(mesh, state.default_mesh, state.default_material, true);
            state.world_feature_entities.push_back(mesh);
            return mesh;
        };
        state.mesh_entity = create_rock("Hero Rock Formation", -5.0f, 5.0f, 1.0f, 18.0f);
        create_rock("Rock Formation East", 11.0f, -2.0f, 0.56f, -34.0f);
        create_rock("Rock Formation Shore", -13.0f, -10.0f, 0.42f, 57.0f);
    }

    if (state.scene.alive(terrain))
        select_entity(state.scene, terrain, state.selected_entity);

    ensure_scene_authoring_metadata(state);

    return state;
}

} // namespace

class editor_preview_application final : public application
{
public:
    application_config configure() const override
    {
        application_config config{};
        config.title = "ARC Editor Preview Runtime";
        config.visible = false;
        config.simulation.snapshot_budget_bytes = 64u * 1024u * 1024u;
        config.simulation.presentation_enabled = false;
        return config;
    }

    void register_worlds(runtime_world_manager& worlds) override
    {
        worlds.create({
            .name = "Editor Preview World",
            .role = runtime_world_role::editor_preview,
            .seed = 0x4152435f45444954ull,
            .presentation_enabled = false
        });
    }
};

struct arc_host::state
{
    explicit state(std::unique_ptr<render::renderer> renderer_value)
        : renderer(std::move(renderer_value))
        , simulation(simulation_application)
    {
        simulation.start();
        simulation.pause();
        const auto worlds = simulation.worlds().ordered_worlds();
        if (!worlds.empty())
        {
            if (runtime_world* world = simulation.worlds().find(worlds.front()))
                world->attach_entities(scene.scene);
        }
    }

    std::unique_ptr<render::renderer> renderer;
    editor_preview_application simulation_application;
    runtime simulation;
    std::uint64_t runtime_revision{ 1 };
    std::uint64_t last_runtime_tick_event{};
    bool preview_stopped{ true };
    editor_project_state project;
    editor_asset_state assets;
    editor_scene_state scene;
    editor_camera_controller camera_controller;
    host_viewport_request viewport_options;
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
    std::uint64_t next_pick_request_id{ 1 };
    editor_history history;
    host_viewport_set_tool_command viewport_tool;
    scene::terrain_brush_settings terrain_brush;
    bool terrain_flatten_height_captured{};
    std::optional<math::vector3f> terrain_brush_local_position;
    gizmo_axis gizmo_highlight{ gizmo_axis::none };
    std::vector<host_event> events;
    std::uint64_t event_sequence{};
    bool project_open{};
};

arc_host::arc_host(std::unique_ptr<render::renderer> renderer)
    : state_(std::make_unique<state>(std::move(renderer)))
{
    arc::info("editor.host", "Arc Host started");
    push_event(state_->events, state_->event_sequence, host_event_type::host_started, "Arc Host started");
}

arc_host::~arc_host()
{
    if (state_)
    {
        arc::info("editor.host", "Arc Host shutdown");
        push_event(state_->events, state_->event_sequence, host_event_type::host_shutdown, "Arc Host shutdown");
        state_->simulation.shutdown();
    }
}

host_response arc_host::open_project(
    const host_open_project_command& command,
    const editor_asset_state& assets,
    std::uint64_t request_id)
{
    state_->assets = assets;
    state_->project.name = command.name.empty() ? "Arc Project" : command.name;
    state_->project.root = command.root;
    state_->scene = create_default_scene(assets, *state_->renderer);
    state_->history.clear(state_->scene, false);
    state_->camera_controller = {};
    state_->camera_controller.focus(defaults::default_camera_focus, defaults::default_camera_focus_radius);
    state_->camera_controller.orbit(defaults::default_camera_orbit_x, defaults::default_camera_orbit_y);
    if (auto* camera_transform = state_->scene.scene.try_get<scene::transform_component>(state_->scene.camera_entity))
        state_->camera_controller.apply_to(*camera_transform);
    state_->project_open = true;

    const std::string message = "Opened project '" + state_->project.name + "'";
    arc::info("editor.host", message);
    push_event(state_->events, state_->event_sequence, host_event_type::project_opened, message);
    push_event(state_->events, state_->event_sequence, host_event_type::scene_changed, "Default editor scene loaded");
    return {
        .request_id = request_id,
        .succeeded = true,
        .payload_json = "{\"entity\":" + to_json(to_host_entity(state_->scene.selected_entity)) + '}'
    };
}

host_response arc_host::execute(host_command_payload command)
{
    return execute({ .command_type = command_type(command), .payload = std::move(command) });
}

host_response arc_host::execute(const host_command_envelope& command)
{
    const bool authoring = std::visit([](const auto& payload) {
        return is_authoring_command<std::decay_t<decltype(payload)>>();
    }, command.payload);
    const std::string edit_label = command.edit && !command.edit->label.empty()
        ? command.edit->label : history_label(command.payload);
    if (command.edit && command.edit->phase == host_edit_phase::cancel)
    {
        if (!state_->history.cancel(command.edit->id, state_->scene))
            return { .request_id = command.request_id, .succeeded = false, .error = "No matching edit transaction to cancel" };
        rebuild_all_terrain_chunks(state_->scene, *state_->renderer);
        state_->terrain_brush_local_position.reset();
        push_event(state_->events, state_->event_sequence, host_event_type::scene_changed, "Edit transaction cancelled");
        return { .request_id = command.request_id, .succeeded = true, .payload_json = "{}" };
    }
    if (authoring && command.edit && command.edit->phase == host_edit_phase::begin &&
        !state_->history.begin(command.edit->id, edit_label, state_->scene))
        return { .request_id = command.request_id, .succeeded = false, .error = "Could not begin edit transaction" };
    if (authoring && command.edit && (command.edit->phase == host_edit_phase::update || command.edit->phase == host_edit_phase::commit) &&
        !state_->history.transaction_matches(command.edit->id))
        return { .request_id = command.request_id, .succeeded = false, .error = "Edit transaction does not match the active transaction" };
    std::optional<editor_scene_state> before;
    if (authoring && !command.edit)
        before = state_->scene;

    auto response = std::visit([this, request_id = command.request_id](const auto& payload) -> host_response {
        using command_type = std::decay_t<decltype(payload)>;
        const auto fail = [this, request_id](std::string message, scene::entity entity = {}) {
            arc::warn("editor.host", message);
            push_event(state_->events, state_->event_sequence, host_event_type::command_failed, message, entity);
            return host_response{ .request_id = request_id, .succeeded = false, .error = std::move(message) };
        };
        const auto success = [request_id](std::string payload_json = "{}") {
            return host_response{ .request_id = request_id, .succeeded = true, .payload_json = std::move(payload_json) };
        };

        if constexpr (std::is_same_v<command_type, host_open_project_command>)
        {
            const auto project_assets = payload.root / "assets";
            const auto asset_root = std::filesystem::is_directory(project_assets)
                ? project_assets
                : payload.root;
            return open_project(payload, load_default_editor_assets(asset_root), request_id);
        }
        else if constexpr (std::is_same_v<command_type, host_close_project_command>)
        {
            if (!state_->project_open)
                return success("{\"message\":\"No project is open\"}");

            const std::string message = "Closed project '" + state_->project.name + "'";
            state_->scene = {};
            state_->project = {};
            state_->project_open = false;
            arc::info("editor.host", message);
            push_event(state_->events, state_->event_sequence, host_event_type::project_closed, message);
            return success();
        }
        else if constexpr (std::is_same_v<command_type, host_runtime_resume_command>)
        {
            state_->simulation.resume();
            state_->preview_stopped = false;
            ++state_->runtime_revision;
            const auto snapshot = runtime_snapshot();
            push_event(
                state_->events,
                state_->event_sequence,
                host_event_type::runtime_state_changed,
                "Preview runtime resumed",
                {},
                to_json(snapshot));
            return success(to_json(snapshot));
        }
        else if constexpr (
            std::is_same_v<command_type, host_runtime_pause_command> ||
            std::is_same_v<command_type, host_runtime_stop_command>)
        {
            state_->simulation.pause();
            state_->preview_stopped = std::is_same_v<command_type, host_runtime_stop_command>;
            ++state_->runtime_revision;
            const auto snapshot = runtime_snapshot();
            push_event(
                state_->events,
                state_->event_sequence,
                host_event_type::runtime_state_changed,
                std::is_same_v<command_type, host_runtime_stop_command>
                    ? "Preview runtime stopped"
                    : "Preview runtime paused",
                {},
                to_json(snapshot));
            return success(to_json(snapshot));
        }
        else if constexpr (std::is_same_v<command_type, host_runtime_step_command>)
        {
            if (!state_->simulation.paused())
                state_->simulation.pause();
            state_->preview_stopped = false;
            if (!state_->simulation.step(payload.ticks))
                return fail("Preview runtime could not queue a fixed-step");
            const frame_time stepped = state_->simulation.advance(0.0);
            ++state_->runtime_revision;
            const auto snapshot = runtime_snapshot();
            push_event(
                state_->events,
                state_->event_sequence,
                host_event_type::runtime_tick_completed,
                "Preview runtime stepped " + std::to_string(stepped.completed_ticks) + " tick(s)",
                {},
                to_json(snapshot));
            return success(to_json(snapshot));
        }
        else if constexpr (std::is_same_v<command_type, host_runtime_set_time_scale_command>)
        {
            if (!state_->simulation.set_time_scale(payload.value))
                return fail("Runtime time scale must be finite and between 0 and 16");
            ++state_->runtime_revision;
            const auto snapshot = runtime_snapshot();
            push_event(
                state_->events,
                state_->event_sequence,
                host_event_type::runtime_state_changed,
                "Preview runtime time scale changed",
                {},
                to_json(snapshot));
            return success(to_json(snapshot));
        }
        else if constexpr (std::is_same_v<command_type, host_runtime_capture_snapshot_command>)
        {
            const auto worlds = state_->simulation.worlds().ordered_worlds();
            if (worlds.empty())
                return fail("Preview runtime has no world to snapshot");
            const world_snapshot_result captured =
                state_->simulation.capture_snapshot(worlds.front(), payload.label);
            if (!captured.succeeded)
                return fail(captured.error);
            return success(
                "{\"snapshotId\":" + std::to_string(captured.metadata.id.value) +
                ",\"tickId\":" + std::to_string(captured.metadata.tick.value) + '}');
        }
        else if constexpr (std::is_same_v<command_type, host_runtime_restore_snapshot_command>)
        {
            const world_snapshot_result restored =
                state_->simulation.restore_snapshot({ payload.snapshot_id });
            if (!restored.succeeded)
                return fail(restored.error);
            if (state_->scene.selected_entity.valid() &&
                !state_->scene.scene.alive(state_->scene.selected_entity))
                clear_selection(state_->scene.scene, state_->scene.selected_entity);
            rebuild_all_terrain_chunks(state_->scene, *state_->renderer);
            if (auto* camera_transform = state_->scene.scene.try_get<scene::transform_component>(state_->scene.camera_entity))
                state_->camera_controller.synchronize_from(*camera_transform);
            ++state_->runtime_revision;
            const auto snapshot = runtime_snapshot();
            push_event(
                state_->events,
                state_->event_sequence,
                host_event_type::scene_changed,
                "Preview runtime snapshot restored",
                state_->scene.selected_entity);
            push_event(
                state_->events,
                state_->event_sequence,
                host_event_type::runtime_state_changed,
                "Preview runtime snapshot restored",
                {},
                to_json(snapshot));
            return success(to_json(snapshot));
        }
        else if constexpr (std::is_same_v<command_type, host_open_scene_command>)
        {
            if (!state_->project_open)
                return fail("Cannot open a scene before a project is open");

            if (payload.path.extension() == ".arcscene")
            {
                if (payload.append)
                    return fail("Appending native ARC scene documents is not supported");
                const auto path = payload.path.is_absolute() ? payload.path : state_->project.root / payload.path;
                const auto loaded = load_scene_document(state_->scene, *state_->renderer, state_->project.root, path);
                if (!loaded.succeeded)
                    return fail(loaded.message);
                state_->history.clear(state_->scene, true);
                if (auto* camera_transform = state_->scene.scene.try_get<scene::transform_component>(state_->scene.camera_entity))
                    state_->camera_controller.synchronize_from(*camera_transform);
                push_event(state_->events, state_->event_sequence, host_event_type::scene_changed, "ARC scene loaded", state_->scene.selected_entity);
                return success("{\"entityCount\":" + std::to_string(loaded.entity_count) + '}');
            }
            const auto mode = payload.append ? editor_scene_open_mode::append : editor_scene_open_mode::replace;
            std::optional<editor_scene_state> import_before;
            if (payload.append)
                import_before = state_->scene;
            const auto asset_root = payload.path.is_absolute() ? payload.path.parent_path() : state_->assets.root;
            const auto result = open_scene_asset_in_editor(
                state_->scene,
                *state_->renderer,
                asset_root,
                payload.path,
                mode);
            if (!result.succeeded)
                return fail(result.message.empty() ? "Failed to open scene asset" : result.message);

            ensure_scene_authoring_metadata(state_->scene);
            if (auto* camera_transform = state_->scene.scene.try_get<scene::transform_component>(state_->scene.camera_entity))
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
            push_event(state_->events, state_->event_sequence, host_event_type::scene_changed, message, state_->scene.selected_entity);
            return success("{\"entityCount\":" + std::to_string(result.entity_count) + '}');
        }
        else if constexpr (std::is_same_v<command_type, host_new_scene_command>)
        {
            state_->scene = create_default_scene(state_->assets, *state_->renderer);
            state_->scene.scene_name = payload.name.empty() ? "Untitled" : payload.name;
            state_->scene.active_scene_path.clear();
            state_->history.clear(state_->scene, false);
            if (auto* camera_transform = state_->scene.scene.try_get<scene::transform_component>(state_->scene.camera_entity))
                state_->camera_controller.synchronize_from(*camera_transform);
            push_event(state_->events, state_->event_sequence, host_event_type::scene_changed, "New scene created", state_->scene.selected_entity);
            return success();
        }
        else if constexpr (std::is_same_v<command_type, host_save_scene_command> || std::is_same_v<command_type, host_save_scene_as_command>)
        {
            std::filesystem::path path;
            if constexpr (std::is_same_v<command_type, host_save_scene_as_command>) path = payload.path;
            else path = state_->scene.active_scene_path;
            if (path.empty()) return fail("Scene has no path; use scene.saveAs first");
            if (path.is_relative()) path = state_->project.root / path;
            if (path.extension() != ".arcscene") path.replace_extension(".arcscene");
            const auto saved = save_scene_document(state_->scene, state_->project.root, path);
            if (!saved.succeeded) return fail(saved.message);
            state_->history.mark_saved();
            push_event(state_->events, state_->event_sequence, host_event_type::scene_changed, "Scene saved", state_->scene.selected_entity);
            return success("{\"path\":" + to_json_string(path.generic_string()) + '}');
        }
        else if constexpr (std::is_same_v<command_type, host_create_entity_command>)
        {
            const auto requested_parent = to_scene_entity(payload.parent);
            if (payload.parent.valid() && !state_->scene.scene.alive(requested_parent))
                return fail("Cannot create a child under a missing parent", requested_parent);
            scene::entity created{};
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
                created = add_primitive_to_scene(state_->scene, *state_->renderer, primitive_type_for(payload.kind));
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
            if (!created.valid())
                return fail("Failed to create entity: " + label);

            if (payload.parent.valid())
            {
                if (!scene::reparent(state_->scene.scene, created, requested_parent, {}, scene::reparent_transform_policy::preserve_local))
                {
                    scene::destroy_subtree(state_->scene.scene, created);
                    return fail("Cannot create a child under an invalid parent", requested_parent);
                }
            }

            const std::string message = "Created entity: " + label;
            arc::info("editor.host", message);
            push_event(state_->events, state_->event_sequence, host_event_type::entity_created, message, created);
            push_event(state_->events, state_->event_sequence, host_event_type::entity_selected, "Selected entity", created);
            push_event(state_->events, state_->event_sequence, host_event_type::scene_changed, "Scene changed", created);
            return success("{\"entity\":" + to_json(to_host_entity(created)) + '}');
        }
        else if constexpr (std::is_same_v<command_type, host_delete_entity_command>)
        {
            const auto entity = to_scene_entity(payload.entity);
            if (!state_->scene.scene.alive(entity))
                return fail("Cannot delete a missing entity", entity);

            const std::string name = entity_name(state_->scene, entity, "Entity");
            const auto removed = scene::subtree(state_->scene.scene, entity);
            scene::destroy_subtree(state_->scene.scene, entity);
            for (const auto nested : removed) forget_entity(state_->scene, nested);
            const std::string message = "Deleted entity: " + name;
            arc::info("editor.host", message);
            push_event(state_->events, state_->event_sequence, host_event_type::entity_deleted, message, entity);
            push_event(state_->events, state_->event_sequence, host_event_type::scene_changed, "Scene changed", entity);
            return success("{\"entity\":" + to_json(payload.entity) + '}');
        }
        else if constexpr (std::is_same_v<command_type, host_duplicate_entity_command>)
        {
            const auto source = to_scene_entity(payload.entity);
            if (!state_->scene.scene.alive(source)) return fail("Cannot duplicate a missing entity", source);
            const auto* links = state_->scene.scene.try_get<scene::hierarchy_component>(source);
            const auto duplicate = duplicate_entity_subtree(state_->scene, source, links ? links->parent : scene::entity{});
            if (!duplicate.valid()) return fail("Entity duplication failed", source);
            select_entity(state_->scene.scene, duplicate, state_->scene.selected_entity);
            push_event(state_->events, state_->event_sequence, host_event_type::entity_created, "Entity duplicated", duplicate);
            push_event(state_->events, state_->event_sequence, host_event_type::scene_changed, "Scene changed", duplicate);
            return success("{\"entity\":" + to_json(to_host_entity(duplicate)) + '}');
        }
        else if constexpr (std::is_same_v<command_type, host_create_prefab_command>)
        {
            const auto entity = to_scene_entity(payload.entity);
            if (!state_->scene.scene.alive(entity))
                return fail("Cannot create a prefab from a missing entity", entity);
            auto path = payload.path;
            if (path.extension() != ".arcprefab")
                path.replace_extension(".arcprefab");
            const auto resolved = resolve_project_document(state_->project.root, path, false);
            if (!resolved)
                return fail("Prefab path must be inside the active project", entity);
            const auto result = save_prefab_document(state_->scene, state_->project.root, entity, *resolved);
            if (!result.succeeded)
                return fail(result.message, entity);
            push_event(state_->events, state_->event_sequence, host_event_type::component_changed, "Prefab created", entity);
            push_event(state_->events, state_->event_sequence, host_event_type::scene_changed, "Scene changed", entity);
            return success("{\"entity\":" + to_json(payload.entity) +
                ",\"path\":" + to_json_string(std::filesystem::relative(*resolved, state_->project.root).generic_string()) + '}');
        }
        else if constexpr (std::is_same_v<command_type, host_instantiate_prefab_command>)
        {
            const auto parent = to_scene_entity(payload.parent);
            if (payload.parent.valid() && !state_->scene.scene.alive(parent))
                return fail("Cannot instantiate a prefab below a missing parent", parent);
            const auto resolved = resolve_project_document(state_->project.root, payload.path, true);
            if (!resolved || resolved->extension() != ".arcprefab")
                return fail("Prefab source is missing or outside the active project");
            const auto result = instantiate_prefab_document(
                state_->scene, *state_->renderer, state_->project.root, *resolved, parent);
            if (!result.succeeded)
                return fail(result.message);
            select_entity(state_->scene.scene, result.root, state_->scene.selected_entity);
            push_event(state_->events, state_->event_sequence, host_event_type::entity_created, "Prefab instantiated", result.root);
            push_event(state_->events, state_->event_sequence, host_event_type::entity_selected, "Selected prefab instance", result.root);
            push_event(state_->events, state_->event_sequence, host_event_type::scene_changed, "Scene changed", result.root);
            return success("{\"entity\":" + to_json(to_host_entity(result.root)) +
                ",\"entityCount\":" + std::to_string(result.entity_count) + '}');
        }
        else if constexpr (std::is_same_v<command_type, host_apply_prefab_command>)
        {
            const auto entity = to_scene_entity(payload.entity);
            if (!state_->scene.scene.alive(entity))
                return fail("Cannot apply a missing prefab instance", entity);
            const auto result = apply_prefab_instance(state_->scene, state_->project.root, entity);
            if (!result.succeeded)
                return fail(result.message, entity);
            push_event(state_->events, state_->event_sequence, host_event_type::component_changed, "Prefab changes applied", entity);
            push_event(state_->events, state_->event_sequence, host_event_type::scene_changed, "Scene changed", entity);
            return success();
        }
        else if constexpr (std::is_same_v<command_type, host_revert_prefab_command>)
        {
            const auto entity = to_scene_entity(payload.entity);
            if (!state_->scene.scene.alive(entity))
                return fail("Cannot revert a missing prefab instance", entity);
            const auto result = revert_prefab_instance(
                state_->scene, *state_->renderer, state_->project.root, entity);
            if (!result.succeeded)
                return fail(result.message, entity);
            select_entity(state_->scene.scene, result.root, state_->scene.selected_entity);
            push_event(state_->events, state_->event_sequence, host_event_type::entity_deleted, "Prefab instance replaced", entity);
            push_event(state_->events, state_->event_sequence, host_event_type::entity_created, "Prefab instance reverted", result.root);
            push_event(state_->events, state_->event_sequence, host_event_type::entity_selected, "Selected prefab instance", result.root);
            push_event(state_->events, state_->event_sequence, host_event_type::scene_changed, "Scene changed", result.root);
            return success("{\"entity\":" + to_json(to_host_entity(result.root)) + '}');
        }
        else if constexpr (std::is_same_v<command_type, host_unpack_prefab_command>)
        {
            const auto entity = to_scene_entity(payload.entity);
            if (!state_->scene.scene.alive(entity))
                return fail("Cannot unpack a missing prefab instance", entity);
            if (!unpack_prefab_instance(state_->scene, entity))
                return fail("Entity is not a prefab instance", entity);
            push_event(state_->events, state_->event_sequence, host_event_type::component_changed, "Prefab instance unpacked", entity);
            push_event(state_->events, state_->event_sequence, host_event_type::scene_changed, "Scene changed", entity);
            return success();
        }
        else if constexpr (std::is_same_v<command_type, host_reparent_entity_command>)
        {
            const auto entity = to_scene_entity(payload.entity);
            const auto parent = to_scene_entity(payload.parent);
            const auto before = to_scene_entity(payload.before_sibling);
            if (!scene::reparent(state_->scene.scene, entity, parent, before, payload.preserve_world
                    ? scene::reparent_transform_policy::preserve_world : scene::reparent_transform_policy::preserve_local))
                return fail("Invalid hierarchy reparent operation", entity);
            push_event(state_->events, state_->event_sequence, host_event_type::scene_changed, "Entity reparented", entity);
            return success();
        }
        else if constexpr (std::is_same_v<command_type, host_reorder_entity_command>)
        {
            const auto entity = to_scene_entity(payload.entity);
            if (!scene::reorder(state_->scene.scene, entity, to_scene_entity(payload.before_sibling)))
                return fail("Invalid hierarchy reorder operation", entity);
            push_event(state_->events, state_->event_sequence, host_event_type::scene_changed, "Entity reordered", entity);
            return success();
        }
        else if constexpr (std::is_same_v<command_type, host_rename_entity_command>)
        {
            const auto entity = to_scene_entity(payload.entity);
            if (!state_->scene.scene.alive(entity))
                return fail("Cannot rename a missing entity", entity);

            state_->scene.scene.emplace<scene::name_component>(entity, payload.name);
            const std::string message = "Renamed entity to '" + payload.name + "'";
            arc::info("editor.host", message);
            push_event(state_->events, state_->event_sequence, host_event_type::component_changed, message, entity);
            return success("{\"entity\":" + to_json(payload.entity) + '}');
        }
        else if constexpr (std::is_same_v<command_type, host_select_entity_command>)
        {
            const auto entity = to_scene_entity(payload.entity);
            if (entity == state_->scene.selected_entity && state_->scene.scene.alive(entity))
                return success("{\"entity\":" + to_json(payload.entity) + '}');
            if (!select_entity(state_->scene.scene, entity, state_->scene.selected_entity))
                return fail("Cannot select a missing entity", entity);

            const std::string message = "Selected entity: " + entity_name(state_->scene, entity, "Entity");
            push_event(state_->events, state_->event_sequence, host_event_type::entity_selected, message, entity);
            return success("{\"entity\":" + to_json(payload.entity) + '}');
        }
        else if constexpr (std::is_same_v<command_type, host_clear_selection_command>)
        {
            if (!state_->scene.scene.alive(state_->scene.selected_entity))
                return success();
            clear_selection(state_->scene.scene, state_->scene.selected_entity);
            push_event(state_->events, state_->event_sequence, host_event_type::entity_selected, "Cleared entity selection");
            return success();
        }
        else if constexpr (std::is_same_v<command_type, host_set_active_command>)
        {
            const auto entity = to_scene_entity(payload.entity);
            if (!state_->scene.scene.alive(entity))
                return fail("Cannot edit a missing entity", entity);

            state_->scene.scene.emplace<scene::active_component>(entity, payload.active);
            push_event(state_->events, state_->event_sequence, host_event_type::component_changed, "Entity active state changed", entity);
            return success("{\"entity\":" + to_json(payload.entity) + '}');
        }
        else if constexpr (std::is_same_v<command_type, host_set_tag_command>)
        {
            const auto entity = to_scene_entity(payload.entity);
            if (!state_->scene.scene.alive(entity))
                return fail("Cannot edit a missing entity", entity);

            state_->scene.scene.emplace<scene::tag_component>(entity, payload.tag);
            push_event(state_->events, state_->event_sequence, host_event_type::component_changed, "Entity tag changed", entity);
            return success("{\"entity\":" + to_json(payload.entity) + '}');
        }
        else if constexpr (std::is_same_v<command_type, host_set_transform_command>)
        {
            const auto entity = to_scene_entity(payload.entity);
            if (!state_->scene.scene.alive(entity))
                return fail("Cannot edit a missing entity", entity);

            state_->scene.scene.emplace<scene::transform_component>(entity, to_scene_transform(payload.transform));
            scene::mark_transform_subtree_dirty(state_->scene.scene, entity);
            scene::update_world_transforms(state_->scene.scene);
            if (entity == state_->scene.camera_entity)
                state_->camera_controller.synchronize_from(state_->scene.scene.get<scene::transform_component>(entity));
            push_event(state_->events, state_->event_sequence, host_event_type::component_changed, "Entity transform changed", entity);
            return success("{\"entity\":" + to_json(payload.entity) + '}');
        }
        else if constexpr (std::is_same_v<command_type, host_set_render_layer_command>)
        {
            const auto entity = to_scene_entity(payload.entity);
            if (!state_->scene.scene.alive(entity))
                return fail("Cannot edit a missing entity", entity);
            if (payload.render_layer_mask == 0u)
                return fail("Render layer mask must contain at least one layer", entity);

            state_->scene.scene.emplace<scene::render_layer_component>(entity, payload.render_layer_mask);
            push_event(state_->events, state_->event_sequence, host_event_type::component_changed, "Entity render layer changed", entity);
            return success("{\"entity\":" + to_json(payload.entity) + '}');
        }
        else if constexpr (std::is_same_v<command_type, host_set_camera_command>)
        {
            const auto entity = to_scene_entity(payload.entity);
            auto* camera = state_->scene.scene.try_get<scene::camera_component>(entity);
            if (!camera)
                return fail("Entity does not have an editable camera component", entity);
            if (!valid_camera(payload.camera))
                return fail("Camera values are outside their valid authored ranges", entity);

            *camera = to_scene_camera(payload.camera);
            push_event(state_->events, state_->event_sequence, host_event_type::component_changed, "Entity camera changed", entity);
            return success("{\"entity\":" + to_json(payload.entity) + '}');
        }
        else if constexpr (std::is_same_v<command_type, host_set_light_command>)
        {
            const auto entity = to_scene_entity(payload.entity);
            if (!state_->scene.scene.alive(entity))
                return fail("Cannot edit a missing light entity", entity);
            if (!valid_light(payload.light))
                return fail("Light values are outside their valid authored ranges", entity);

            const auto unit = to_render_light_unit(payload.light.unit);
            const auto color = to_math_vec3(payload.light.color);
            if (auto* light = state_->scene.scene.try_get<scene::directional_light_component>(entity))
            {
                if (payload.light.kind != host_light_kind::directional)
                    return fail("Light kind does not match the directional light component", entity);
                light->color = color;
                light->intensity = payload.light.intensity;
                light->intensity_unit = unit;
                light->enabled = payload.light.enabled;
                light->casts_shadows = payload.light.casts_shadows;
                light->use_color_temperature = payload.light.use_color_temperature;
                light->temperature_kelvin = payload.light.temperature_kelvin;
                light->shadow.enabled = payload.light.casts_shadows;
            }
            else if (auto* light = state_->scene.scene.try_get<scene::point_light_component>(entity))
            {
                if (payload.light.kind != host_light_kind::point)
                    return fail("Light kind does not match the point light component", entity);
                light->color = color;
                light->intensity = payload.light.intensity;
                light->range = payload.light.range;
                light->intensity_unit = unit;
                light->enabled = payload.light.enabled;
                light->casts_shadows = payload.light.casts_shadows;
                light->use_color_temperature = payload.light.use_color_temperature;
                light->temperature_kelvin = payload.light.temperature_kelvin;
                light->shadow.enabled = payload.light.casts_shadows;
            }
            else if (auto* light = state_->scene.scene.try_get<scene::spot_light_component>(entity))
            {
                if (payload.light.kind != host_light_kind::spot)
                    return fail("Light kind does not match the spot light component", entity);
                light->color = color;
                light->intensity = payload.light.intensity;
                light->range = payload.light.range;
                light->inner_angle = math::to_radians(payload.light.inner_angle_degrees);
                light->outer_angle = math::to_radians(payload.light.outer_angle_degrees);
                light->intensity_unit = unit;
                light->enabled = payload.light.enabled;
                light->casts_shadows = payload.light.casts_shadows;
                light->use_color_temperature = payload.light.use_color_temperature;
                light->temperature_kelvin = payload.light.temperature_kelvin;
                light->shadow.enabled = payload.light.casts_shadows;
            }
            else if (auto* light = state_->scene.scene.try_get<scene::area_light_component>(entity))
            {
                if (payload.light.kind != host_light_kind::rectangle && payload.light.kind != host_light_kind::disk)
                    return fail("Light kind does not match the area light component", entity);
                light->shape = payload.light.kind == host_light_kind::disk
                    ? render::area_light_shape::disk
                    : render::area_light_shape::rectangle;
                light->color = color;
                light->intensity = payload.light.intensity;
                light->width = payload.light.width;
                light->height = payload.light.height;
                light->two_sided = payload.light.two_sided;
                light->intensity_unit = unit;
                light->enabled = payload.light.enabled;
                light->casts_shadows = payload.light.casts_shadows;
                light->use_color_temperature = payload.light.use_color_temperature;
                light->temperature_kelvin = payload.light.temperature_kelvin;
                light->shadow.enabled = payload.light.casts_shadows;
            }
            else
                return fail("Entity does not have an editable light component", entity);

            push_event(state_->events, state_->event_sequence, host_event_type::component_changed, "Entity light changed", entity);
            return success("{\"entity\":" + to_json(payload.entity) + '}');
        }
        else if constexpr (std::is_same_v<command_type, host_set_mesh_renderer_command>)
        {
            const auto entity = to_scene_entity(payload.entity);
            auto* mesh_renderer = state_->scene.scene.try_get<scene::mesh_renderer_component>(entity);
            if (!mesh_renderer)
                return fail("Entity does not have an editable mesh renderer component", entity);
            if (!valid_base_color_tint(payload.base_color_tint))
                return fail("Mesh renderer tint channels must be finite and between 0 and 1", entity);
            mesh_renderer->visible = payload.visible;
            mesh_renderer->base_color_tint = to_math_vec4(payload.base_color_tint);
            push_event(state_->events, state_->event_sequence, host_event_type::component_changed, "Entity mesh renderer changed", entity);
            return success("{\"entity\":" + to_json(payload.entity) + '}');
        }
        else if constexpr (std::is_same_v<command_type, host_set_terrain_command>)
        {
            const auto entity = to_scene_entity(payload.entity);
            auto* terrain = state_->scene.scene.try_get<scene::terrain_component>(entity);
            if (!terrain)
                return fail("Entity does not have an editable terrain component", entity);
            terrain->enabled = payload.enabled;
            terrain->receive_shadows = payload.receive_shadows;
            ++terrain->content_revision;
            push_event(state_->events, state_->event_sequence, host_event_type::component_changed, "Terrain settings changed", entity);
            return success("{\"entity\":" + to_json(payload.entity) + '}');
        }
        else if constexpr (std::is_same_v<command_type, host_set_terrain_brush_command>)
        {
            const auto entity = to_scene_entity(payload.entity);
            if (!state_->scene.scene.has<scene::terrain_component>(entity))
                return fail("Terrain brush requires a terrain entity", entity);
            if (!std::isfinite(payload.radius) || !std::isfinite(payload.strength) || !std::isfinite(payload.falloff) ||
                payload.radius < 0.25f || payload.radius > 128.0f || payload.strength <= 0.0f || payload.strength > 1.0f ||
                payload.falloff < 0.0f || payload.falloff > 1.0f || payload.active_layer >= 4u)
                return fail("Terrain brush values are outside supported ranges", entity);
            state_->terrain_brush.tool = payload.tool == host_terrain_brush_tool::smooth ? scene::terrain_brush_tool::smooth :
                payload.tool == host_terrain_brush_tool::flatten ? scene::terrain_brush_tool::flatten :
                payload.tool == host_terrain_brush_tool::paint ? scene::terrain_brush_tool::paint : scene::terrain_brush_tool::sculpt;
            state_->terrain_brush.radius = payload.radius;
            state_->terrain_brush.strength = payload.strength;
            state_->terrain_brush.falloff = payload.falloff;
            state_->terrain_brush.active_layer = payload.active_layer;
            push_event(state_->events, state_->event_sequence, host_event_type::terrain_tool_changed,
                "Terrain brush changed", entity, to_json(terrain_tool_snapshot()));
            return success(to_json(terrain_tool_snapshot()));
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
                if (!texture.valid())
                    return fail("Terrain layer texture could not be uploaded", entity);
                state_->scene.default_textures.push_back(texture);
                resolved_path = *path;
            }
            state_->scene.terrain_material_desc.domain = render::material_domain::terrain;
            state_->scene.terrain_material_desc.terrain_layers[payload.layer].base_color_texture = texture;
            if (!state_->renderer->update_material(terrain->material, state_->scene.terrain_material_desc))
                return fail("Terrain material could not be updated", entity);
            state_->scene.terrain_layer_paths[payload.layer] = std::move(resolved_path);
            ++terrain->content_revision;
            push_event(state_->events, state_->event_sequence, host_event_type::component_changed, "Terrain layer assigned", entity);
            return success("{\"entity\":" + to_json(payload.entity) + '}');
        }
        else if constexpr (std::is_same_v<command_type, host_terrain_stroke_command>)
        {
            const auto entity = to_scene_entity(payload.entity);
            auto* terrain = state_->scene.scene.try_get<scene::terrain_component>(entity);
            auto* terrain_transform = state_->scene.scene.try_get<scene::transform_component>(entity);
            const auto* camera = state_->scene.scene.try_get<scene::camera_component>(state_->scene.camera_entity);
            const auto* camera_transform = state_->scene.scene.try_get<scene::transform_component>(state_->scene.camera_entity);
            if (!terrain || !terrain_transform || !camera || !camera_transform)
                return fail("Terrain stroke is missing terrain or viewport camera data", entity);
            if (state_->viewport_tool.tool != host_viewport_tool::terrain)
                return fail("Terrain strokes require Terrain viewport mode", entity);
            editor_viewport viewport;
            viewport.set_size(static_cast<float>(state_->viewport_options.width), static_cast<float>(state_->viewport_options.height));
            if (!viewport.valid())
                return fail("Terrain stroke requires a valid viewport", entity);
            scene::update_world_transforms(state_->scene.scene);
            const auto ray = screen_ray_from_camera(*camera, *camera_transform, viewport,
                static_cast<float>(payload.x), static_cast<float>(payload.y));
            math::matrix4f inverse_terrain{};
            const auto terrain_world = terrain_transform->dirty ? scene::local_matrix(*terrain_transform) : terrain_transform->world;
            if (!scene::inverse_affine(terrain_world, inverse_terrain))
                return fail("Terrain transform cannot be inverted", entity);
            const auto local_origin = math::transform_point(inverse_terrain, ray.origin);
            const auto local_direction = math::normalize(math::transform_vector(inverse_terrain, ray.direction));
            const auto hit = scene::raycast_terrain(*terrain, local_origin, local_direction);
            if (!hit.hit)
            {
                state_->terrain_brush_local_position.reset();
                return success("{\"hit\":false}");
            }
            state_->terrain_brush_local_position = hit.position;
            if (payload.phase == host_edit_phase::commit)
            {
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
            }
            const auto dirty = scene::apply_terrain_brush(*terrain, hit.position, state_->terrain_brush);
            if (dirty.valid && !rebuild_terrain_chunks(state_->scene, *state_->renderer, entity, &dirty))
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
            const auto* camera_transform = state_->scene.scene.try_get<scene::transform_component>(state_->scene.camera_entity);
            if (!terrain || !terrain_transform || !camera || !camera_transform)
                return fail("Terrain hover is missing terrain or viewport camera data", entity);
            editor_viewport viewport;
            viewport.set_size(static_cast<float>(state_->viewport_options.width), static_cast<float>(state_->viewport_options.height));
            if (!viewport.valid())
                return fail("Terrain hover requires a valid viewport", entity);
            scene::update_world_transforms(state_->scene.scene);
            const auto ray = screen_ray_from_camera(
                *camera, *camera_transform, viewport, static_cast<float>(payload.x), static_cast<float>(payload.y));
            math::matrix4f inverse_terrain;
            const auto terrain_world = terrain_transform->dirty ? scene::local_matrix(*terrain_transform) : terrain_transform->world;
            if (!scene::inverse_affine(terrain_world, inverse_terrain))
                return fail("Terrain transform cannot be inverted", entity);
            const auto hit = scene::raycast_terrain(
                *terrain,
                math::transform_point(inverse_terrain, ray.origin),
                math::normalize(math::transform_vector(inverse_terrain, ray.direction)));
            if (hit.hit)
                state_->terrain_brush_local_position = hit.position;
            else
                state_->terrain_brush_local_position.reset();
            return success(to_json(terrain_tool_snapshot()));
        }
        else if constexpr (std::is_same_v<command_type, host_set_entity_material_command>)
        {
            const auto entity = to_scene_entity(payload.entity);
            if (!state_->scene.scene.has<scene::mesh_renderer_component>(entity))
                return fail("Entity does not have an editable mesh renderer component", entity);
            const auto path = resolve_project_asset(state_->assets.root, payload.path);
            if (!path || !is_material_asset_path(*path))
                return fail("Material must be an .arcmat project asset", entity);
            std::string message;
            if (!apply_material_asset_to_entity(
                    state_->scene.material_library,
                    *state_->renderer,
                    state_->assets.root,
                    *path,
                    state_->scene.scene,
                    entity,
                    &message))
                return fail(message.empty() ? "Material assignment failed" : message, entity);
            ensure_scene_authoring_metadata(state_->scene);
            const auto guid = entity_guid_of(state_->scene, entity);
            auto* binding = find_asset_binding(state_->scene, guid);
            if (!binding)
            {
                state_->scene.asset_bindings.push_back({ .entity = guid });
                binding = &state_->scene.asset_bindings.back();
            }
            binding->material_path = *path;
            push_event(state_->events, state_->event_sequence, host_event_type::component_changed, message, entity);
            return success("{\"entity\":" + to_json(payload.entity) + '}');
        }
        else if constexpr (std::is_same_v<command_type, host_set_world_environment_command>)
        {
            const auto entity = to_scene_entity(payload.environment.entity);
            auto settings = scene::read_world_environment_settings(state_->scene.scene, entity);
            if (!settings)
                return fail("Cannot edit a missing or incomplete world environment", entity);

            const auto next_settings = apply_host_world_environment_snapshot(payload.environment, *settings);
            const auto validation = scene::validate_world_environment(next_settings);
            if (!validation.valid)
                return fail(validation_message(validation), entity);
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
            if (!settings)
                return fail("Cannot apply a preset to a missing world environment", entity);

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
            if (!world || !lighting)
                return fail("Cannot assign an HDRI to a missing world environment", entity);

            const auto update_environment_resource = [&] {
                if (!lighting->environment.valid())
                    return;
                render::environment_desc environment;
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
            state_->scene.world_environment_hdri_path = std::filesystem::relative(*path, state_->assets.root, relative_error);
            if (relative_error)
                state_->scene.world_environment_hdri_path = path->filename();
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
                push_event(state_->events, state_->event_sequence, host_event_type::component_changed, "Camera projection changed", state_->scene.camera_entity);
                return success("{\"entity\":" + to_json(to_host_entity(state_->scene.camera_entity)) + '}');
            }
            return fail("No editor camera is available", state_->scene.camera_entity);
        }
        else if constexpr (std::is_same_v<command_type, host_viewport_attach_command>)
        {
            state_->viewport_options.width = payload.width;
            state_->viewport_options.height = payload.height;
            return success("{\"nativeHandle\":" + std::to_string(payload.native_handle) + '}');
        }
        else if constexpr (std::is_same_v<command_type, host_viewport_resize_command>)
        {
            state_->viewport_options.width = payload.width;
            state_->viewport_options.height = payload.height;
            return success();
        }
        else if constexpr (std::is_same_v<command_type, host_viewport_set_camera_mode_command>)
        {
            return execute(host_command_envelope{
                .request_id = request_id,
                .command_type = "camera.setProjection",
                .payload = host_set_camera_projection_command{ .projection = payload.projection } });
        }
        else if constexpr (std::is_same_v<command_type, host_viewport_set_render_options_command>)
        {
            state_->viewport_options.render_mode = payload.render_mode;
            state_->viewport_options.visualization = payload.visualization;
            state_->viewport_options.overlay = payload.overlay;
            state_->viewport_options.shadows = payload.shadows;
            state_->viewport_options.environment = payload.environment;
            return success();
        }
        else if constexpr (std::is_same_v<command_type, host_viewport_camera_input_command>)
        {
            if (payload.focus_selected)
                focus_selected_entity(state_->scene.scene, state_->scene.selected_entity, state_->camera_controller);
            if (payload.orbit_x != 0.0f || payload.orbit_y != 0.0f)
                state_->camera_controller.orbit(payload.orbit_x, payload.orbit_y);
            if (payload.pan_x != 0.0f || payload.pan_y != 0.0f)
                state_->camera_controller.pan(payload.pan_x, payload.pan_y);
            if (payload.forward != 0.0f)
                state_->camera_controller.move_forward(payload.forward);
            if (payload.zoom != 0.0f)
                state_->camera_controller.zoom(payload.zoom);
            if (auto* camera_transform = state_->scene.scene.try_get<scene::transform_component>(state_->scene.camera_entity))
            {
                state_->camera_controller.apply_to(*camera_transform);
                return success("{\"entity\":" + to_json(to_host_entity(state_->scene.camera_entity)) + '}');
            }
            return fail("No editor camera is available", state_->scene.camera_entity);
        }
        else if constexpr (std::is_same_v<command_type, host_history_undo_command>)
        {
            if (!state_->history.undo(state_->scene)) return fail("Nothing to undo");
            if (auto* camera_transform = state_->scene.scene.try_get<scene::transform_component>(state_->scene.camera_entity))
                state_->camera_controller.synchronize_from(*camera_transform);
            if (const auto& changed = state_->history.last_terrain_change(); changed)
                rebuild_terrain_chunks(state_->scene, *state_->renderer,
                    find_entity_by_guid(state_->scene, changed->entity), &changed->region);
            else
                rebuild_all_terrain_chunks(state_->scene, *state_->renderer);
            push_event(state_->events, state_->event_sequence, host_event_type::scene_changed, "Undo completed", state_->scene.selected_entity);
            return success();
        }
        else if constexpr (std::is_same_v<command_type, host_history_redo_command>)
        {
            if (!state_->history.redo(state_->scene)) return fail("Nothing to redo");
            if (auto* camera_transform = state_->scene.scene.try_get<scene::transform_component>(state_->scene.camera_entity))
                state_->camera_controller.synchronize_from(*camera_transform);
            if (const auto& changed = state_->history.last_terrain_change(); changed)
                rebuild_terrain_chunks(state_->scene, *state_->renderer,
                    find_entity_by_guid(state_->scene, changed->entity), &changed->region);
            else
                rebuild_all_terrain_chunks(state_->scene, *state_->renderer);
            push_event(state_->events, state_->event_sequence, host_event_type::scene_changed, "Redo completed", state_->scene.selected_entity);
            return success();
        }
        else if constexpr (std::is_same_v<command_type, host_viewport_set_tool_command>)
        {
            if (!(payload.translation_snap > 0.0f && payload.rotation_snap_degrees > 0.0f && payload.scale_snap > 0.0f))
                return fail("Viewport snap values must be positive");
            state_->viewport_tool = payload;
            const char* tool = payload.tool == host_viewport_tool::translate ? "translate" :
                payload.tool == host_viewport_tool::rotate ? "rotate" :
                payload.tool == host_viewport_tool::scale ? "scale" :
                payload.tool == host_viewport_tool::terrain ? "terrain" : "select";
            push_event(state_->events, state_->event_sequence, host_event_type::component_changed,
                "Viewport tool changed", {}, "{\"tool\":" + to_json_string(tool) +
                    ",\"coordinateSpace\":" + to_json_string(payload.coordinate_space == host_coordinate_space::local ? "local" : "world") +
                    ",\"snapping\":" + std::string(payload.snapping ? "true" : "false") +
                    ",\"translationSnap\":" + std::to_string(payload.translation_snap) +
                    ",\"rotationSnapDegrees\":" + std::to_string(payload.rotation_snap_degrees) +
                    ",\"scaleSnap\":" + std::to_string(payload.scale_snap) + '}');
            return success();
        }
        else if constexpr (std::is_same_v<command_type, host_viewport_pick_command>)
        {
            const std::uint64_t request_id = state_->next_pick_request_id++;
            scene::update_world_transforms(state_->scene.scene);
            editor_viewport viewport;
            viewport.set_size(static_cast<float>(state_->viewport_options.width), static_cast<float>(state_->viewport_options.height));
            const auto* camera = state_->scene.scene.try_get<scene::camera_component>(state_->scene.camera_entity);
            const auto* transform = state_->scene.scene.try_get<scene::transform_component>(state_->scene.camera_entity);
            editor_pick_result cpu_fallback{};
            if (camera && transform && viewport.valid())
                cpu_fallback = pick_scene_entity(
                    state_->scene.scene,
                    *state_->renderer,
                    screen_ray_from_camera(*camera, *transform, viewport, static_cast<float>(payload.x), static_cast<float>(payload.y)));
            state_->renderer->request_object_pick(request_id, payload.x, payload.y);

            if (state_->renderer->backend())
            {
                state_->pending_pick = state::pending_viewport_pick{
                    .request_id = request_id,
                    .x = payload.x,
                    .y = payload.y,
                    .cpu_fallback = cpu_fallback,
                    .requested_after_frame = state_->viewport_frame_index
                };
                return success("{\"pending\":true}");
            }

            const auto previous_selection = state_->scene.selected_entity;
            if (state_->scene.scene.alive(cpu_fallback.entity))
                select_entity(state_->scene.scene, cpu_fallback.entity, state_->scene.selected_entity);
            else
                clear_selection(state_->scene.scene, state_->scene.selected_entity);
            if (state_->scene.selected_entity != previous_selection)
                push_event(state_->events, state_->event_sequence, host_event_type::entity_selected,
                    cpu_fallback.entity.valid() ? "Viewport entity selected" : "Viewport selection cleared", cpu_fallback.entity);
            return success("{\"entity\":" + to_json(to_host_entity(cpu_fallback.entity)) + '}');
        }

        return fail("Unsupported host command");
    }, command.payload);

    if (authoring && !response.succeeded && command.edit)
        state_->history.cancel(command.edit->id, state_->scene);
    else if (authoring && response.succeeded)
    {
        if (command.edit && command.edit->phase == host_edit_phase::commit)
        {
            if (const auto* terrain_stroke = std::get_if<host_terrain_stroke_command>(&command.payload))
            {
                const auto terrain_entity = to_scene_entity(terrain_stroke->entity);
                state_->history.commit_terrain(
                    command.edit->id, state_->scene, entity_guid_of(state_->scene, terrain_entity));
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
    return response;
}

host_response arc_host::query(const host_query_envelope& query) const
{
    return std::visit([this, request_id = query.request_id](const auto& payload) -> host_response {
        using query_type = std::decay_t<decltype(payload)>;
        if constexpr (std::is_same_v<query_type, host_scene_hierarchy_query>)
        {
            return { .request_id = request_id, .succeeded = true, .payload_json = to_json(scene_snapshot()) };
        }
        else if constexpr (std::is_same_v<query_type, host_selected_entity_query>)
        {
            return { .request_id = request_id, .succeeded = true, .payload_json = to_json(selected_entity_snapshot()) };
        }
        else if constexpr (std::is_same_v<query_type, host_project_assets_query>)
        {
            return { .request_id = request_id, .succeeded = true, .payload_json = to_json(project_assets_snapshot()) };
        }
        else if constexpr (std::is_same_v<query_type, host_asset_thumbnail_query>)
        {
            const auto thumbnail = asset_thumbnail(payload.path, payload.max_size);
            if (!thumbnail)
                return { .request_id = request_id, .succeeded = false, .error = "Texture thumbnail could not be generated" };
            return { .request_id = request_id, .succeeded = true, .payload_json = to_json(*thumbnail) };
        }
        else if constexpr (std::is_same_v<query_type, host_viewport_state_query>)
        {
            return {
                .request_id = request_id,
                .succeeded = true,
                .payload_json = "{\"width\":" + std::to_string(state_->viewport_options.width) +
                    ",\"height\":" + std::to_string(state_->viewport_options.height) +
                    ",\"fps\":" + std::to_string(state_->viewport_fps) +
                    ",\"frameTimeMs\":" + std::to_string(state_->viewport_frame_ms) +
                    ",\"drawCalls\":" + std::to_string(state_->viewport_draw_calls) +
                    ",\"frameIndex\":" + std::to_string(state_->viewport_frame_index) +
                    ",\"submitted\":" + std::string(state_->viewport_submitted ? "true" : "false") + '}'
            };
        }
        else if constexpr (std::is_same_v<query_type, host_world_environment_query>)
        {
            const auto snapshot = world_environment_snapshot(payload.entity);
            if (!snapshot)
                return { .request_id = request_id, .succeeded = false, .error = "World environment is missing" };
            return { .request_id = request_id, .succeeded = true, .payload_json = to_json(*snapshot) };
        }
        else if constexpr (std::is_same_v<query_type, host_history_state_query>)
        {
            const auto history = state_->history.snapshot();
            return { .request_id = request_id, .succeeded = true,
                .payload_json = "{\"canUndo\":" + std::string(history.can_undo ? "true" : "false") +
                    ",\"canRedo\":" + std::string(history.can_redo ? "true" : "false") +
                    ",\"dirty\":" + std::string(history.dirty ? "true" : "false") +
                    ",\"transactionActive\":" + std::string(history.transaction_active ? "true" : "false") +
                    ",\"undoLabel\":" + to_json_string(history.undo_label) +
                    ",\"redoLabel\":" + to_json_string(history.redo_label) +
                    ",\"revision\":" + std::to_string(history.revision) + '}'};
        }
        else if constexpr (std::is_same_v<query_type, host_runtime_state_query>)
        {
            return {
                .request_id = request_id,
                .succeeded = true,
                .payload_json = to_json(runtime_snapshot())
            };
        }
        else if constexpr (std::is_same_v<query_type, host_terrain_tool_state_query>)
        {
            return {
                .request_id = request_id,
                .succeeded = true,
                .payload_json = to_json(terrain_tool_snapshot())
            };
        }

        return { .request_id = request_id, .succeeded = false, .error = "Unsupported host query" };
    }, query.payload);
}

host_runtime_snapshot arc_host::runtime_snapshot() const
{
    host_runtime_snapshot snapshot;
    snapshot.state = state_->preview_stopped
        ? host_runtime_state::stopped
        : state_->simulation.paused()
        ? host_runtime_state::paused
        : state_->simulation.running()
            ? host_runtime_state::running
            : host_runtime_state::stopped;
    for (const runtime_world_id id : state_->simulation.worlds().ordered_worlds())
    {
        const runtime_world* world = state_->simulation.worlds().find(id);
        if (world && world->state() == runtime_world_state::faulted)
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
    snapshot.tool = state_->terrain_brush.tool == scene::terrain_brush_tool::smooth
        ? host_terrain_brush_tool::smooth
        : state_->terrain_brush.tool == scene::terrain_brush_tool::flatten
            ? host_terrain_brush_tool::flatten
            : state_->terrain_brush.tool == scene::terrain_brush_tool::paint
                ? host_terrain_brush_tool::paint
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
    snapshot.scene_guid = scene::to_string(state_->scene.scene_guid);
    snapshot.scene_name = state_->scene.scene_name;
    snapshot.active_scene_path = state_->scene.active_scene_path.generic_string();
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
        if (root != state_->scene.camera_entity)
            root_orders[root.index] = root_order++;
    const auto add = [&](scene::entity entity) {
        if (!entity.valid() || !state_->scene.scene.alive(entity))
            return;
        host_entity_kind kind = host_entity_kind::unknown;
        if (state_->scene.scene.has<scene::world_environment_component>(entity) ||
            state_->scene.scene.has<scene::terrain_component>(entity) || state_->scene.scene.has<scene::water_component>(entity) ||
            state_->scene.scene.has<scene::vegetation_component>(entity) || state_->scene.scene.has<scene::decal_component>(entity))
            kind = host_entity_kind::environment;
        else if (state_->scene.scene.has<scene::directional_light_component>(entity) ||
            state_->scene.scene.has<scene::point_light_component>(entity) ||
            state_->scene.scene.has<scene::spot_light_component>(entity) ||
            state_->scene.scene.has<scene::area_light_component>(entity))
            kind = host_entity_kind::light;
        else if (state_->scene.scene.has<scene::camera_component>(entity)) kind = host_entity_kind::camera;
        else if (state_->scene.scene.has<scene::mesh_renderer_component>(entity)) kind = host_entity_kind::mesh;
        std::string parent_guid;
        std::uint32_t sibling_order{};
        if (const auto* hierarchy = state_->scene.scene.try_get<scene::hierarchy_component>(entity))
        {
            if (state_->scene.scene.alive(hierarchy->parent))
            {
                parent_guid = scene::to_string(entity_guid_of(state_->scene, hierarchy->parent));
                auto sibling = hierarchy->previous_sibling;
                std::unordered_set<scene::entity, ecs::entity_hash> visited_siblings;
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
        snapshot.entities.push_back({
            .entity = to_host_entity(entity),
            .guid = scene::to_string(entity_guid_of(state_->scene, entity)),
            .parent_guid = std::move(parent_guid),
            .sibling_order = sibling_order,
            .name = entity_name(state_->scene, entity, "Entity"),
            .kind = kind,
            .active = entity_active(state_->scene, entity),
            .selected = entity == state_->scene.selected_entity
        });
    };

    for (const auto entity : state_->scene.scene.entities())
        if (entity != state_->scene.camera_entity)
            add(entity);
    return snapshot;
}

host_selected_entity_snapshot arc_host::selected_entity_snapshot() const
{
    host_selected_entity_snapshot snapshot;
    const auto selected = state_->scene.selected_entity;
    if (!state_->scene.scene.alive(selected))
        return snapshot;

    snapshot.entity = to_host_entity(selected);
    snapshot.name = entity_name(state_->scene, selected, "Unnamed Entity");
    if (const auto* tag = state_->scene.scene.try_get<scene::tag_component>(selected))
        snapshot.tag = tag->value;
    snapshot.active = entity_active(state_->scene, selected);
    if (const auto* layer = state_->scene.scene.try_get<scene::render_layer_component>(selected))
        snapshot.render_layer_mask = layer->mask;
    if (const auto* transform = state_->scene.scene.try_get<scene::transform_component>(selected))
    {
        snapshot.transform = to_host_transform(*transform);
        add_component_snapshot(snapshot.components, host_component_kind::transform, "Transform");
    }
    if (const auto* camera = state_->scene.scene.try_get<scene::camera_component>(selected))
    {
        snapshot.camera = to_host_camera(*camera);
        add_component_snapshot(snapshot.components, host_component_kind::camera, "Camera");
    }
    if (const auto* mesh_renderer = state_->scene.scene.try_get<scene::mesh_renderer_component>(selected))
    {
        snapshot.mesh_renderer = mesh_renderer_snapshot(state_->scene, state_->assets, *mesh_renderer);
        add_component_snapshot(snapshot.components, host_component_kind::mesh_renderer, "Mesh Renderer");
    }
    if (const auto* light = state_->scene.scene.try_get<scene::directional_light_component>(selected))
    {
        snapshot.light = to_host_light(*light);
        add_component_snapshot(snapshot.components, host_component_kind::directional_light, "Directional Light");
    }
    if (const auto* light = state_->scene.scene.try_get<scene::point_light_component>(selected))
    {
        snapshot.light = to_host_light(*light);
        add_component_snapshot(snapshot.components, host_component_kind::point_light, "Point Light");
    }
    if (const auto* light = state_->scene.scene.try_get<scene::spot_light_component>(selected))
    {
        snapshot.light = to_host_light(*light);
        add_component_snapshot(snapshot.components, host_component_kind::spot_light, "Spot Light");
    }
    if (const auto* light = state_->scene.scene.try_get<scene::area_light_component>(selected))
    {
        snapshot.light = to_host_light(*light);
        add_component_snapshot(snapshot.components, host_component_kind::area_light, "Area Light");
    }
    if (state_->scene.scene.has<scene::sky_atmosphere_component>(selected))
        add_component_snapshot(snapshot.components, host_component_kind::sky_atmosphere, "Sky Atmosphere");
    if (state_->scene.scene.has<scene::world_environment_component>(selected))
        add_component_snapshot(snapshot.components, host_component_kind::world_environment, "World Environment");
    if (state_->scene.scene.has<scene::celestial_sky_component>(selected))
        add_component_snapshot(snapshot.components, host_component_kind::celestial_sky, "Sun, Moon & Time");
    if (state_->scene.scene.has<scene::cloud_layers_component>(selected))
        add_component_snapshot(snapshot.components, host_component_kind::cloud_layers, "Cloud Layers");
    if (state_->scene.scene.has<scene::environment_lighting_component>(selected))
        add_component_snapshot(snapshot.components, host_component_kind::environment_lighting, "Environment Lighting");
    if (state_->scene.scene.has<scene::height_fog_component>(selected))
        add_component_snapshot(snapshot.components, host_component_kind::height_fog, "Height Fog");
    if (const auto* terrain = state_->scene.scene.try_get<scene::terrain_component>(selected))
    {
        host_terrain_snapshot terrain_snapshot;
        terrain_snapshot.enabled = terrain->enabled;
        terrain_snapshot.size = terrain->size;
        terrain_snapshot.resolution = terrain->subdivisions + 1u;
        terrain_snapshot.chunk_quads = terrain->chunk_quads;
        terrain_snapshot.receive_shadows = terrain->receive_shadows;
        terrain_snapshot.content_revision = terrain->content_revision;
        terrain_snapshot.brush_tool = state_->terrain_brush.tool == scene::terrain_brush_tool::smooth ? host_terrain_brush_tool::smooth :
            state_->terrain_brush.tool == scene::terrain_brush_tool::flatten ? host_terrain_brush_tool::flatten :
            state_->terrain_brush.tool == scene::terrain_brush_tool::paint ? host_terrain_brush_tool::paint : host_terrain_brush_tool::sculpt;
        terrain_snapshot.brush_radius = state_->terrain_brush.radius;
        terrain_snapshot.brush_strength = state_->terrain_brush.strength;
        terrain_snapshot.brush_falloff = state_->terrain_brush.falloff;
        terrain_snapshot.active_layer = state_->terrain_brush.active_layer;
        for (std::size_t layer = 0; layer < state_->scene.terrain_layer_paths.size(); ++layer)
            terrain_snapshot.layer_base_color_paths[layer] = asset_relative_path(
                state_->assets.root, state_->scene.terrain_layer_paths[layer]);
        snapshot.terrain = std::move(terrain_snapshot);
        add_component_snapshot(snapshot.components, host_component_kind::terrain, "Terrain");
    }
    if (state_->scene.scene.has<scene::water_component>(selected))
        add_component_snapshot(snapshot.components, host_component_kind::water, "Water");
    if (state_->scene.scene.has<scene::vegetation_component>(selected))
        add_component_snapshot(snapshot.components, host_component_kind::vegetation, "Vegetation");
    if (state_->scene.scene.has<scene::decal_component>(selected))
        add_component_snapshot(snapshot.components, host_component_kind::decal, "Decal");
    if (const auto* prefab = state_->scene.scene.try_get<scene::prefab_instance_component>(selected))
    {
        snapshot.prefab = host_prefab_snapshot{
            .prefab_guid = scene::to_string(prefab->prefab_guid),
            .prefab_path = prefab->prefab_path,
            .override_count = prefab->overrides.size(),
            .source_missing = prefab->source_missing
        };
        add_component_snapshot(snapshot.components, host_component_kind::prefab_instance, "Prefab Instance");
    }
    return snapshot;
}

std::optional<host_world_environment_snapshot> arc_host::world_environment_snapshot(host_entity_id host_entity) const
{
    const auto settings = scene::read_world_environment_settings(
        state_->scene.scene,
        to_scene_entity(host_entity));
    if (!settings)
        return std::nullopt;
    return to_host_world_environment_snapshot(
        host_entity,
        *settings,
        state_->scene.world_environment_hdri_path);
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
    if (!state_->assets.default_mesh_path.empty())
    {
        snapshot.assets.push_back({
            .path = state_->assets.default_mesh_path.generic_string(),
            .kind = "scene",
            .imported = state_->assets.default_mesh_loaded,
            .import_running = false
        });
    }
    for (const auto& material : state_->scene.material_library.materials)
    {
        snapshot.assets.push_back({
            .path = asset_relative_path(state_->assets.root, material.path),
            .kind = "material",
            .imported = true,
            .import_running = false
        });
    }
    if (!state_->assets.root.empty())
    {
        std::error_code error;
        for (std::filesystem::recursive_directory_iterator iterator(
                 state_->assets.root,
                 std::filesystem::directory_options::skip_permission_denied,
                 error), end;
             iterator != end && !error;
             iterator.increment(error))
        {
            if (!iterator->is_regular_file(error))
                continue;
            auto extension = iterator->path().extension().string();
            std::transform(extension.begin(), extension.end(), extension.begin(), [](unsigned char value) {
                return static_cast<char>(std::tolower(value));
            });
            const bool texture_asset = render::is_supported_texture_asset(iterator->path());
            const bool material_asset_path = is_material_asset_path(iterator->path());
            const bool prefab_asset_path = extension == ".arcprefab";
            if (!texture_asset && !material_asset_path && !prefab_asset_path)
                continue;
            const auto relative_path = std::filesystem::relative(iterator->path(), state_->assets.root, error).generic_string();
            if (std::any_of(snapshot.assets.begin(), snapshot.assets.end(), [&](const auto& asset) {
                    return asset.path == relative_path;
                }))
                continue;
            snapshot.assets.push_back({
                .path = relative_path,
                .kind = prefab_asset_path ? "prefab" :
                    material_asset_path ? "material" : extension == ".hdr" ? "environment" : "texture",
                .imported = true,
                .import_running = false
            });
        }
    }
    return snapshot;
}

std::optional<host_asset_thumbnail_snapshot> arc_host::asset_thumbnail(std::string_view path, std::uint32_t max_size) const
{
    const auto resolved = resolve_project_asset(state_->assets.root, std::filesystem::path{ path });
    if (!resolved)
        return std::nullopt;
    render::texture_data preview;
    if (is_material_asset_path(*resolved))
    {
        material_asset material;
        std::string message;
        if (!load_material_asset(*resolved, state_->assets.root, material, message))
            return std::nullopt;
        auto rendered = render_material_preview(material, state_->assets.root, std::clamp(max_size, 32u, 256u));
        if (!rendered.succeeded())
            return std::nullopt;
        preview = std::move(rendered.texture);
    }
    else
    {
        if (!render::is_supported_texture_asset(*resolved))
            return std::nullopt;
        const auto loaded = render::load_texture_asset(*resolved);
        if (!loaded.succeeded())
            return std::nullopt;
        preview = loaded.texture;
    }
    const auto bmp = texture_preview_bmp(preview, std::clamp(max_size, 32u, 256u));
    if (bmp.empty())
        return std::nullopt;
    const float scale = std::min(
        1.0f,
        static_cast<float>(std::clamp(max_size, 32u, 256u)) /
            static_cast<float>(std::max(preview.width, preview.height)));
    return host_asset_thumbnail_snapshot{
        .path = std::string(path),
        .width = std::max(1u, static_cast<std::uint32_t>(std::lround(preview.width * scale))),
        .height = std::max(1u, static_cast<std::uint32_t>(std::lround(preview.height * scale))),
        .data_url = "data:image/bmp;base64," + base64_encode(bmp)
    };
}

std::vector<host_event> arc_host::poll_events()
{
    auto events = std::move(state_->events);
    state_->events.clear();
    return events;
}

host_viewport_frame arc_host::request_viewport(const host_viewport_request& request)
{
    const auto frame_start = std::chrono::steady_clock::now();
    float delta_seconds = 0.0f;
    if (state_->last_viewport_frame_time.time_since_epoch().count() != 0)
        delta_seconds = std::chrono::duration<float>(frame_start - state_->last_viewport_frame_time).count();
    const frame_time runtime_frame = state_->simulation.advance(delta_seconds);
    if (runtime_frame.last_completed_tick.value != state_->last_runtime_tick_event)
    {
        state_->last_runtime_tick_event = runtime_frame.last_completed_tick.value;
        ++state_->runtime_revision;
        host_runtime_snapshot snapshot = runtime_snapshot();
        snapshot.interpolation_alpha = runtime_frame.interpolation_alpha;
        push_event(
            state_->events,
            state_->event_sequence,
            snapshot.state == host_runtime_state::faulted
                ? host_event_type::runtime_fault
                : host_event_type::runtime_tick_completed,
            snapshot.state == host_runtime_state::faulted
                ? "Preview runtime faulted"
                : "Preview runtime tick completed",
            {},
            to_json(snapshot));
    }
    state_->viewport_options = request;
    state_->viewport_frame_index = request.frame_index;
    if (!state_->renderer->backend())
    {
        const std::string message = "Viewport render skipped: no render backend attached";
        state_->viewport_submitted = false;
        push_event(state_->events, state_->event_sequence, host_event_type::viewport_error, message);
        return { .message = message };
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
            scene::entity picked = gpu_ready && gpu.hit
                ? scene::entity{ gpu.object.index, gpu.object.generation }
                : state_->pending_pick->cpu_fallback.entity;
            if (gpu_ready && gpu.hit && state_->scene.scene.alive(picked))
            {
                const bool gpu_background = state_->scene.scene.has<scene::terrain_component>(picked) ||
                    state_->scene.scene.has<scene::water_component>(picked) ||
                    state_->scene.scene.has<scene::world_environment_component>(picked);
                if (gpu_background && state_->pending_pick->cpu_fallback.entity.valid() &&
                    !state_->pending_pick->cpu_fallback.background &&
                    state_->pending_pick->cpu_fallback.exact)
                    picked = state_->pending_pick->cpu_fallback.entity;
            }
            if (gpu_ready && gpu.hit && !state_->scene.scene.alive(picked))
                picked = state_->pending_pick->cpu_fallback.entity;
            arc::debug(
                "editor.pick",
                "pick request=" + std::to_string(state_->pending_pick->request_id) +
                    " source=" + std::string(gpu_ready && gpu.hit ? "gpu" : "cpu") +
                    " x=" + std::to_string(state_->pending_pick->x) +
                    " y=" + std::to_string(state_->pending_pick->y) +
                    " entity=" + std::to_string(picked.index) +
                    " cpuDistance=" + std::to_string(state_->pending_pick->cpu_fallback.distance));
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
        if (auto* camera_transform = state_->scene.scene.try_get<scene::transform_component>(state_->scene.camera_entity))
            state_->camera_controller.apply_to(*camera_transform);
        state_->scene.focus_imported_scene_requested = false;
    }

    auto debug_overlay = build_editor_gizmo_overlay(
        state_->scene.scene,
        state_->scene.selected_entity,
        state_->scene.camera_entity,
        editor_gizmo_context{
            .tool = to_editor_tool(state_->viewport_tool.tool),
            .coordinate_space = state_->viewport_tool.coordinate_space == host_coordinate_space::local
                ? gizmo_coordinate_space::local : gizmo_coordinate_space::world,
            .highlighted_axis = state_->gizmo_highlight,
            .viewport_width = request.width,
            .viewport_height = request.height });
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
                    (*state_->terrain_brush_local_position)[0] + std::cos(angle0) * state_->terrain_brush.radius,
                    0.0f,
                    (*state_->terrain_brush_local_position)[2] + std::sin(angle0) * state_->terrain_brush.radius };
                auto point1 = math::vector3f{
                    (*state_->terrain_brush_local_position)[0] + std::cos(angle1) * state_->terrain_brush.radius,
                    0.0f,
                    (*state_->terrain_brush_local_position)[2] + std::sin(angle1) * state_->terrain_brush.radius };
                point0[1] = scene::sample_terrain_height(*terrain, point0[0], point0[2]) + 0.04f;
                point1[1] = scene::sample_terrain_height(*terrain, point1[0], point1[2]) + 0.04f;
                debug_overlay.lines.push_back({
                    .start = math::transform_point(world, point0),
                    .end = math::transform_point(world, point1),
                    .color = { 0.95f, 0.68f, 0.16f, 1.0f },
                    .depth = render::debug_overlay_depth_mode::tested });
            }
        }
    }
    state_->scene.last_render = scene::render_scene(
        state_->scene.scene,
        *state_->renderer,
        request.width,
        request.height,
        to_render_mode(request.render_mode),
        to_visualization(request.visualization),
        to_overlay(request.overlay),
        request.shadows,
        to_scene_visibility(request.environment),
        delta_seconds,
        std::move(debug_overlay));

    const auto submit_result = state_->renderer->render_frame(
        request.frame_index,
        render::make_scene_draw_graph(
            "viewport",
            state_->renderer->resolved_config(),
            true,
            state_->scene.last_render.environment));
    const auto frame_end = std::chrono::steady_clock::now();
    state_->viewport_frame_ms = std::chrono::duration<double, std::milli>(frame_end - frame_start).count();
    if (state_->last_viewport_frame_time.time_since_epoch().count() != 0)
    {
        const double delta_seconds = std::chrono::duration<double>(frame_end - state_->last_viewport_frame_time).count();
        if (delta_seconds > 0.0)
            state_->viewport_fps = 1.0 / delta_seconds;
    }
    state_->last_viewport_frame_time = frame_end;
    state_->viewport_draw_calls = state_->scene.last_render.submitted_draw_count;
    state_->viewport_submitted = submit_result.submitted;
    if (!submit_result.submitted && !submit_result.message.empty())
    {
        arc::error("editor.host", submit_result.message);
        push_event(state_->events, state_->event_sequence, host_event_type::viewport_error, submit_result.message);
    }

    return {
        .submitted = submit_result.submitted,
        .message = submit_result.message,
        .payload_json = "{\"drawCalls\":" + std::to_string(state_->scene.last_render.submitted_draw_count) + '}'
    };
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

in_process_host_session::in_process_host_session(std::shared_ptr<arc_host> host)
    : host_(std::move(host))
{
}

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
        arc::info("editor.host", "Acquired existing Arc Host");
        return existing;
    }

    auto created = std::make_shared<arc_host>(std::move(renderer));
    host_ = created;
    return created;
}

} // namespace arc::editor
