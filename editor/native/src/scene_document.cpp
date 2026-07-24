#include <arc/editor/scene_document.h>

#include <arc/editor/editor_interaction.h>
#include <arc/editor/world_environment_host.h>

#include <nlohmann/json.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <limits>
#include <optional>
#include <span>
#include <sstream>
#include <unordered_map>
#include <unordered_set>

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

namespace arc::editor
{
namespace
{
using json = nlohmann::json;

constexpr std::string_view base64_alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

std::string base64_encode(std::span<const std::uint8_t> bytes)
{
    std::string output;
    output.reserve(((bytes.size() + 2u) / 3u) * 4u);
    for (std::size_t offset = 0; offset < bytes.size(); offset += 3u)
    {
        const std::uint32_t first = bytes[offset];
        const std::uint32_t second = offset + 1u < bytes.size() ? bytes[offset + 1u] : 0u;
        const std::uint32_t third = offset + 2u < bytes.size() ? bytes[offset + 2u] : 0u;
        const std::uint32_t packed = (first << 16u) | (second << 8u) | third;
        output.push_back(base64_alphabet[(packed >> 18u) & 63u]);
        output.push_back(base64_alphabet[(packed >> 12u) & 63u]);
        output.push_back(offset + 1u < bytes.size() ? base64_alphabet[(packed >> 6u) & 63u] : '=');
        output.push_back(offset + 2u < bytes.size() ? base64_alphabet[packed & 63u] : '=');
    }
    return output;
}

std::optional<std::vector<std::uint8_t>> base64_decode(std::string_view text)
{
    if (text.size() % 4u != 0u)
        return std::nullopt;
    std::array<int, 256> lookup{};
    lookup.fill(-1);
    for (std::size_t index = 0; index < base64_alphabet.size(); ++index)
        lookup[static_cast<std::uint8_t>(base64_alphabet[index])] = static_cast<int>(index);
    std::vector<std::uint8_t> output;
    output.reserve((text.size() / 4u) * 3u);
    for (std::size_t offset = 0; offset < text.size(); offset += 4u)
    {
        std::uint32_t packed{};
        int padding{};
        for (std::size_t channel = 0; channel < 4u; ++channel)
        {
            const char value = text[offset + channel];
            if (value == '=')
            {
                ++padding;
                packed <<= 6u;
            }
            else
            {
                if (padding != 0 || lookup[static_cast<std::uint8_t>(value)] < 0)
                    return std::nullopt;
                packed = (packed << 6u) | static_cast<std::uint32_t>(lookup[static_cast<std::uint8_t>(value)]);
            }
        }
        output.push_back(static_cast<std::uint8_t>((packed >> 16u) & 0xffu));
        if (padding < 2) output.push_back(static_cast<std::uint8_t>((packed >> 8u) & 0xffu));
        if (padding < 1) output.push_back(static_cast<std::uint8_t>(packed & 0xffu));
    }
    return output;
}

json vector3(const math::vector3f& value) { return json::array({ value[0], value[1], value[2] }); }
json vector4(const math::vector4f& value) { return json::array({ value[0], value[1], value[2], value[3] }); }
json quaternion(const math::quatf& value) { return json::array({ value[0], value[1], value[2], value[3] }); }

bool finite_array(const json& value, std::size_t size)
{
    if (!value.is_array() || value.size() != size)
        return false;
    return std::all_of(value.begin(), value.end(), [](const json& item) {
        return item.is_number() && std::isfinite(item.get<double>());
    });
}

bool finite_number(const json& object, std::string_view key)
{
    const auto found = object.find(std::string(key));
    return found != object.end() && found->is_number() && std::isfinite(found->get<double>());
}

bool finite_color(const json& value, std::size_t size)
{
    return finite_array(value, size) && std::all_of(value.begin(), value.end(), [](const json& channel) {
        const auto number = channel.get<double>();
        return number >= 0.0 && number <= 1.0;
    });
}

bool validate_component_json(std::string_view name, const json& value, std::string& error)
{
    if (!value.is_object() || !value.contains("version") || !value["version"].is_number_unsigned())
    {
        error = "component '" + std::string(name) + "' has no valid schema version";
        return false;
    }
    static const std::unordered_set<std::string_view> known{
        "Name", "Tag", "Active", "RenderLayer", "Transform", "Camera", "MeshRenderer", "DirectionalLight",
        "PointLight", "SpotLight", "AreaLight", "WorldEnvironment", "Terrain", "Water", "Vegetation", "Decal",
        "PrefabInstance", "WorldRegion" };
    if (!known.contains(name)) return true;
    const auto component_version = value["version"].get<std::uint32_t>();
    const bool supports_v2 = name == "Terrain" || name == "Camera" ||
        name == "DirectionalLight" || name == "PointLight" ||
        name == "SpotLight" || name == "AreaLight";
    if (component_version != 1u && !(supports_v2 && component_version == 2u))
    {
        error = "component '" + std::string(name) + "' uses an unsupported schema version";
        return false;
    }
    const auto fail = [&](std::string_view detail) {
        error = "component '" + std::string(name) + "' " + std::string(detail);
        return false;
    };
    if (name == "Name" || name == "Tag")
        return value.contains("value") && value["value"].is_string() ? true : fail("has an invalid string value");
    if (name == "Active")
        return value.contains("value") && value["value"].is_boolean() ? true : fail("has an invalid active value");
    if (name == "RenderLayer")
        return value.contains("mask") && value["mask"].is_number_unsigned() ? true : fail("has an invalid layer mask");
    if (name == "WorldRegion")
        return value.contains("id") && value["id"].is_string() &&
            scene::parse_entity_guid(value["id"].get<std::string>()).has_value()
            ? true : fail("has an invalid region id");
    if (name == "PrefabInstance")
    {
        if (!value.contains("prefabGuid") || !value["prefabGuid"].is_string() ||
            !scene::parse_entity_guid(value["prefabGuid"].get<std::string>()) ||
            !value.contains("prefabPath") || !value["prefabPath"].is_string() ||
            !value.contains("sourceRoot") || !value["sourceRoot"].is_string() ||
            !scene::parse_entity_guid(value["sourceRoot"].get<std::string>()) ||
            !value.contains("mapping") || !value["mapping"].is_array() ||
            !value.contains("overrides") || !value["overrides"].is_array())
            return fail("has malformed prefab identity or override data");
        return true;
    }
    if (name == "Transform")
    {
        if (!value.contains("position") || !value.contains("rotation") || !value.contains("scale") ||
            !finite_array(value["position"], 3) || !finite_array(value["rotation"], 4) || !finite_array(value["scale"], 3))
            return fail("has non-finite TRS values");
        double rotation_length{};
        for (const auto& channel : value["rotation"]) rotation_length += channel.get<double>() * channel.get<double>();
        if (rotation_length <= 1.0e-12) return fail("has a zero-length rotation");
        return true;
    }
    if (name == "Camera")
    {
        if (!value.contains("projection") || !value["projection"].is_number_integer() ||
            value["projection"].get<int>() < 0 || value["projection"].get<int>() > 1 ||
            !finite_number(value, "fovY") || !finite_number(value, "orthographicHeight") ||
            !finite_number(value, "near") || !finite_number(value, "far") ||
            !value.contains("active") || !value["active"].is_boolean() ||
            !value.contains("clearColor") || !finite_color(value["clearColor"], 4))
            return fail("has invalid authored values");
        const auto fov = value["fovY"].get<double>();
        const auto near_plane = value["near"].get<double>();
        const bool projection_valid = fov > math::to_radians(1.0) && fov < math::to_radians(179.0) &&
            value["orthographicHeight"].get<double>() > 0.0 && near_plane > 0.0 && value["far"].get<double>() > near_plane
            ;
        if (!projection_valid)
            return fail("is outside supported projection ranges");
        if (value.contains("exposure"))
        {
            const auto& exposure = value["exposure"];
            if (!exposure.is_object() ||
                !finite_number(exposure, "manualEV100") ||
                !finite_number(exposure, "compensationEV") ||
                !finite_number(exposure, "minimumEV100") ||
                !finite_number(exposure, "maximumEV100") ||
                !finite_number(exposure, "brightenSpeed") ||
                !finite_number(exposure, "darkenSpeed") ||
                exposure["maximumEV100"].get<double>() <= exposure["minimumEV100"].get<double>() ||
                exposure["brightenSpeed"].get<double>() < 0.0 ||
                exposure["darkenSpeed"].get<double>() < 0.0)
                return fail("has invalid exposure settings");
        }
        return true;
    }
    if (name == "MeshRenderer")
        return value.contains("visible") && value["visible"].is_boolean() && value.contains("baseColorTint") &&
            finite_color(value["baseColorTint"], 4) ? true : fail("has invalid renderer values");
    if (name == "WorldEnvironment")
    {
        host_world_environment_snapshot snapshot;
        std::string parse_error;
        if (!from_json(value.dump(), snapshot, parse_error)) return fail("is malformed");
        scene::world_environment_settings defaults;
        const auto validation = scene::validate_world_environment(apply_host_world_environment_snapshot(snapshot, defaults));
        return validation.valid ? true : fail("is outside supported ranges");
    }
    const auto enabled_valid = !value.contains("enabled") || value["enabled"].is_boolean();
    if (!enabled_valid) return fail("has an invalid enabled value");
    if (name == "Terrain")
    {
        const bool common_valid = finite_number(value, "size") && value["size"].get<double>() > 0.0 &&
            value.contains("subdivisions") && value["subdivisions"].is_number_unsigned() &&
            value["subdivisions"].get<std::uint32_t>() >= 2u && finite_number(value, "heightScale") &&
            value["heightScale"].get<double>() >= 0.0 && value.contains("baseColor") && finite_color(value["baseColor"], 3) &&
            value.contains("receiveShadows") && value["receiveShadows"].is_boolean();
        if (!common_valid)
            return fail("has invalid terrain values");
        if (component_version == 1u)
            return true;
        if (value["subdivisions"].get<std::uint32_t>() != 256u ||
            !value.contains("chunkQuads") || !value["chunkQuads"].is_number_unsigned() || value["chunkQuads"].get<std::uint32_t>() != 128u ||
            !finite_number(value, "minHeight") || !finite_number(value, "heightRange") || value["heightRange"].get<double>() < 0.0 ||
            !value.contains("heights") || !value["heights"].is_string() ||
            !value.contains("weights") || !value["weights"].is_string())
            return fail("has invalid version-2 terrain metadata");
        const auto heights = base64_decode(value["heights"].get<std::string>());
        const auto weights = base64_decode(value["weights"].get<std::string>());
        constexpr std::size_t sample_count = 257u * 257u;
        if (!heights || heights->size() != sample_count * sizeof(std::uint16_t) ||
            !weights || weights->size() != sample_count * 4u)
            return fail("has invalid height/weight payload byte counts");
        for (std::size_t sample = 0; sample < sample_count; ++sample)
        {
            const auto offset = sample * 4u;
            const auto sum = static_cast<std::uint32_t>((*weights)[offset]) + (*weights)[offset + 1u] +
                (*weights)[offset + 2u] + (*weights)[offset + 3u];
            if (sum != 255u)
                return fail("contains non-normalized layer weights");
        }
        return true;
    }
    if (name == "Water")
        return finite_number(value, "size") && value["size"].get<double>() > 0.0 && value.contains("color") &&
            finite_color(value["color"], 3) && finite_number(value, "roughness") && value["roughness"].get<double>() >= 0.0 &&
            value["roughness"].get<double>() <= 1.0 && finite_number(value, "waveScale") && value["waveScale"].get<double>() >= 0.0 &&
            finite_number(value, "waveSpeed") && value["waveSpeed"].get<double>() >= 0.0 && finite_number(value, "transparency") &&
            value["transparency"].get<double>() >= 0.0 && value["transparency"].get<double>() <= 1.0 ? true : fail("has invalid water values");
    if (name == "Vegetation")
        return finite_number(value, "density") && value["density"].get<double>() >= 0.0 && finite_number(value, "patchSize") &&
            value["patchSize"].get<double>() > 0.0 && value.contains("color") && finite_color(value["color"], 3) &&
            finite_number(value, "windStrength") && value["windStrength"].get<double>() >= 0.0 &&
            finite_number(value, "windSpeed") && value["windSpeed"].get<double>() >= 0.0 ? true : fail("has invalid vegetation values");
    if (name == "Decal")
        return value.contains("color") && finite_color(value["color"], 4) && finite_number(value, "opacity") &&
            value["opacity"].get<double>() >= 0.0 && value["opacity"].get<double>() <= 1.0 ? true : fail("has invalid decal values");
    // Light validation is completed before any scene resource is created.
    if (!value.contains("color") || !finite_color(value["color"], 3) || !finite_number(value, "intensity") ||
        value["intensity"].get<double>() < 0.0 || !value.contains("castsShadows") || !value["castsShadows"].is_boolean() ||
        !value.contains("useColorTemperature") || !value["useColorTemperature"].is_boolean() ||
        !finite_number(value, "temperatureKelvin") || value["temperatureKelvin"].get<double>() < 1000.0 ||
        value["temperatureKelvin"].get<double>() > 40000.0 || !value.contains("intensityUnit") ||
        !value["intensityUnit"].is_number_integer())
        return fail("has invalid light values");
    if (value.contains("shadow"))
    {
        const auto& shadow = value["shadow"];
        if (!shadow.is_object() || (shadow.contains("enabled") && !shadow["enabled"].is_boolean()) ||
            (shadow.contains("resolution") && !shadow["resolution"].is_number_unsigned()) ||
            (shadow.contains("bias") && !finite_number(shadow, "bias")) ||
            (shadow.contains("normalBias") && !finite_number(shadow, "normalBias")) ||
            (shadow.contains("strength") && !finite_number(shadow, "strength")) ||
            (shadow.contains("filter") && !shadow["filter"].is_number_integer()))
            return fail("has invalid shadow settings");
    }
    if (name != "DirectionalLight" && name != "AreaLight" &&
        (!finite_number(value, "range") || value["range"].get<double>() <= 0.0))
        return fail("has an invalid range");
    if (name == "SpotLight" && (!finite_number(value, "innerAngle") || !finite_number(value, "outerAngle") ||
        value["innerAngle"].get<double>() < 0.0 || value["outerAngle"].get<double>() <= value["innerAngle"].get<double>() ||
        value["outerAngle"].get<double>() > math::pi<double>))
        return fail("has invalid cone angles");
    if (name == "AreaLight" &&
        (!finite_number(value, "width") || value["width"].get<double>() <= 0.0 ||
            !finite_number(value, "height") || value["height"].get<double>() <= 0.0 ||
            !value.contains("shape") || !value["shape"].is_number_integer() ||
            value["shape"].get<int>() < 0 || value["shape"].get<int>() > 1 ||
            !value.contains("twoSided") || !value["twoSided"].is_boolean()))
        return fail("has invalid area-light dimensions");
    return true;
}

math::vector3f read_vector3(const json& value)
{
    return { value[0].get<float>(), value[1].get<float>(), value[2].get<float>() };
}

math::vector4f read_vector4(const json& value)
{
    return { value[0].get<float>(), value[1].get<float>(), value[2].get<float>(), value[3].get<float>() };
}

math::quatf read_quaternion(const json& value)
{
    return math::normalize(math::quatf{
        value[0].get<float>(), value[1].get<float>(), value[2].get<float>(), value[3].get<float>() });
}

std::filesystem::path relative_asset_path(const std::filesystem::path& path, const std::filesystem::path& project_root)
{
    if (path.empty())
        return {};
    std::error_code error;
    const auto relative = std::filesystem::relative(path, project_root, error);
    return error ? path.lexically_normal() : relative.lexically_normal();
}

std::vector<scene::entity> ordered_entities(const editor_scene_state& state);

bool is_normal_project_relative_path(const std::filesystem::path& path, const std::filesystem::path& project_root)
{
    if (path.empty()) return true;
    const auto relative = relative_asset_path(path, project_root);
    if (relative.empty() || relative.is_absolute() || relative.has_root_name()) return false;
    return std::none_of(relative.begin(), relative.end(), [](const auto& part) { return part == ".."; });
}

std::optional<std::filesystem::path> resolve_document_asset_path(
    std::string_view text, const std::filesystem::path& project_root)
{
    if (text.empty()) return std::filesystem::path{};
    if (text.find('\\') != std::string_view::npos) return std::nullopt;
    const std::filesystem::path authored(text);
    const auto normalized = authored.lexically_normal();
    if (normalized.is_absolute() || normalized.has_root_name() ||
        std::any_of(normalized.begin(), normalized.end(), [](const auto& part) { return part == ".."; }))
        return std::nullopt;
    return (project_root / normalized).lexically_normal();
}

bool validate_scene_for_save(const editor_scene_state& state, const std::filesystem::path& project_root, std::string& error)
{
    const auto all_entities = state.scene.entities();
    std::unordered_set<std::string> guids;
    std::size_t persisted_count{};
    for (const auto value : all_entities)
    {
        if (value == state.camera_entity) continue;
        ++persisted_count;
        const auto guid = entity_guid_of(state, value);
        if (!guid.valid() || !guids.insert(scene::to_string(guid)).second)
        {
            error = "scene contains a missing or duplicate entity GUID";
            return false;
        }
        const auto* hierarchy = state.scene.try_get<scene::hierarchy_component>(value);
        if (!hierarchy)
        {
            error = "scene entity is missing hierarchy metadata";
            return false;
        }
        if (hierarchy->parent.valid() && (!state.scene.alive(hierarchy->parent) || hierarchy->parent == state.camera_entity))
        {
            error = "scene entity has an invalid or editor-only parent";
            return false;
        }
        auto ancestor = hierarchy->parent;
        for (std::size_t depth = 0; state.scene.alive(ancestor); ++depth)
        {
            if (ancestor == value || depth >= all_entities.size())
            {
                error = "scene hierarchy contains a cycle";
                return false;
            }
            const auto* parent_links = state.scene.try_get<scene::hierarchy_component>(ancestor);
            ancestor = parent_links ? parent_links->parent : scene::entity{};
        }
        if (state.scene.has<scene::world_environment_component>(value) &&
            !scene::read_world_environment_settings(state.scene, value))
        {
            error = "world environment is incomplete";
            return false;
        }
    }
    for (const auto& binding : state.asset_bindings)
    {
        if (!is_normal_project_relative_path(binding.source_path, project_root) ||
            !is_normal_project_relative_path(binding.material_path, project_root))
        {
            error = "scene asset references must be project-relative";
            return false;
        }
    }
    if (ordered_entities(state).size() != persisted_count)
    {
        error = "scene hierarchy ordering is invalid";
        return false;
    }
    return true;
}

json component_version() { return json{ { "version", 1 } }; }

std::vector<scene::entity> ordered_entities(const editor_scene_state& state)
{
    std::vector<scene::entity> result;
    std::unordered_set<scene::entity, ecs::entity_hash> visited;
    const auto append = [&](auto&& self, scene::entity value) -> void {
        if (!visited.insert(value).second)
            return;
        result.push_back(value);
        for (const auto child : scene::children(state.scene, value))
            self(self, child);
    };
    for (const auto value : scene::roots(state.scene))
    {
        if (value == state.camera_entity)
            continue;
        append(append, value);
    }
    return result;
}

json serialize_light_common(const math::vector3f& color, float intensity, bool shadows, bool enabled,
    bool use_temperature, float temperature, render::light_intensity_unit unit, const render::shadow_settings& shadow)
{
    return {
        { "version", 2 }, { "color", vector3(color) }, { "intensity", intensity },
        { "castsShadows", shadows }, { "enabled", enabled }, { "useColorTemperature", use_temperature },
        { "temperatureKelvin", temperature }, { "intensityUnit", static_cast<int>(unit) },
        { "shadow", { { "enabled", shadow.enabled }, { "resolution", shadow.resolution },
            { "bias", shadow.bias }, { "normalBias", shadow.normal_bias }, { "strength", shadow.strength },
            { "filter", static_cast<int>(shadow.filter) } } }
    };
}

void deserialize_light_common(const json& source, math::vector3f& color, float& intensity, bool& shadows,
    bool& enabled, bool& use_temperature, float& temperature, render::light_intensity_unit& unit,
    render::shadow_settings& shadow)
{
    color = read_vector3(source.at("color"));
    intensity = source.at("intensity").get<float>();
    shadows = source.value("castsShadows", false);
    enabled = source.value("enabled", true);
    use_temperature = source.value("useColorTemperature", false);
    temperature = source.value("temperatureKelvin", 6500.0f);
    unit = static_cast<render::light_intensity_unit>(source.value("intensityUnit", 0));
    if (const auto found = source.find("shadow"); found != source.end())
    {
        shadow.enabled = found->value("enabled", shadows);
        shadow.resolution = found->value("resolution", 2048u);
        shadow.bias = found->value("bias", 0.0015f);
        shadow.normal_bias = found->value("normalBias", 0.01f);
        shadow.strength = found->value("strength", 0.75f);
        shadow.filter = static_cast<render::shadow_filter>(found->value("filter", 1));
    }
}

json serialize_entity(const editor_scene_state& state, scene::entity value, const std::filesystem::path& project_root)
{
    const auto guid = entity_guid_of(state, value);
    json output{
        { "id", scene::to_string(guid) },
        { "parent", nullptr },
        { "components", json::object() }
    };
    if (const auto* hierarchy = state.scene.try_get<scene::hierarchy_component>(value); hierarchy && state.scene.alive(hierarchy->parent))
        output["parent"] = scene::to_string(entity_guid_of(state, hierarchy->parent));
    auto& components = output["components"];
    if (const auto* component = state.scene.try_get<scene::name_component>(value))
        components["Name"] = { { "version", 1 }, { "value", component->value } };
    if (const auto* component = state.scene.try_get<scene::tag_component>(value))
        components["Tag"] = { { "version", 1 }, { "value", component->value } };
    if (const auto* component = state.scene.try_get<scene::active_component>(value))
        components["Active"] = { { "version", 1 }, { "value", component->active } };
    if (const auto* component = state.scene.try_get<scene::render_layer_component>(value))
        components["RenderLayer"] = { { "version", 1 }, { "mask", component->mask } };
    if (const auto* component = state.scene.try_get<scene::world_region_component>(value))
        components["WorldRegion"] = {
            { "version", 1 }, { "id", scene::to_string(component->region.value) }
        };
    if (const auto* component = state.scene.try_get<scene::prefab_instance_component>(value))
    {
        json mapping = json::array();
        for (const auto& [source, instance] : component->source_to_instance)
            mapping.push_back({
                { "source", scene::to_string(source) },
                { "instance", scene::to_string(instance) }
            });
        json overrides = json::array();
        for (const auto& override_value : component->overrides)
        {
            const auto bytes = std::span<const std::uint8_t>(
                reinterpret_cast<const std::uint8_t*>(override_value.value.data()),
                override_value.value.size());
            overrides.push_back({
                { "source", scene::to_string(override_value.key.source_entity) },
                { "component", ecs::to_string(override_value.key.component) },
                { "field", override_value.key.field },
                { "kind", static_cast<std::uint32_t>(override_value.key.kind) },
                { "value", base64_encode(bytes) }
            });
        }
        components["PrefabInstance"] = {
            { "version", 1 },
            { "prefabGuid", scene::to_string(component->prefab_guid) },
            { "prefabPath", relative_asset_path(component->prefab_path, project_root).generic_string() },
            { "sourceRoot", scene::to_string(component->source_root) },
            { "mapping", std::move(mapping) },
            { "overrides", std::move(overrides) }
        };
    }
    if (const auto* component = state.scene.try_get<scene::transform_component>(value))
        components["Transform"] = { { "version", 1 }, { "position", vector3(component->position) },
            { "rotation", quaternion(component->rotation) }, { "scale", vector3(component->scale) } };
    if (const auto* component = state.scene.try_get<scene::camera_component>(value))
        components["Camera"] = { { "version", 2 }, { "projection", static_cast<int>(component->projection) },
            { "fovY", component->fov_y_radians }, { "near", component->near_plane }, { "far", component->far_plane },
            { "orthographicHeight", component->orthographic_height }, { "active", component->active },
            { "clearColor", vector4(component->clear_color) },
            { "exposure", {
                { "mode", static_cast<int>(component->exposure.mode) },
                { "metering", static_cast<int>(component->exposure.metering) },
                { "manualEV100", component->exposure.manual_ev100 },
                { "compensationEV", component->exposure.compensation_ev },
                { "minimumEV100", component->exposure.minimum_ev100 },
                { "maximumEV100", component->exposure.maximum_ev100 },
                { "brightenSpeed", component->exposure.brighten_speed },
                { "darkenSpeed", component->exposure.darken_speed }
            } } };
    if (const auto* component = state.scene.try_get<scene::mesh_renderer_component>(value))
        components["MeshRenderer"] = { { "version", 1 }, { "visible", component->visible },
            { "baseColorTint", vector4(component->base_color_tint) } };
    if (const auto* component = state.scene.try_get<scene::directional_light_component>(value))
        components["DirectionalLight"] = serialize_light_common(component->color, component->intensity,
            component->casts_shadows, component->enabled, component->use_color_temperature,
            component->temperature_kelvin, component->intensity_unit, component->shadow);
    if (const auto* component = state.scene.try_get<scene::point_light_component>(value))
    {
        components["PointLight"] = serialize_light_common(component->color, component->intensity,
            component->casts_shadows, component->enabled, component->use_color_temperature,
            component->temperature_kelvin, component->intensity_unit, component->shadow);
        components["PointLight"]["range"] = component->range;
    }
    if (const auto* component = state.scene.try_get<scene::spot_light_component>(value))
    {
        components["SpotLight"] = serialize_light_common(component->color, component->intensity,
            component->casts_shadows, component->enabled, component->use_color_temperature,
            component->temperature_kelvin, component->intensity_unit, component->shadow);
        components["SpotLight"]["range"] = component->range;
        components["SpotLight"]["innerAngle"] = component->inner_angle;
        components["SpotLight"]["outerAngle"] = component->outer_angle;
    }
    if (const auto* component = state.scene.try_get<scene::area_light_component>(value))
    {
        components["AreaLight"] = serialize_light_common(component->color, component->intensity,
            component->casts_shadows, component->enabled, component->use_color_temperature,
            component->temperature_kelvin, component->intensity_unit, component->shadow);
        components["AreaLight"]["width"] = component->width;
        components["AreaLight"]["height"] = component->height;
        components["AreaLight"]["shape"] = static_cast<int>(component->shape);
        components["AreaLight"]["twoSided"] = component->two_sided;
    }
    if (state.scene.has<scene::world_environment_component>(value))
    {
        const auto settings = scene::read_world_environment_settings(state.scene, value);
        if (!settings)
            return output;
        auto snapshot = to_host_world_environment_snapshot({}, *settings, state.world_environment_hdri_path);
        components["WorldEnvironment"] = json::parse(to_json(snapshot));
        components["WorldEnvironment"]["version"] = 1;
        if (state.scene.alive(settings->celestial.sun_light))
            components["WorldEnvironment"]["sunLight"] = scene::to_string(entity_guid_of(state, settings->celestial.sun_light));
    }
    if (const auto* component = state.scene.try_get<scene::terrain_component>(value))
    {
        if (!scene::terrain_heightfield_valid(*component))
            throw std::runtime_error("terrain heightfield cannot be serialized because its authored arrays are invalid");
        const auto [minimum, maximum] = std::minmax_element(component->heights.begin(), component->heights.end());
        const float range = *maximum - *minimum;
        std::vector<std::uint8_t> height_bytes(component->heights.size() * sizeof(std::uint16_t));
        std::vector<std::uint8_t> weight_bytes(component->layer_weights.size() * 4u);
        for (std::size_t sample = 0; sample < component->heights.size(); ++sample)
        {
            const float normalized = range > std::numeric_limits<float>::epsilon()
                ? std::clamp((component->heights[sample] - *minimum) / range, 0.0f, 1.0f)
                : 0.0f;
            const auto quantized = static_cast<std::uint16_t>(std::lround(normalized * 65535.0f));
            height_bytes[sample * 2u] = static_cast<std::uint8_t>(quantized & 0xffu);
            height_bytes[sample * 2u + 1u] = static_cast<std::uint8_t>(quantized >> 8u);
            std::copy(component->layer_weights[sample].begin(), component->layer_weights[sample].end(),
                weight_bytes.begin() + static_cast<std::ptrdiff_t>(sample * 4u));
        }
        components["Terrain"] = {
            { "version", 2 }, { "enabled", component->enabled }, { "size", component->size },
            { "subdivisions", component->subdivisions }, { "chunkQuads", component->chunk_quads },
            { "heightScale", component->height_scale }, { "baseColor", vector3(component->base_color) },
            { "receiveShadows", component->receive_shadows }, { "minHeight", *minimum },
            { "heightRange", range }, { "heights", base64_encode(height_bytes) },
            { "weights", base64_encode(weight_bytes) }, { "revision", component->content_revision }
        };
    }
    if (const auto* component = state.scene.try_get<scene::water_component>(value))
        components["Water"] = { { "version", 1 }, { "enabled", component->enabled }, { "size", component->size },
            { "color", vector3(component->color) }, { "roughness", component->roughness },
            { "waveScale", component->wave_scale }, { "waveSpeed", component->wave_speed },
            { "transparency", component->transparency } };
    if (const auto* component = state.scene.try_get<scene::vegetation_component>(value))
        components["Vegetation"] = { { "version", 1 }, { "enabled", component->enabled }, { "density", component->density },
            { "patchSize", component->patch_size }, { "color", vector3(component->color) },
            { "windStrength", component->wind_strength }, { "windSpeed", component->wind_speed } };
    if (const auto* component = state.scene.try_get<scene::decal_component>(value))
        components["Decal"] = { { "version", 1 }, { "enabled", component->enabled },
            { "color", vector4(component->color) }, { "opacity", component->opacity } };

    if (const auto* binding = find_asset_binding(state, guid))
        output["assetBinding"] = { { "kind", binding->source_kind },
            { "path", relative_asset_path(binding->source_path, project_root).generic_string() },
            { "subresource", binding->subresource },
            { "material", relative_asset_path(binding->material_path, project_root).generic_string() } };
    const auto unknown = std::find_if(state.unknown_component_records.begin(), state.unknown_component_records.end(),
        [guid](const auto& entry) { return entry.first == guid; });
    if (unknown != state.unknown_component_records.end())
    {
        const auto unknown_json = json::parse(unknown->second, nullptr, false);
        if (unknown_json.is_object())
            components.update(unknown_json);
    }
    return output;
}

bool replace_file_atomically(const std::filesystem::path& temporary, const std::filesystem::path& target, std::string& error)
{
#if defined(_WIN32)
    const BOOL replaced = MoveFileExW(
        temporary.c_str(), target.c_str(), MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH);
    if (!replaced)
    {
        error = "atomic scene replacement failed with Win32 error " + std::to_string(GetLastError());
        return false;
    }
#else
    std::error_code move_error;
    std::filesystem::rename(temporary, target, move_error);
    if (move_error)
    {
        error = move_error.message();
        return false;
    }
#endif
    return true;
}

editor_primitive_type primitive_from_name(std::string_view value)
{
    if (value == "Cube") return editor_primitive_type::cube;
    if (value == "Sphere") return editor_primitive_type::sphere;
    if (value == "Cylinder") return editor_primitive_type::cylinder;
    return editor_primitive_type::plane;
}

} // namespace

scene_document_text_result serialize_scene_subtree_as_prefab(
    editor_scene_state& state,
    const std::filesystem::path& project_root,
    scene::entity root,
    scene::entity_guid prefab_guid,
    std::string_view prefab_name)
{
    ensure_scene_authoring_metadata(state);
    if (!state.scene.alive(root) || root == state.camera_entity)
        return { .message = "prefab root is missing or editor-only" };
    if (!prefab_guid.valid())
        return { .message = "prefab GUID is invalid" };

    const auto values = scene::subtree(state.scene, root);
    std::unordered_set<scene::entity, ecs::entity_hash> included(values.begin(), values.end());
    json document{
        { "format", "arc.prefab" },
        { "formatVersion", ecs::prefab_asset::current_format_version },
        { "prefab", {
            { "id", scene::to_string(prefab_guid) },
            { "name", std::string(prefab_name) },
            { "root", scene::to_string(entity_guid_of(state, root)) }
        } },
        { "entities", json::array() }
    };
    try
    {
        for (const scene::entity value : values)
        {
            json record = serialize_entity(state, value, project_root);
            record["components"].erase("PrefabInstance");
            const auto* hierarchy = state.scene.try_get<scene::hierarchy_component>(value);
            if (!hierarchy || !included.contains(hierarchy->parent))
                record["parent"] = nullptr;
            document["entities"].push_back(std::move(record));
        }
    }
    catch (const std::exception& error)
    {
        return { .message = std::string("prefab serialization failed: ") + error.what() };
    }
    return {
        .succeeded = true,
        .entity_count = values.size(),
        .text = document.dump(2) + '\n',
        .message = "Prefab serialized"
    };
}

scene_document_result save_scene_document(
    editor_scene_state& state,
    const std::filesystem::path& project_root,
    const std::filesystem::path& path)
{
    ensure_scene_authoring_metadata(state);
    if (path.empty())
        return { .message = "scene save path is empty" };
    std::string validation_error;
    if (!validate_scene_for_save(state, project_root, validation_error))
        return { .message = std::move(validation_error) };
    const auto saved_scene_name = path.stem().string();
    json document{
        { "format", "arc.scene" }, { "formatVersion", arc_scene_format_version },
        { "scene", { { "id", scene::to_string(state.scene_guid) }, { "name", saved_scene_name } } },
        { "entities", json::array() }
    };
    try
    {
        for (const auto entity : ordered_entities(state))
            document["entities"].push_back(serialize_entity(state, entity, project_root));
    }
    catch (const std::exception& error)
    {
        return { .message = std::string("scene serialization failed: ") + error.what() };
    }

    std::error_code directory_error;
    if (!path.parent_path().empty())
        std::filesystem::create_directories(path.parent_path(), directory_error);
    if (directory_error)
        return { .message = "could not create scene directory: " + directory_error.message() };
    const auto temporary = path.parent_path() / (path.filename().string() + ".tmp");
    {
        std::ofstream stream(temporary, std::ios::binary | std::ios::trunc);
        if (!stream)
            return { .message = "could not open temporary scene file" };
        stream << document.dump(2) << '\n';
        stream.flush();
        if (!stream)
            return { .message = "failed while writing temporary scene file" };
    }
    std::string replacement_error;
    if (!replace_file_atomically(temporary, path, replacement_error))
    {
        std::error_code ignored;
        std::filesystem::remove(temporary, ignored);
        return { .message = replacement_error };
    }
    state.active_scene_path = path;
    state.scene_name = saved_scene_name;
    return { .succeeded = true, .entity_count = document["entities"].size(), .message = "Scene saved" };
}

scene_document_result load_scene_document(
    editor_scene_state& state,
    render::renderer& renderer,
    const std::filesystem::path& project_root,
    const std::filesystem::path& path)
{
    std::ifstream stream(path, std::ios::binary);
    if (!stream)
        return { .message = "could not open scene file" };
    json document;
    try { stream >> document; }
    catch (const std::exception& error) { return { .message = std::string("invalid scene JSON: ") + error.what() }; }
    if (!document.is_object() || !document.contains("format") || !document["format"].is_string() ||
        !document.contains("formatVersion") || !document["formatVersion"].is_number_unsigned() ||
        document["format"].get<std::string>() != "arc.scene" ||
        document["formatVersion"].get<std::uint32_t>() != arc_scene_format_version)
        return { .message = "unsupported ARC scene format or version" };
    if (!document.contains("scene") || !document["scene"].is_object() || !document.contains("entities") || !document["entities"].is_array())
        return { .message = "scene document is missing required objects" };
    if (!document["scene"].contains("id") || !document["scene"]["id"].is_string() ||
        (document["scene"].contains("name") && !document["scene"]["name"].is_string()))
        return { .message = "scene document has invalid scene metadata" };
    const auto scene_id = scene::parse_entity_guid(document["scene"]["id"].get<std::string>());
    if (!scene_id)
        return { .message = "scene document has an invalid scene GUID" };

    std::unordered_set<std::string> ids;
    std::unordered_map<std::string, std::string> parent_ids;
    std::vector<std::string> sun_references;
    for (const auto& record : document["entities"])
    {
        if (!record.is_object() || !record.contains("id") || !record["id"].is_string() ||
            !record.contains("parent") || !record.contains("components") || !record["components"].is_object())
            return { .message = "scene contains a malformed entity record" };
        const auto id = record["id"].get<std::string>();
        if (!scene::parse_entity_guid(id) || !ids.insert(id).second)
            return { .message = "scene contains an invalid/duplicate entity or component object" };
        if (!record["parent"].is_null() && !record["parent"].is_string())
            return { .message = "scene contains an invalid parent reference" };
        parent_ids.emplace(id, record["parent"].is_string() ? record["parent"].get<std::string>() : std::string{});
        if (record.contains("assetBinding"))
        {
            const auto& binding = record["assetBinding"];
            if (!binding.is_object() ||
                (binding.contains("kind") && !binding["kind"].is_string()) ||
                (binding.contains("path") && !binding["path"].is_string()) ||
                (binding.contains("subresource") && !binding["subresource"].is_string()) ||
                (binding.contains("material") && !binding["material"].is_string()))
                return { .message = "scene contains an invalid asset binding" };
            const auto source = resolve_document_asset_path(binding.value("path", ""), project_root);
            const auto material = resolve_document_asset_path(binding.value("material", ""), project_root);
            if (!source || !material)
                return { .message = "asset binding paths must be normalized project-relative paths" };
        }
        for (const auto& [component_name, component] : record["components"].items())
        {
            std::string validation_error;
            if (!validate_component_json(component_name, component, validation_error))
                return { .message = std::move(validation_error) };
            if (component_name == "WorldEnvironment" && component.contains("sunLight"))
            {
                if (!component["sunLight"].is_string())
                    return { .message = "world environment has an invalid sun reference" };
                sun_references.push_back(component["sunLight"].get<std::string>());
            }
        }
    }
    for (const auto& record : document["entities"])
        if (!record["parent"].is_null() && !ids.contains(record["parent"].get<std::string>()))
            return { .message = "scene contains a missing parent reference" };
    for (const auto& [id, _] : parent_ids)
    {
        std::unordered_set<std::string> ancestors;
        auto current = id;
        while (!current.empty())
        {
            if (!ancestors.insert(current).second)
                return { .message = "scene hierarchy contains a cycle" };
            const auto found = parent_ids.find(current);
            current = found != parent_ids.end() ? found->second : std::string{};
        }
    }
    if (std::any_of(sun_references.begin(), sun_references.end(), [&](const auto& reference) {
        return !ids.contains(reference);
    }))
        return { .message = "world environment references a missing sun" };

    editor_scene_state loaded = state;
    const auto old_editor_transform = state.scene.try_get<scene::transform_component>(state.camera_entity)
        ? *state.scene.try_get<scene::transform_component>(state.camera_entity) : scene::transform_component{};
    const auto old_editor_camera = state.scene.try_get<scene::camera_component>(state.camera_entity)
        ? *state.scene.try_get<scene::camera_component>(state.camera_entity) : scene::camera_component{};
    loaded.scene = scene::registry{};
    loaded.selected_entity = {};
    loaded.game_camera_entity = {};
    loaded.sun_entity = {};
    loaded.world_environment_entity = {};
    loaded.mesh_entity = {};
    loaded.terrain_entity = {};
    loaded.water_entity = {};
    loaded.vegetation_entity = {};
    loaded.primitive_entities.clear();
    loaded.imported_scene_entities.clear();
    loaded.world_feature_entities.clear();
    loaded.asset_bindings.clear();
    loaded.unknown_component_records.clear();
    loaded.scene_guid = *scene_id;
    loaded.scene_name = document["scene"].value("name", path.stem().string());

    loaded.camera_entity = loaded.scene.create();
    loaded.scene.emplace<scene::persistent_id_component>(loaded.camera_entity, scene::generate_entity_guid());
    loaded.scene.emplace<scene::hierarchy_component>(loaded.camera_entity);
    loaded.scene.emplace<scene::name_component>(loaded.camera_entity, "Editor Camera");
    loaded.scene.emplace<scene::tag_component>(loaded.camera_entity, "EditorOnly");
    loaded.scene.emplace<scene::active_component>(loaded.camera_entity);
    loaded.scene.emplace<scene::transform_component>(loaded.camera_entity, old_editor_transform);
    loaded.scene.emplace<scene::camera_component>(loaded.camera_entity, old_editor_camera);

    std::unordered_map<std::string, scene::entity> entities;
    std::vector<std::pair<scene::entity, std::string>> pending_parents;
    std::vector<std::pair<scene::entity, std::string>> pending_suns;
    std::vector<std::string> diagnostics;
    try
    {
        for (const auto& record : document["entities"])
        {
            const auto id_text = record.at("id").get<std::string>();
            const auto guid = *scene::parse_entity_guid(id_text);
            const auto& components = record.at("components");
            for (const auto& [component_name, component] : components.items())
                if (!component.is_object() || !component.contains("version") || !component["version"].is_number_unsigned())
                    throw std::runtime_error("component '" + component_name + "' has no valid schema version");
            const auto binding_json = record.value("assetBinding", json::object());
            const std::string binding_kind = binding_json.value("kind", "");
            scene::entity entity{};
            if (binding_kind == "primitive")
                entity = add_primitive_to_scene(loaded, renderer, primitive_from_name(binding_json.value("subresource", "Plane")));
            else if (components.contains("Terrain"))
                entity = add_terrain_to_scene(loaded, renderer);
            else if (components.contains("Water"))
                entity = add_water_to_scene(loaded, renderer);
            else if (components.contains("Vegetation"))
                entity = add_grass_patch_to_scene(loaded, renderer);
            else if (components.contains("Decal"))
                entity = add_decal_to_scene(loaded);
            else if (components.contains("WorldEnvironment"))
                entity = add_world_environment_to_scene(loaded);
            else
                entity = loaded.scene.create();
            if (!loaded.scene.alive(entity))
                throw std::runtime_error("could not create scene entity");
            const auto generated_guid = entity_guid_of(loaded, entity);
            loaded.asset_bindings.erase(std::remove_if(loaded.asset_bindings.begin(), loaded.asset_bindings.end(),
                [generated_guid](const auto& value) { return value.entity == generated_guid; }), loaded.asset_bindings.end());
            if (loaded.scene.has<scene::persistent_id_component>(entity))
                loaded.scene.get<scene::persistent_id_component>(entity).value = guid;
            else
                loaded.scene.emplace<scene::persistent_id_component>(entity, guid);
            if (!loaded.scene.has<scene::hierarchy_component>(entity))
                loaded.scene.emplace<scene::hierarchy_component>(entity);
            entities.emplace(id_text, entity);

            if (components.contains("Name")) loaded.scene.emplace<scene::name_component>(entity, components["Name"].value("value", "Entity"));
            if (components.contains("Tag")) loaded.scene.emplace<scene::tag_component>(entity, components["Tag"].value("value", "Untagged"));
            if (components.contains("Active")) loaded.scene.emplace<scene::active_component>(entity, components["Active"].value("value", true));
            if (components.contains("RenderLayer")) loaded.scene.emplace<scene::render_layer_component>(entity, components["RenderLayer"].value("mask", 1u));
            if (components.contains("WorldRegion"))
            {
                const auto region = scene::parse_entity_guid(components["WorldRegion"].value("id", ""));
                if (!region) throw std::runtime_error("invalid world region identity");
                loaded.scene.emplace<scene::world_region_component>(entity, ecs::world_region_id{ *region });
            }
            if (components.contains("PrefabInstance"))
            {
                const auto& source = components["PrefabInstance"];
                scene::prefab_instance_component instance;
                instance.prefab_guid = *scene::parse_entity_guid(source.value("prefabGuid", ""));
                instance.source_root = *scene::parse_entity_guid(source.value("sourceRoot", ""));
                const auto prefab_path = resolve_document_asset_path(source.value("prefabPath", ""), project_root);
                if (!prefab_path) throw std::runtime_error("invalid prefab asset path");
                instance.prefab_path = prefab_path->generic_string();
                for (const auto& mapping : source["mapping"])
                {
                    if (!mapping.is_object() || !mapping.contains("source") || !mapping.contains("instance"))
                        throw std::runtime_error("invalid prefab entity mapping");
                    const auto source_id = scene::parse_entity_guid(mapping["source"].get<std::string>());
                    const auto instance_id = scene::parse_entity_guid(mapping["instance"].get<std::string>());
                    if (!source_id || !instance_id) throw std::runtime_error("invalid prefab mapped entity GUID");
                    instance.source_to_instance.emplace_back(*source_id, *instance_id);
                }
                for (const auto& stored : source["overrides"])
                {
                    if (!stored.is_object() || !stored.contains("source") || !stored.contains("component") ||
                        !stored.contains("field") || !stored.contains("kind") || !stored.contains("value"))
                        throw std::runtime_error("invalid prefab override");
                    const auto source_id = scene::parse_entity_guid(stored["source"].get<std::string>());
                    const auto component_id = ecs::parse_component_type_id(stored["component"].get<std::string>());
                    const auto bytes = base64_decode(stored["value"].get<std::string>());
                    const auto kind = stored["kind"].get<std::uint32_t>();
                    if (!source_id || !component_id || !bytes ||
                        kind > static_cast<std::uint32_t>(ecs::prefab_override_kind::instance_child_added))
                        throw std::runtime_error("invalid prefab override identity or payload");
                    ecs::prefab_override override_value;
                    override_value.key = {
                        *source_id, *component_id, stored["field"].get<std::uint64_t>(),
                        static_cast<ecs::prefab_override_kind>(kind)
                    };
                    override_value.value.resize(bytes->size());
                    std::memcpy(override_value.value.data(), bytes->data(), bytes->size());
                    instance.overrides.push_back(std::move(override_value));
                }
                loaded.scene.emplace<scene::prefab_instance_component>(entity, std::move(instance));
            }
            if (components.contains("Transform"))
            {
                const auto& value = components["Transform"];
                if (!finite_array(value.at("position"), 3) || !finite_array(value.at("rotation"), 4) || !finite_array(value.at("scale"), 3))
                    throw std::runtime_error("invalid transform values");
                scene::transform_component transform;
                transform.position = read_vector3(value.at("position"));
                transform.rotation = read_quaternion(value.at("rotation"));
                transform.scale = read_vector3(value.at("scale"));
                loaded.scene.emplace<scene::transform_component>(entity, transform);
            }
            if (components.contains("Camera"))
            {
                const auto& value = components["Camera"];
                scene::camera_component camera;
                camera.projection = static_cast<scene::camera_projection>(value.value("projection", 0));
                camera.fov_y_radians = value.value("fovY", camera.fov_y_radians);
                camera.near_plane = value.value("near", camera.near_plane);
                camera.far_plane = value.value("far", camera.far_plane);
                camera.orthographic_height = value.value("orthographicHeight", camera.orthographic_height);
                camera.active = value.value("active", true);
                camera.clear_color = read_vector4(value.at("clearColor"));
                if (value.contains("exposure"))
                {
                    const auto& exposure = value["exposure"];
                    camera.exposure.mode = static_cast<render::exposure_mode>(
                        exposure.value("mode", static_cast<int>(camera.exposure.mode)));
                    camera.exposure.metering = static_cast<render::exposure_metering_mode>(
                        exposure.value("metering", static_cast<int>(camera.exposure.metering)));
                    camera.exposure.manual_ev100 = exposure.value("manualEV100", camera.exposure.manual_ev100);
                    camera.exposure.compensation_ev = exposure.value("compensationEV", camera.exposure.compensation_ev);
                    camera.exposure.minimum_ev100 = exposure.value("minimumEV100", camera.exposure.minimum_ev100);
                    camera.exposure.maximum_ev100 = exposure.value("maximumEV100", camera.exposure.maximum_ev100);
                    camera.exposure.brighten_speed = exposure.value("brightenSpeed", camera.exposure.brighten_speed);
                    camera.exposure.darken_speed = exposure.value("darkenSpeed", camera.exposure.darken_speed);
                }
                if (!(camera.near_plane > 0.0f && camera.far_plane > camera.near_plane))
                    throw std::runtime_error("invalid camera clip planes");
                loaded.scene.emplace<scene::camera_component>(entity, camera);
                if (!loaded.game_camera_entity.valid()) loaded.game_camera_entity = entity;
            }
            if (components.contains("MeshRenderer") && !loaded.scene.has<scene::mesh_renderer_component>(entity))
            {
                const auto& value = components["MeshRenderer"];
                loaded.scene.emplace<scene::mesh_renderer_component>(entity, loaded.default_mesh, loaded.default_material,
                    value.value("visible", true), read_vector4(value.at("baseColorTint")));
                diagnostics.push_back("Mesh asset for '" + id_text + "' used the default fallback");
            }
            else if (components.contains("MeshRenderer"))
            {
                auto& mesh = loaded.scene.get<scene::mesh_renderer_component>(entity);
                mesh.visible = components["MeshRenderer"].value("visible", true);
                mesh.base_color_tint = read_vector4(components["MeshRenderer"].at("baseColorTint"));
            }
            if (components.contains("DirectionalLight"))
            {
                scene::directional_light_component light;
                deserialize_light_common(components["DirectionalLight"], light.color, light.intensity, light.casts_shadows,
                    light.enabled, light.use_color_temperature, light.temperature_kelvin, light.intensity_unit, light.shadow);
                if (components["DirectionalLight"].value("version", 1u) < 2u &&
                    light.intensity_unit == render::light_intensity_unit::unitless)
                {
                    light.intensity *= 20000.0f;
                    light.intensity_unit = render::light_intensity_unit::lux;
                }
                loaded.scene.emplace<scene::directional_light_component>(entity, light);
                if (!loaded.sun_entity.valid()) loaded.sun_entity = entity;
            }
            if (components.contains("PointLight"))
            {
                scene::point_light_component light;
                deserialize_light_common(components["PointLight"], light.color, light.intensity, light.casts_shadows,
                    light.enabled, light.use_color_temperature, light.temperature_kelvin, light.intensity_unit, light.shadow);
                if (components["PointLight"].value("version", 1u) < 2u &&
                    light.intensity_unit == render::light_intensity_unit::unitless)
                {
                    light.intensity *= 1000.0f;
                    light.intensity_unit = render::light_intensity_unit::lumen;
                }
                light.range = components["PointLight"].value("range", light.range);
                loaded.scene.emplace<scene::point_light_component>(entity, light);
            }
            if (components.contains("SpotLight"))
            {
                scene::spot_light_component light;
                deserialize_light_common(components["SpotLight"], light.color, light.intensity, light.casts_shadows,
                    light.enabled, light.use_color_temperature, light.temperature_kelvin, light.intensity_unit, light.shadow);
                if (components["SpotLight"].value("version", 1u) < 2u &&
                    light.intensity_unit == render::light_intensity_unit::unitless)
                {
                    light.intensity *= 1000.0f;
                    light.intensity_unit = render::light_intensity_unit::lumen;
                }
                light.range = components["SpotLight"].value("range", light.range);
                light.inner_angle = components["SpotLight"].value("innerAngle", light.inner_angle);
                light.outer_angle = components["SpotLight"].value("outerAngle", light.outer_angle);
                loaded.scene.emplace<scene::spot_light_component>(entity, light);
            }
            if (components.contains("AreaLight"))
            {
                scene::area_light_component light;
                deserialize_light_common(components["AreaLight"], light.color, light.intensity, light.casts_shadows,
                    light.enabled, light.use_color_temperature, light.temperature_kelvin, light.intensity_unit, light.shadow);
                light.width = components["AreaLight"].value("width", light.width);
                light.height = components["AreaLight"].value("height", light.height);
                light.shape = static_cast<render::area_light_shape>(
                    components["AreaLight"].value("shape", static_cast<int>(light.shape)));
                light.two_sided = components["AreaLight"].value("twoSided", false);
                loaded.scene.emplace<scene::area_light_component>(entity, light);
            }
            if (components.contains("WorldEnvironment"))
            {
                host_world_environment_snapshot snapshot;
                std::string parse_error;
                if (!from_json(components["WorldEnvironment"].dump(), snapshot, parse_error))
                    throw std::runtime_error(parse_error);
                const auto current = scene::read_world_environment_settings(loaded.scene, entity);
                if (!current) throw std::runtime_error("incomplete world environment components");
                scene::set_world_environment_settings(loaded.scene, entity, apply_host_world_environment_snapshot(snapshot, *current));
                loaded.world_environment_entity = entity;
                loaded.world_environment_hdri_path = snapshot.hdri_path;
                if (components["WorldEnvironment"].contains("sunLight"))
                    pending_suns.emplace_back(entity, components["WorldEnvironment"]["sunLight"].get<std::string>());
            }
            if (components.contains("Terrain"))
            {
                auto& value = loaded.scene.get<scene::terrain_component>(entity);
                const auto& source = components["Terrain"];
                value.enabled = source.value("enabled", true); value.size = source.value("size", value.size);
                value.subdivisions = source.value("subdivisions", value.subdivisions);
                value.chunk_quads = source.value("chunkQuads", scene::default_terrain_chunk_quads);
                value.height_scale = source.value("heightScale", value.height_scale);
                value.base_color = read_vector3(source.at("baseColor")); value.receive_shadows = source.value("receiveShadows", true);
                if (source.value("version", 1u) == 1u)
                {
                    scene::generate_terrain_heightfield(value);
                }
                else
                {
                    const auto heights = base64_decode(source.at("heights").get<std::string>());
                    const auto weights = base64_decode(source.at("weights").get<std::string>());
                    if (!heights || !weights)
                        throw std::runtime_error("terrain payload failed base64 decoding");
                    const auto sample_count = static_cast<std::size_t>(value.subdivisions + 1u) * (value.subdivisions + 1u);
                    const float minimum = source.at("minHeight").get<float>();
                    const float range = source.at("heightRange").get<float>();
                    value.heights.resize(sample_count);
                    value.layer_weights.resize(sample_count);
                    for (std::size_t sample = 0; sample < sample_count; ++sample)
                    {
                        const auto quantized = static_cast<std::uint16_t>((*heights)[sample * 2u]) |
                            (static_cast<std::uint16_t>((*heights)[sample * 2u + 1u]) << 8u);
                        value.heights[sample] = minimum + (static_cast<float>(quantized) / 65535.0f) * range;
                        std::copy_n(weights->begin() + static_cast<std::ptrdiff_t>(sample * 4u), 4u,
                            value.layer_weights[sample].begin());
                    }
                    value.content_revision = source.value("revision", std::uint64_t{ 1 });
                }
                if (!rebuild_terrain_chunks(loaded, renderer, entity))
                    throw std::runtime_error("terrain runtime chunks could not be rebuilt");
            }
            if (components.contains("Water"))
            {
                auto& value = loaded.scene.get<scene::water_component>(entity); const auto& source = components["Water"];
                value.enabled = source.value("enabled", true); value.size = source.value("size", value.size);
                value.color = read_vector3(source.at("color")); value.roughness = source.value("roughness", value.roughness);
                value.wave_scale = source.value("waveScale", value.wave_scale); value.wave_speed = source.value("waveSpeed", value.wave_speed);
                value.transparency = source.value("transparency", value.transparency);
            }
            if (components.contains("Vegetation"))
            {
                auto& value = loaded.scene.get<scene::vegetation_component>(entity); const auto& source = components["Vegetation"];
                value.enabled = source.value("enabled", true); value.density = source.value("density", value.density);
                value.patch_size = source.value("patchSize", value.patch_size); value.color = read_vector3(source.at("color"));
                value.wind_strength = source.value("windStrength", value.wind_strength); value.wind_speed = source.value("windSpeed", value.wind_speed);
            }
            if (components.contains("Decal"))
            {
                auto& value = loaded.scene.get<scene::decal_component>(entity); const auto& source = components["Decal"];
                value.enabled = source.value("enabled", true); value.color = read_vector4(source.at("color"));
                value.opacity = source.value("opacity", value.opacity);
            }

            static const std::unordered_set<std::string> known{ "Name", "Tag", "Active", "RenderLayer", "Transform", "Camera",
                "MeshRenderer", "DirectionalLight", "PointLight", "SpotLight", "AreaLight", "WorldEnvironment", "Terrain", "Water",
                "Vegetation", "Decal", "PrefabInstance", "WorldRegion" };
            json unknown = json::object();
            for (const auto& [name, value] : components.items())
                if (!known.contains(name)) unknown[name] = value;
            if (!unknown.empty()) loaded.unknown_component_records.emplace_back(guid, unknown.dump());

            if (!binding_kind.empty())
            {
                const auto source_text = binding_json.value("path", "");
                const auto material_text = binding_json.value("material", "");
                const auto source_path = resolve_document_asset_path(source_text, project_root);
                const auto material_path = resolve_document_asset_path(material_text, project_root);
                if (!source_path || !material_path)
                    throw std::runtime_error("asset binding paths must be normalized project-relative paths");
                loaded.asset_bindings.push_back({ guid, binding_kind, *source_path,
                    binding_json.value("subresource", ""), *material_path });
                if (!material_path->empty() && loaded.scene.has<scene::mesh_renderer_component>(entity))
                {
                    std::string material_message;
                    if (!apply_material_asset_to_entity(loaded.material_library, renderer, project_root / "assets",
                            *material_path, loaded.scene, entity, &material_message))
                        diagnostics.push_back("Material '" + material_text + "' is missing; using fallback");
                }
            }
            if (!record["parent"].is_null()) pending_parents.emplace_back(entity, record["parent"].get<std::string>());
        }
        for (const auto& [child, parent_id] : pending_parents)
            if (!scene::reparent(loaded.scene, child, entities.at(parent_id), {}, scene::reparent_transform_policy::preserve_local))
                throw std::runtime_error("scene hierarchy contains a cycle");
        for (const auto& [environment, sun_id] : pending_suns)
        {
            const auto found = entities.find(sun_id);
            if (found == entities.end()) throw std::runtime_error("world environment references a missing sun");
            loaded.scene.get<scene::celestial_sky_component>(environment).sun_light = found->second;
        }
    }
    catch (const std::exception& error)
    {
        return { .message = std::string("scene validation failed: ") + error.what(), .diagnostics = std::move(diagnostics) };
    }

    scene::update_world_transforms(loaded.scene);
    for (const auto value : loaded.scene.entities())
    {
        if (value != loaded.camera_entity && loaded.scene.has<scene::name_component>(value))
        {
            select_entity(loaded.scene, value, loaded.selected_entity);
            break;
        }
    }
    loaded.active_scene_path = path;
    state = std::move(loaded);
    return { .succeeded = true, .entity_count = document["entities"].size(), .message = "Scene loaded", .diagnostics = std::move(diagnostics) };
}

} // namespace arc::editor
