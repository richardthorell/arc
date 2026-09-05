#include <arc/editor/procedural_mesh.h>

#include <arc/editor/editor_state.h>
#include <arc/editor/material_library.h>
#include <arc/diagnostics/diagnostics.h>
#include <arc/render/texture.h>
#include <arc/scene/scene.h>

#include <nlohmann/json.hpp>

#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <filesystem>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace arc::editor
{
namespace
{
using json = nlohmann::json;

constexpr std::string_view material_parameter_prefix = "__arc_material_parameter__";
constexpr std::string_view instance_name_marker = "__arc_instance_overrides__";
constexpr std::string_view mesh_renderer_component_name = "MeshRenderer";
constexpr std::string_view persisted_override_field = "materialParameterOverrides";

struct material_parameter_edit
{
    std::string name;
    std::string type;
    std::string kind;
    std::vector<float> value;
    std::string texture;
    bool reset{};
};

struct pending_parameter_edit
{
    editor_scene_state* scene{};
    ecs::entity entity{};
    procedural_mesh_component dummy{};
    std::optional<material_parameter_edit> material;
};

struct runtime_material_instance
{
    editor_scene_state* scene{};
    ecs::entity_guid entity{};
    render::material_handle parent{};
    render::material_handle instance{};
};

thread_local pending_parameter_edit pending_edit;
thread_local std::vector<runtime_material_instance> runtime_instances;

std::optional<std::string> decode_hex(std::string_view hex)
{
    if (hex.empty() || (hex.size() % 2u) != 0u) return std::nullopt;
    const auto nibble = [](char value) -> int
    {
        if (value >= '0' && value <= '9') return value - '0';
        value = static_cast<char>(std::tolower(static_cast<unsigned char>(value)));
        if (value >= 'a' && value <= 'f') return value - 'a' + 10;
        return -1;
    };

    std::string result;
    result.reserve(hex.size() / 2u);
    for (std::size_t offset = 0; offset < hex.size(); offset += 2u)
    {
        const int high = nibble(hex[offset]);
        const int low = nibble(hex[offset + 1u]);
        if (high < 0 || low < 0) return std::nullopt;
        result.push_back(static_cast<char>((high << 4) | low));
    }
    return result;
}

std::string encode_hex(std::string_view text)
{
    constexpr char digits[] = "0123456789abcdef";
    std::string result;
    result.resize(text.size() * 2u);
    for (std::size_t index = 0; index < text.size(); ++index)
    {
        const auto value = static_cast<unsigned char>(text[index]);
        result[index * 2u] = digits[(value >> 4u) & 0xfu];
        result[index * 2u + 1u] = digits[value & 0xfu];
    }
    return result;
}

std::optional<material_parameter_edit> parse_material_parameter(std::string_view parameter)
{
    if (!parameter.starts_with(material_parameter_prefix)) return std::nullopt;
    const auto decoded = decode_hex(parameter.substr(material_parameter_prefix.size()));
    if (!decoded) return std::nullopt;
    const auto payload = json::parse(*decoded, nullptr, false);
    if (!payload.is_object() || !payload.contains("name") || !payload["name"].is_string()) return std::nullopt;

    material_parameter_edit edit;
    edit.name = payload["name"].get<std::string>();
    edit.type = payload.value("type", std::string{});
    edit.kind = payload.value("kind", std::string{});
    edit.texture = payload.value("texture", std::string{});
    edit.reset = payload.value("reset", false);
    if (const auto found = payload.find("value"); found != payload.end())
    {
        if (!found->is_array() || found->size() > 4u) return std::nullopt;
        for (const auto& channel : *found)
        {
            if (!channel.is_number()) return std::nullopt;
            const float value = channel.get<float>();
            if (!std::isfinite(value)) return std::nullopt;
            edit.value.push_back(value);
        }
    }
    return edit.name.empty() ? std::nullopt : std::optional<material_parameter_edit>{std::move(edit)};
}

json persisted_overrides(const editor_scene_state& scene, ecs::entity_guid entity)
{
    for (const auto& preserved : scene.preserved_component_records)
    {
        if (preserved.entity != entity || preserved.component_name != mesh_renderer_component_name) continue;
        const auto component = json::parse(preserved.json, nullptr, false);
        if (!component.is_object()) break;
        const auto found = component.find(std::string(persisted_override_field));
        return found != component.end() && found->is_array() ? *found : json::array();
    }
    return json::array();
}

void store_persisted_overrides(editor_scene_state& scene, ecs::entity_guid entity, const json& overrides)
{
    for (auto& preserved : scene.preserved_component_records)
    {
        if (preserved.entity != entity || preserved.component_name != mesh_renderer_component_name) continue;
        auto component = json::parse(preserved.json, nullptr, false);
        if (!component.is_object()) component = json::object();
        component[std::string(persisted_override_field)] = overrides;
        preserved.json = component.dump();
        return;
    }

    json component = {{"version", 4}, {std::string(persisted_override_field), overrides}};
    scene.preserved_component_records.push_back(
        {.entity = entity, .component_name = std::string(mesh_renderer_component_name), .json = component.dump()});
}

json edit_to_json(const material_parameter_edit& edit)
{
    json value = {{"name", edit.name}, {"type", edit.type}, {"kind", edit.kind}};
    if (!edit.value.empty()) value["value"] = edit.value;
    if (edit.kind == "texture") value["texture"] = edit.texture;
    return value;
}

json apply_edit(json overrides, const material_parameter_edit& edit)
{
    if (!overrides.is_array()) overrides = json::array();
    overrides.erase(std::remove_if(overrides.begin(), overrides.end(), [&](const json& entry)
                                   { return entry.is_object() && entry.value("name", std::string{}) == edit.name; }),
                    overrides.end());
    if (!edit.reset) overrides.push_back(edit_to_json(edit));
    return overrides;
}

bool same_path_suffix(const std::filesystem::path& candidate, std::string_view hint)
{
    if (hint.empty()) return false;
    auto candidate_text = candidate.lexically_normal().generic_string();
    auto hint_text = std::filesystem::path{hint}.lexically_normal().generic_string();
    std::transform(candidate_text.begin(), candidate_text.end(), candidate_text.begin(),
                   [](unsigned char value) { return static_cast<char>(std::tolower(value)); });
    std::transform(hint_text.begin(), hint_text.end(), hint_text.begin(),
                   [](unsigned char value) { return static_cast<char>(std::tolower(value)); });
    return candidate_text == hint_text ||
           (candidate_text.size() > hint_text.size() && candidate_text.ends_with("/" + hint_text));
}

runtime_material_instance* runtime_for(editor_scene_state& scene, ecs::entity_guid entity)
{
    const auto found = std::ranges::find_if(runtime_instances, [&](const runtime_material_instance& value)
                                            { return value.scene == &scene && value.entity == entity; });
    return found == runtime_instances.end() ? nullptr : &*found;
}

const editor_material_record* base_material_record(editor_scene_state& scene, ecs::entity entity)
{
    const auto guid = entity_guid_of(scene, entity);
    const auto* runtime = runtime_for(scene, guid);
    const auto* component = scene.scene.try_get<scene::mesh_renderer_component>(entity);
    if (!component) return nullptr;

    if (runtime)
    {
        const auto found =
            std::ranges::find(scene.material_library.materials, runtime->parent, &editor_material_record::material);
        if (found != scene.material_library.materials.end()) return &*found;
    }

    for (const auto& record : scene.material_library.materials)
    {
        if (record.asset.name.find(instance_name_marker) != std::string::npos) continue;
        if (record.material == component->material) return &record;
    }

    // Runtime instance tracking is intentionally rebuilt during scene synchronization. During that window the
    // component can still reference the previous instance handle, so recover its parent from the synthetic instance
    // record. Synthetic records retain the base material path, which gives us a stable identity even without an asset
    // binding (for example, immediately after history restore or scene reload).
    const auto instance_record =
        std::ranges::find(scene.material_library.materials, component->material, &editor_material_record::material);
    if (instance_record != scene.material_library.materials.end() &&
        instance_record->asset.name.find(instance_name_marker) != std::string::npos)
    {
        const auto base = std::ranges::find_if(scene.material_library.materials, [&](const editor_material_record& value)
                                               {
                                                   return value.asset.name.find(instance_name_marker) == std::string::npos &&
                                                          value.path.lexically_normal() ==
                                                              instance_record->path.lexically_normal();
                                               });
        if (base != scene.material_library.materials.end()) return &*base;
    }

    if (const auto* binding = find_asset_binding(scene, guid); binding && !binding->material.path_hint.empty())
    {
        for (const auto& record : scene.material_library.materials)
        {
            if (record.asset.name.find(instance_name_marker) != std::string::npos) continue;
            if (same_path_suffix(record.path, binding->material.path_hint)) return &record;
        }
    }
    return nullptr;
}

std::filesystem::path resolve_texture_path(const editor_material_record& material, std::string_view path)
{
    if (path.empty()) return {};
    std::filesystem::path authored{path};
    if (authored.is_absolute()) return authored.lexically_normal();

    auto directory = material.path.parent_path();
    for (auto current = directory; !current.empty(); current = current.parent_path())
    {
        const auto candidate = (current / authored).lexically_normal();
        std::error_code ec;
        if (std::filesystem::exists(candidate, ec) && !ec) return candidate;
        const auto parent = current.parent_path();
        if (parent == current) break;
    }
    return (directory / authored).lexically_normal();
}

render::texture_handle ensure_override_texture(editor_scene_state& scene, render::renderer& renderer,
                                               const editor_material_record& material, std::string_view path)
{
    if (path.empty()) return {};
    auto resolved = resolve_texture_path(material, path);
    std::error_code ec;
    auto key = std::filesystem::absolute(resolved, ec).lexically_normal();
    if (ec) key = resolved.lexically_normal();
    key += "#material-parameter";
    for (const auto& [texture_path, handle] : scene.material_library.textures)
        if (texture_path == key) return handle;

    auto loaded = render::load_texture_asset(resolved);
    if (!loaded.succeeded())
    {
        arc::diagnostics::warn("editor.materials", "Material instance texture could not be loaded: " +
                                                       resolved.generic_string() + " (" + loaded.message + ")");
        return {};
    }
    loaded.texture.semantic = render::texture_semantic::generic_color;
    loaded.texture.color_space = render::required_color_space(loaded.texture.semantic);
    const auto handle = renderer.create_texture(std::move(loaded.texture));
    if (handle.valid()) scene.material_library.textures.push_back({std::move(key), handle});
    return handle;
}

std::optional<render::material_parameter_value> override_value(editor_scene_state& scene, render::renderer& renderer,
                                                               const editor_material_record& base,
                                                               const render::shader_parameter_descriptor& parameter,
                                                               const json& authored)
{
    const auto values = authored.value("value", std::vector<float>{});
    const auto finite = [](const std::vector<float>& source, std::size_t count)
    { return source.size() == count && std::ranges::all_of(source, [](float value) { return std::isfinite(value); }); };

    switch (parameter.type)
    {
        case render::shader_parameter_type::float32:
            if (finite(values, 1u)) return values[0];
            break;
        case render::shader_parameter_type::float2:
            if (finite(values, 2u)) return math::vector2f{values[0], values[1]};
            break;
        case render::shader_parameter_type::float3:
            if (finite(values, 3u)) return math::vector3f{values[0], values[1], values[2]};
            break;
        case render::shader_parameter_type::float4:
            if (finite(values, 4u)) return math::vector4f{values[0], values[1], values[2], values[3]};
            break;
        case render::shader_parameter_type::texture_2d:
            return render::resource_handle{
                ensure_override_texture(scene, renderer, base, authored.value("texture", std::string{}))};
        default:
            break;
    }
    return std::nullopt;
}

bool realize_overrides(editor_scene_state& scene, render::renderer& renderer, ecs::entity entity, const json& overrides)
{
    auto* component = scene.scene.try_get<scene::mesh_renderer_component>(entity);
    if (!component) return false;
    const auto guid = entity_guid_of(scene, entity);
    const auto* base_pointer = base_material_record(scene, entity);
    if (!base_pointer) return false;
    const editor_material_record base = *base_pointer;

    if (!overrides.is_array() || overrides.empty())
    {
        component->material = base.material;
        runtime_instances.erase(std::remove_if(runtime_instances.begin(), runtime_instances.end(),
                                               [&](const runtime_material_instance& value)
                                               { return value.scene == &scene && value.entity == guid; }),
                                runtime_instances.end());
        return true;
    }

    if (!base.asset.material.runtime_program)
    {
        arc::diagnostics::warn("editor.materials",
                               "Material instance requires a compiled parameter layout for '" + base.asset.name + "'");
        return false;
    }

    render::material_instance_descriptor instance;
    instance.parent = base.material;
    instance.name = base.asset.name + " Instance";
    for (const auto& authored : overrides)
    {
        if (!authored.is_object()) continue;
        const auto name = authored.value("name", std::string{});
        const auto layout = std::ranges::find(base.asset.material.runtime_program->parameters, name,
                                              &render::shader_parameter_descriptor::name);
        if (layout == base.asset.material.runtime_program->parameters.end())
        {
            arc::diagnostics::warn("editor.materials", "Ignoring stale material instance parameter '" + name + "'");
            continue;
        }
        const auto value = override_value(scene, renderer, base, *layout, authored);
        if (!value)
        {
            arc::diagnostics::warn("editor.materials",
                                   "Ignoring incompatible material instance parameter '" + name + "'");
            continue;
        }
        instance.overrides.push_back({.id = layout->id, .name = name, .value = *value});
    }

    render::material_definition_descriptor definition;
    definition.material = base.asset.material;
    definition.parameter_layout = base.asset.material.runtime_program->parameters;
    auto resolved = render::resolve_material_instance(definition, instance);
    if (!resolved)
    {
        arc::diagnostics::warn("editor.materials",
                               "Material instance could not be resolved: " + resolved.error().message);
        return false;
    }

    auto* runtime = runtime_for(scene, guid);
    render::material_handle instance_handle{};
    if (runtime && renderer.material_alive(runtime->instance))
    {
        instance_handle = runtime->instance;
        if (!renderer.update_material(instance_handle, std::move(resolved).value())) return false;
        runtime->parent = base.material;
    }
    else
    {
        instance_handle = renderer.create_material(std::move(resolved).value());
        if (!instance_handle.valid()) return false;
        runtime_instances.push_back(
            {.scene = &scene, .entity = guid, .parent = base.material, .instance = instance_handle});
    }
    component->material = instance_handle;

    const std::string encoded = encode_hex(overrides.dump());
    const std::string instance_asset_name = base.asset.name + std::string(instance_name_marker) + encoded;
    auto record =
        std::ranges::find(scene.material_library.materials, instance_handle, &editor_material_record::material);
    if (record == scene.material_library.materials.end())
    {
        auto asset = base.asset;
        asset.name = instance_asset_name;
        scene.material_library.materials.push_back({base.path, std::move(asset), instance_handle});
    }
    else
    {
        record->path = base.path;
        record->asset = base.asset;
        record->asset.name = instance_asset_name;
    }
    return true;
}

bool apply_material_edit(editor_scene_state& scene, render::renderer& renderer, ecs::entity entity,
                         const material_parameter_edit& edit)
{
    const auto guid = entity_guid_of(scene, entity);
    if (!guid.valid()) return false;
    auto overrides = apply_edit(persisted_overrides(scene, guid), edit);
    if (!realize_overrides(scene, renderer, entity, overrides)) return false;
    store_persisted_overrides(scene, guid, overrides);
    return true;
}

} // namespace

procedural_mesh_component* ensure_procedural_or_material_parameter_component(editor_scene_state& scene,
                                                                             ecs::entity entity)
{
    pending_edit = {};
    if (!scene.scene.has<scene::mesh_renderer_component>(entity)) return nullptr;
    pending_edit.scene = &scene;
    pending_edit.entity = entity;
    if (auto* procedural = ensure_procedural_mesh_component(scene, entity)) return procedural;
    return &pending_edit.dummy;
}

bool set_procedural_or_material_parameter(procedural_mesh_component& component, std::string_view parameter,
                                          double value)
{
    if (pending_edit.scene && parameter.starts_with(material_parameter_prefix))
    {
        pending_edit.material = parse_material_parameter(parameter);
        return pending_edit.material.has_value();
    }
    return set_procedural_mesh_parameter(component, parameter, value);
}

bool regenerate_procedural_or_material_parameter(editor_scene_state& scene, render::renderer& renderer,
                                                 ecs::entity entity)
{
    if (pending_edit.scene == &scene && pending_edit.entity == entity && pending_edit.material)
    {
        const auto edit = std::move(*pending_edit.material);
        pending_edit = {};
        return apply_material_edit(scene, renderer, entity, edit);
    }
    pending_edit = {};
    return regenerate_procedural_mesh(scene, renderer, entity);
}

void synchronize_procedural_and_material_instances(editor_scene_state& scene, render::renderer& renderer)
{
    synchronize_procedural_mesh_components(scene, renderer);
    runtime_instances.erase(std::remove_if(runtime_instances.begin(), runtime_instances.end(),
                                           [&](const runtime_material_instance& value)
                                           { return value.scene == &scene; }),
                            runtime_instances.end());

    for (const auto& preserved : scene.preserved_component_records)
    {
        if (preserved.component_name != mesh_renderer_component_name) continue;
        const auto entity = find_entity_by_guid(scene, preserved.entity);
        if (!scene.scene.alive(entity) || !scene.scene.has<scene::mesh_renderer_component>(entity)) continue;
        const auto component = json::parse(preserved.json, nullptr, false);
        if (!component.is_object()) continue;
        const auto found = component.find(std::string(persisted_override_field));
        if (found == component.end() || !found->is_array() || found->empty()) continue;
        (void)realize_overrides(scene, renderer, entity, *found);
    }
}

} // namespace arc::editor
