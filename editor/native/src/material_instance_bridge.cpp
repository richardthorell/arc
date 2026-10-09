#include <arc/editor/procedural_mesh.h>

#include <arc/editor/editor_state.h>
#include <arc/editor/material_library.h>
#include <arc/editor/material_preview_realizer.h>
#include <arc/diagnostics/diagnostics.h>
#include <arc/render/texture.h>
#include <arc/scene/scene.h>

#include <nlohmann/json.hpp>

#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <optional>
#include <set>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

namespace arc::editor
{
namespace
{
using json = nlohmann::json;

constexpr std::string_view material_parameter_prefix = "__arc_material_parameter__";
constexpr std::string_view material_function_prefix = "__arc_material_function__";
constexpr std::string_view instance_name_marker = "__arc_instance_overrides__";
constexpr std::string_view mesh_renderer_component_name = "MeshRenderer";
constexpr std::string_view persisted_override_field = "materialParameterOverrides";

enum class material_parameter_edit_kind : std::uint8_t
{
    scalar,
    vector,
    color,
    texture
};

struct material_parameter_edit
{
    std::uint64_t parameter_id{};
    std::string slot_id;
    std::string name;
    render::shader_parameter_type type{render::shader_parameter_type::float32};
    material_parameter_edit_kind kind{material_parameter_edit_kind::scalar};
    std::vector<float> value;
    std::string texture;
    bool reset{};
};

struct material_function_edit
{
    std::string slot_id;
    std::string function_guid;
    std::string function_path;
    bool reset{};
};

struct pending_parameter_edit
{
    editor_scene_state* scene{};
    ecs::entity entity{};
    procedural_mesh_component dummy{};
    std::optional<material_parameter_edit> material;
    std::optional<material_function_edit> function;
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

std::optional<render::shader_parameter_type> material_parameter_type_from_string(std::string_view value) noexcept
{
    if (value == "float") return render::shader_parameter_type::float32;
    if (value == "vec2") return render::shader_parameter_type::float2;
    if (value == "vec3") return render::shader_parameter_type::float3;
    if (value == "vec4") return render::shader_parameter_type::float4;
    if (value == "texture2d") return render::shader_parameter_type::texture_2d;
    return std::nullopt;
}

std::string_view material_parameter_type_name(render::shader_parameter_type type) noexcept
{
    switch (type)
    {
        case render::shader_parameter_type::float32:
            return "float";
        case render::shader_parameter_type::float2:
            return "vec2";
        case render::shader_parameter_type::float3:
            return "vec3";
        case render::shader_parameter_type::float4:
            return "vec4";
        case render::shader_parameter_type::texture_2d:
            return "texture2d";
        default:
            return {};
    }
}

std::optional<material_parameter_edit_kind> material_parameter_kind_from_string(std::string_view value) noexcept
{
    if (value == "scalar") return material_parameter_edit_kind::scalar;
    if (value == "vector") return material_parameter_edit_kind::vector;
    if (value == "color") return material_parameter_edit_kind::color;
    if (value == "texture") return material_parameter_edit_kind::texture;
    return std::nullopt;
}

std::string_view material_parameter_kind_name(material_parameter_edit_kind kind) noexcept
{
    switch (kind)
    {
        case material_parameter_edit_kind::scalar:
            return "scalar";
        case material_parameter_edit_kind::vector:
            return "vector";
        case material_parameter_edit_kind::color:
            return "color";
        case material_parameter_edit_kind::texture:
            return "texture";
    }
    return {};
}

bool material_parameter_edit_metadata_matches(render::shader_parameter_type type,
                                              material_parameter_edit_kind kind) noexcept
{
    switch (kind)
    {
        case material_parameter_edit_kind::scalar:
            return type == render::shader_parameter_type::float32;
        case material_parameter_edit_kind::vector:
            return type == render::shader_parameter_type::float2 || type == render::shader_parameter_type::float3 ||
                   type == render::shader_parameter_type::float4;
        case material_parameter_edit_kind::color:
            return type == render::shader_parameter_type::float3 || type == render::shader_parameter_type::float4;
        case material_parameter_edit_kind::texture:
            return type == render::shader_parameter_type::texture_2d;
    }
    return false;
}

std::optional<material_parameter_edit> parse_material_parameter(std::string_view parameter)
{
    if (!parameter.starts_with(material_parameter_prefix)) return std::nullopt;
    const auto decoded = decode_hex(parameter.substr(material_parameter_prefix.size()));
    if (!decoded) return std::nullopt;
    const auto payload = json::parse(*decoded, nullptr, false);
    if (!payload.is_object() || !payload.contains("name") || !payload["name"].is_string()) return std::nullopt;

    const auto type_found = payload.find("type");
    const auto kind_found = payload.find("kind");
    if (type_found == payload.end() || !type_found->is_string() || kind_found == payload.end() ||
        !kind_found->is_string())
        return std::nullopt;

    const auto type = material_parameter_type_from_string(type_found->get<std::string>());
    if (!type.has_value()) return std::nullopt;
    const auto kind = material_parameter_kind_from_string(kind_found->get<std::string>());
    if (!kind.has_value()) return std::nullopt;

    const auto typed_type = type.value();
    const auto typed_kind = kind.value();
    if (!material_parameter_edit_metadata_matches(typed_type, typed_kind)) return std::nullopt;

    material_parameter_edit edit;
    if (const auto found = payload.find("parameterId"); found != payload.end() && found->is_string())
    {
        try
        {
            edit.parameter_id = std::stoull(found->get<std::string>());
        }
        catch (...)
        {
            return std::nullopt;
        }
    }
    edit.slot_id = payload.value("slotId", std::string{});
    edit.name = payload["name"].get<std::string>();
    edit.type = typed_type;
    edit.kind = typed_kind;
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

std::optional<material_function_edit> parse_material_function(std::string_view parameter)
{
    if (!parameter.starts_with(material_function_prefix)) return std::nullopt;
    const auto decoded = decode_hex(parameter.substr(material_function_prefix.size()));
    if (!decoded) return std::nullopt;
    const auto payload = json::parse(*decoded, nullptr, false);
    if (!payload.is_object()) return std::nullopt;

    material_function_edit edit;
    edit.slot_id = payload.value("slotId", std::string{});
    edit.reset = payload.value("reset", false);
    if (const auto found = payload.find("function"); found != payload.end() && found->is_object())
    {
        edit.function_guid = found->value("guid", std::string{});
        edit.function_path = found->value("pathHint", std::string{});
    }
    if (edit.slot_id.empty()) return std::nullopt;
    if (!edit.reset && (edit.function_guid.empty() || edit.function_path.empty())) return std::nullopt;
    return edit;
}

std::string read_text_file(const std::filesystem::path& path)
{
    std::ifstream stream(path, std::ios::binary);
    if (!stream) return {};
    std::ostringstream output;
    output << stream.rdbuf();
    return output.str();
}

bool collect_function_paths(const json& graph, std::vector<std::string>& paths)
{
    if (!graph.is_object() || !graph.contains("nodes") || !graph["nodes"].is_array()) return false;
    for (const auto& node : graph["nodes"])
    {
        if (!node.is_object()) continue;
        const auto type = node.value("type", std::string{});
        if (type != "functionCall" && type != "functionSlot") continue;
        const auto values = node.value("values", json::object());
        const auto path = values.value("path", std::string{});
        if (!path.empty()) paths.push_back(path);
    }
    return true;
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
    json value = {{"name", edit.name},
                  {"type", std::string(material_parameter_type_name(edit.type))},
                  {"kind", std::string(material_parameter_kind_name(edit.kind))}};
    if (edit.parameter_id != 0u) value["parameterId"] = std::to_string(edit.parameter_id);
    if (!edit.slot_id.empty()) value["slotId"] = edit.slot_id;
    if (!edit.value.empty()) value["value"] = edit.value;
    if (edit.kind == material_parameter_edit_kind::texture) value["texture"] = edit.texture;
    return value;
}

json apply_edit(json overrides, const material_parameter_edit& edit)
{
    if (!overrides.is_array()) overrides = json::array();
    overrides.erase(
        std::remove_if(overrides.begin(), overrides.end(),
                       [&](const json& entry)
                       {
                           if (!entry.is_object() || entry.value("kind", std::string{}) == "function") return false;
                           if (edit.parameter_id != 0u)
                               return entry.value("parameterId", std::string{}) == std::to_string(edit.parameter_id);
                           return entry.value("name", std::string{}) == edit.name;
                       }),
        overrides.end());
    if (!edit.reset) overrides.push_back(edit_to_json(edit));
    return overrides;
}

json apply_function_edit(json overrides, const material_function_edit& edit)
{
    if (!overrides.is_array()) overrides = json::array();
    overrides.erase(
        std::remove_if(overrides.begin(), overrides.end(),
                       [&](const json& entry)
                       {
                           if (!entry.is_object() || entry.value("slotId", std::string{}) != edit.slot_id) return false;
                           return entry.value("kind", std::string{}) == "function" ||
                                  !entry.value("parameterId", std::string{}).empty();
                       }),
        overrides.end());
    if (!edit.reset)
        overrides.push_back({{"kind", "function"},
                             {"slotId", edit.slot_id},
                             {"function", {{"guid", edit.function_guid}, {"pathHint", edit.function_path}}}});
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
        const auto base =
            std::ranges::find_if(scene.material_library.materials,
                                 [&](const editor_material_record& value)
                                 {
                                     return value.asset.name.find(instance_name_marker) == std::string::npos &&
                                            value.path.lexically_normal() == instance_record->path.lexically_normal();
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

std::optional<material_preview_descriptor_result>
realize_function_specialization(editor_scene_state& scene, render::renderer& renderer,
                                const editor_material_record& base, const json& overrides)
{
    std::vector<render::tools::material_function_slot_override> slot_overrides;
    struct selected_function
    {
        std::string guid;
        std::string path;
    };
    std::vector<selected_function> selected;
    for (const auto& entry : overrides)
    {
        if (!entry.is_object() || entry.value("kind", std::string{}) != "function") continue;
        const auto slot_id = entry.value("slotId", std::string{});
        const auto function = entry.value("function", json::object());
        const auto guid = function.value("guid", std::string{});
        const auto path = function.value("pathHint", std::string{});
        if (slot_id.empty() || guid.empty() || path.empty()) continue;
        slot_overrides.push_back({.slot_id = slot_id, .function_path = path});
        selected.push_back({.guid = guid, .path = path});
    }
    if (slot_overrides.empty()) return std::nullopt;

    const auto source = read_text_file(base.path);
    const auto document = json::parse(source, nullptr, false);
    if (source.empty() || document.is_discarded() || !document.is_object() || !document.contains("graph"))
        return material_preview_descriptor_result{.message = "Material source is unavailable for Function specialization"};

    std::vector<std::string> pending;
    if (!collect_function_paths(document["graph"], pending))
        return material_preview_descriptor_result{.message = "Material graph is malformed while resolving Functions"};
    for (const auto& replacement : selected)
        pending.push_back(replacement.path);

    std::vector<render::tools::material_function_source> functions;
    std::set<std::string> visited;
    for (std::size_t index = 0; index < pending.size(); ++index)
    {
        const auto source_path = resolve_texture_path(base, pending[index]);
        auto key = source_path.lexically_normal().generic_string();
        std::transform(key.begin(), key.end(), key.begin(),
                       [](unsigned char value) { return static_cast<char>(std::tolower(value)); });
        if (!visited.insert(key).second) continue;

        const auto function_source = read_text_file(source_path);
        const auto function_document = json::parse(function_source, nullptr, false);
        if (function_source.empty() || function_document.is_discarded() || !function_document.is_object())
            return material_preview_descriptor_result{.message = "Material Function could not be loaded: " + key};

        std::string identity = key;
        for (const auto& replacement : selected)
            if (same_path_suffix(source_path, replacement.path))
            {
                identity = replacement.guid;
                break;
            }

        functions.push_back({.path = source_path.generic_string(), .identity = std::move(identity), .source = function_source});
        if (function_document.contains("graph")) collect_function_paths(function_document["graph"], pending);
    }

    auto realized = realize_material_preview_descriptor(source, base.asset.name + " Instance", functions, slot_overrides);
    if (realized.succeeded)
    {
        const auto asset_root = base.path.parent_path().parent_path();
        resolve_material_runtime_textures(scene.material_library, renderer, asset_root, base.path,
                                          realized.texture_sources, realized.material);
    }
    return realized;
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

    render::material_descriptor specialized_material = base.asset.material;
    if (const auto specialized = realize_function_specialization(scene, renderer, base, overrides))
    {
        if (!specialized->succeeded)
        {
            arc::diagnostics::warn("editor.materials", "Material Function specialization failed: " + specialized->message);
            return false;
        }
        specialized_material = specialized->material;
    }

    if (!specialized_material.runtime_program)
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
        if (!authored.is_object() || authored.value("kind", std::string{}) == "function") continue;
        const auto name = authored.value("name", std::string{});
        const auto parameter_id_text = authored.value("parameterId", std::string{});
        std::uint64_t parameter_id{};
        if (!parameter_id_text.empty())
        {
            try
            {
                parameter_id = std::stoull(parameter_id_text);
            }
            catch (...)
            {
                parameter_id = 0u;
            }
        }
        const auto& parameters = specialized_material.runtime_program->parameters;
        const auto layout = parameter_id != 0u
                                ? std::ranges::find(parameters, parameter_id, &render::shader_parameter_descriptor::id)
                                : std::ranges::find(parameters, name, &render::shader_parameter_descriptor::name);
        if (layout == parameters.end())
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
    definition.material = specialized_material;
    definition.parameter_layout = specialized_material.runtime_program->parameters;
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

bool apply_function_edit(editor_scene_state& scene, render::renderer& renderer, ecs::entity entity,
                         const material_function_edit& edit)
{
    const auto guid = entity_guid_of(scene, entity);
    if (!guid.valid()) return false;
    auto overrides = apply_function_edit(persisted_overrides(scene, guid), edit);
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
    if (pending_edit.scene && parameter.starts_with(material_function_prefix))
    {
        pending_edit.function = parse_material_function(parameter);
        return pending_edit.function.has_value();
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
    if (pending_edit.scene == &scene && pending_edit.entity == entity && pending_edit.function)
    {
        const auto edit = std::move(*pending_edit.function);
        pending_edit = {};
        return apply_function_edit(scene, renderer, entity, edit);
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
