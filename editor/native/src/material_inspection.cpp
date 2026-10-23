#include <arc/editor/material_inspection.h>

#include <arc/assets/assets.h>
#include <arc/render_tools/material_asset.h>
#include <arc/render_tools/material_graph.h>
#include <arc/render_tools/texture_cooker.h>

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <map>
#include <set>
#include <stdexcept>
#include <vector>

namespace arc::editor
{
namespace
{
using json = nlohmann::json;
namespace rt = render::tools;
constexpr std::size_t source_limit = 8 * 1024 * 1024;
constexpr std::size_t total_limit = 32 * 1024 * 1024;
constexpr std::size_t dependency_limit = 128;
constexpr std::size_t alternative_limit = 16;

std::string_view type_name(render::shader_parameter_type type)
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
            return "unsupported";
    }
}

json pin_json(const rt::material_function_pin& pin)
{
    json value = nullptr;
    if (pin.has_default)
        value = std::vector<float>(pin.default_value.values.begin(),
                                   pin.default_value.values.begin() + pin.default_value.components);
    return {{"id", pin.id},
            {"name", pin.name},
            {"type", type_name(pin.type)},
            {"hasDefault", pin.has_default},
            {"defaultValue", value}};
}

json checked_value(json value, const render::shader_parameter_descriptor& parameter)
{
    const auto type = type_name(parameter.type);
    if (type == "texture2d")
    {
        if (value.is_null()) return nullptr;
        if (value.is_string() && value.get<std::string>().empty()) return nullptr;
        if (value.is_string() || (value.is_object() && value.contains("guid") && value["guid"].is_string() &&
                                  assets::parse_asset_guid(value["guid"].get<std::string>())))
            return value;
        throw std::runtime_error("Texture override requires an asset reference");
    }
    const auto count = type == "float" ? 1u : type == "vec2" ? 2u : type == "vec3" ? 3u : type == "vec4" ? 4u : 0u;
    if (count == 1 && value.is_number()) value = json::array({value});
    if (!count || !value.is_array() || value.size() != count)
        throw std::runtime_error("Override shape does not match parameter " +
                                 std::to_string(parameter.id.representation()));
    for (const auto& component : value)
        if (!component.is_number() || !std::isfinite(component.get<float>()) ||
            (parameter.has_range &&
             (component.get<float>() < parameter.minimum || component.get<float>() > parameter.maximum)))
            throw std::runtime_error("Override is non-finite or outside the declared parameter range");
    return value;
}

struct inspection
{
    const material_inspection_resolver& resolve;
    std::map<std::string, material_inspection_source> sources;
    std::map<std::string, std::string> aliases;
    std::size_t bytes{};

    void remember(const material_inspection_source& source)
    {
        if (source.guid.empty()) throw std::runtime_error("Dependency has no registered asset identity");
        if (sources.contains(source.guid)) return;
        if (source.source.size() > source_limit || sources.size() >= dependency_limit ||
            source.source.size() > total_limit - bytes)
            throw std::runtime_error("Material inspection dependency/source budget exceeded");
        bytes += source.source.size();
        sources.emplace(source.guid, source);
    }

    material_inspection_source load(std::string_view guid, std::string_view hint)
    {
        const auto source = resolve(guid, hint);
        if (!source || (!guid.empty() && source->guid != guid))
            throw std::runtime_error("Missing registered dependency: " + std::string(guid.empty() ? hint : guid));
        remember(*source);
        return *source;
    }

    void functions(const json& graph, std::vector<rt::material_function_source>& library)
    {
        if (!graph.contains("nodes") || !graph["nodes"].is_array()) return;
        for (const auto& node : graph["nodes"])
        {
            const auto type = node.value("type", std::string{});
            if (type != "functionCall" && type != "functionSlot") continue;
            const auto values = node.value("values", json::object());
            std::vector<std::string> paths{values.value("path", std::string{})};
            if (values.contains("functions") && values["functions"].is_array())
                for (const auto& entry : values["functions"])
                    paths.push_back(entry.value("path", std::string{}));
            for (const auto& path : paths)
                function(path, {}, library);
        }
    }

    void function(const std::string& path, const std::string& guid, std::vector<rt::material_function_source>& library)
    {
        if (path.empty() && guid.empty()) return;
        const auto source = load(guid, path);
        const auto alias = path.empty() ? source.path : path;
        if (const auto found = aliases.find(alias); found != aliases.end())
        {
            if (found->second != source.guid) throw std::runtime_error("Ambiguous Material Function locator");
            return;
        }
        if (aliases.size() >= dependency_limit) throw std::runtime_error("Too many Material Function aliases");
        aliases.emplace(alias, source.guid);
        if (source.kind != "materialFunction") throw std::runtime_error("Dependency is not a Material Function");
        library.push_back({.path = alias, .identity = source.guid, .source = source.source});
        const auto document = json::parse(source.source);
        if (document.contains("graph")) functions(document["graph"], library);
    }

    std::string revision() const
    {
        // Sorted GUIDs, not names or filesystem timestamps, determine the dependency revision.
        json identity = json::array();
        for (const auto& [guid, source] : sources)
            identity.push_back({guid, source.source});
        const auto serialized = identity.dump();
        return assets::to_string(assets::hash_bytes(std::as_bytes(std::span(serialized.data(), serialized.size()))));
    }
};
} // namespace

std::string inspect_saved_material(const material_inspection_source& source,
                                   const material_inspection_resolver& resolve, std::string_view entity_overrides,
                                   const material_function_discovery_request& discovery)
{
    json result{{"guid", source.guid},          {"kind", source.kind},
                {"provenance", "savedSource"},  {"definitionRevision", ""},
                {"parentGuid", nullptr},        {"dependencies", json::array()},
                {"parameters", json::array()},  {"functionSlots", json::array()},
                {"functionInterface", nullptr}, {"diagnostics", json::array()},
                {"definitionReady", false},     {"runtimeStatus", "notInspected"}};
    inspection state{.resolve = resolve};
    try
    {
        if (!discovery.slot_id.empty() &&
            (discovery.limit == 0 || discovery.limit > alternative_limit || discovery.offset > dependency_limit ||
             (discovery.offset != 0 && discovery.revision.empty())))
            throw std::runtime_error("Invalid function discovery bounds or missing discovery revision");
        const auto local = json::parse(entity_overrides);
        if (!local.is_array() || local.size() > 1024) throw std::runtime_error("Invalid entity override state");
        state.remember(source);
        if (source.kind == "materialFunction")
        {
            const auto validated = rt::validate_material_function_json(source.source, source.path);
            if (!validated) throw std::runtime_error(validated.error().message);
            const auto document = json::parse(source.source);
            const auto inputs = document.value("inputs", json::array()).size();
            json signature{{"inputs", json::array()}, {"outputs", json::array()}};
            for (std::size_t i = 0; i < validated.value().size(); ++i)
                signature[i < inputs ? "inputs" : "outputs"].push_back(pin_json(validated.value()[i]));
            result["functionInterface"] = std::move(signature);
            result["definitionReady"] = true;
        }
        else
        {
            auto parent = source;
            std::optional<rt::material_instance_authoring_document> instance;
            if (source.kind == "materialInstance")
            {
                auto parsed = rt::parse_material_instance_authoring_json(source.source);
                if (!parsed) throw std::runtime_error(parsed.error().message);
                instance = std::move(parsed).value();
                parent = state.load(instance->parent.guid, instance->parent.path_hint);
                result["parentGuid"] = parent.guid;
            }
            if (parent.kind != "material") throw std::runtime_error("Expected a Material parent asset");
            const auto authored = rt::parse_material_authoring_json(parent.source);
            if (!authored) throw std::runtime_error(authored.error().message);
            if (authored.value().graph_json.empty())
                throw std::runtime_error("Handwritten shader parameter inspection is not yet supported");
            const auto graph = json::parse(authored.value().graph_json);
            std::vector<rt::material_function_source> functions;
            state.functions(graph, functions);
            std::vector<rt::material_function_slot_override> selections;
            std::map<std::string, std::string> inherited_functions;
            if (instance)
                for (const auto& selection : instance->function_overrides)
                {
                    const auto selected = state.load(selection.function.guid, selection.function.path_hint);
                    // The compiler enforces authored alternative lists, including their original locators.
                    auto path = selected.path;
                    for (const auto& [alias, guid] : state.aliases)
                        if (guid == selected.guid)
                        {
                            path = alias;
                            break;
                        }
                    state.function(path, selected.guid, functions);
                    selections.push_back({.slot_id = selection.slot_id, .function_path = path});
                    inherited_functions.emplace(selection.slot_id, selected.guid);
                }
            for (const auto& entry : local)
            {
                if (entry.value("kind", std::string{}) != "function") continue;
                const auto reference = entry.at("function");
                const auto selected =
                    state.load(reference.value("guid", std::string{}), reference.value("pathHint", std::string{}));
                auto path = selected.path;
                for (const auto& [alias, guid] : state.aliases)
                    if (guid == selected.guid)
                    {
                        path = alias;
                        break;
                    }
                state.function(path, selected.guid, functions);
                const auto id = entry.at("slotId").get<std::string>();
                std::erase_if(selections, [&](const auto& selection) { return selection.slot_id == id; });
                selections.push_back({.slot_id = id, .function_path = path});
            }
            const auto compiled = rt::compile_material_graph_json(authored.value().graph_json, functions, selections);
            if (!compiled) throw std::runtime_error(compiled.error().message);
            std::map<std::uint64_t, json> overrides;
            if (instance)
            {
                for (const auto& entry : instance->parameter_overrides)
                    overrides[entry.parameter_id.representation()] = json::parse(entry.value_json);
                for (const auto& selection : instance->function_overrides)
                {
                    // An entity replacement inherits the newly selected function's defaults, not
                    // input overrides belonging to the instance's previous selection.
                    const bool replaced = std::ranges::any_of(
                        local,
                        [&](const auto& entry)
                        {
                            return entry.value("kind", std::string{}) == "function" &&
                                   entry.value("slotId", std::string{}) == selection.slot_id &&
                                   entry.value("function", json::object()).value("guid", std::string{}) !=
                                       selection.function.guid;
                        });
                    if (replaced) continue;
                    for (const auto& entry : selection.input_overrides)
                        overrides[render::make_shader_parameter_id("slot::" + selection.slot_id +
                                                                   "::" + selection.function.guid + "::" + entry.pin_id)
                                      .representation()] = json::parse(entry.value_json);
                }
            }
            for (const auto& parameter : compiled.value().descriptor.parameters)
            {
                const auto type = type_name(parameter.type);
                json default_value = nullptr;
                json texture = nullptr;
                std::string texture_status;
                // Resolve locators through the registry, never expose a path as asset identity.
                const auto resolve_texture = [&](const json& value, std::string& status) -> json
                {
                    if (value.is_null() || (value.is_string() && value.get<std::string>().empty()))
                    {
                        status = "empty";
                        return nullptr;
                    }
                    std::string guid, hint;
                    if (value.is_object())
                    {
                        guid = value.value("guid", "");
                        hint = value.value("pathHint", "");
                    }
                    else if (value.is_string())
                    {
                        const auto locator = value.get<std::string>();
                        if (assets::parse_asset_guid(locator))
                            guid = locator;
                        else
                            hint = locator;
                    }
                    const auto asset = (guid.empty() && hint.empty()) ? std::nullopt : resolve(guid, hint);
                    if (!asset || asset->kind != "texture2d" || !assets::parse_asset_guid(asset->guid) ||
                        (!guid.empty() && asset->guid != guid))
                    {
                        status = "unresolved";
                        result["diagnostics"].push_back(
                            {{"code", "unresolved-texture"},
                             {"message", "Texture parameter " + std::to_string(parameter.id.representation()) +
                                             " references a missing or incompatible Texture2D asset: " +
                                             (guid.empty() ? hint : guid)}});
                        return nullptr;
                    }
                    status = "resolved";
                    return {{"guid", asset->guid}, {"pathHint", asset->path}};
                };
                const auto count = type == "float"  ? 1u
                                   : type == "vec2" ? 2u
                                   : type == "vec3" ? 3u
                                   : type == "vec4" ? 4u
                                                    : 0u;
                if (count && parameter.default_value.size() == count * sizeof(float))
                {
                    std::vector<float> values(count);
                    std::memcpy(values.data(), parameter.default_value.data(), parameter.default_value.size());
                    default_value = values;
                }
                auto resolved = default_value;
                bool value_available = !default_value.is_null();
                if (type == "texture2d")
                {
                    const auto binding =
                        std::ranges::find_if(compiled.value().descriptor.textures,
                                             [&](const auto& entry) { return entry.parameter_id == parameter.id; });
                    texture_status = "unresolved";
                    if (binding != compiled.value().descriptor.textures.end())
                        default_value = resolve_texture(binding->default_source, texture_status);
                    texture = {{"assetType", "texture2d"},
                               {"dimension", "2d"},
                               {"semantic", binding == compiled.value().descriptor.textures.end()
                                                ? "unknown"
                                                : std::string(rt::texture_semantic_name(binding->semantic))},
                               {"defaultStatus", texture_status}};
                    resolved = default_value;
                    value_available = texture_status != "unresolved";
                }
                if (const auto found = overrides.find(parameter.id.representation()); found != overrides.end())
                {
                    resolved = checked_value(found->second, parameter);
                    value_available = true;
                    if (type == "texture2d")
                    {
                        resolved = resolve_texture(resolved, texture_status);
                        value_available = texture_status != "unresolved";
                    }
                    overrides.erase(found);
                }
                const auto inherited = resolved;
                if (type == "texture2d") texture["inheritedStatus"] = texture_status;
                bool overridden = false;
                json local_value = nullptr;
                for (const auto& entry : local)
                    if (entry.value("kind", std::string{}) != "function" &&
                        entry.value("parameterId", std::string{}) == std::to_string(parameter.id.representation()))
                    {
                        try
                        {
                            resolved =
                                checked_value(type == "texture2d" ? entry.value("textureReference",
                                                                                entry.value("texture", json(nullptr)))
                                                                  : entry.at("value"),
                                              parameter);
                            overridden = true;
                            value_available = true;
                            if (type == "texture2d")
                            {
                                resolved = resolve_texture(resolved, texture_status);
                                value_available = texture_status != "unresolved";
                            }
                            local_value = resolved;
                        }
                        catch (const std::exception& error)
                        {
                            result["diagnostics"].push_back(
                                {{"code", "stale-entity-override"}, {"message", error.what()}});
                        }
                    }
                std::string slot_id;
                for (const auto& slot : compiled.value().descriptor.function_slots)
                {
                    for (const auto& selected : slot.selected_parameters)
                        if (selected.parameter_id == parameter.id) slot_id = slot.id;
                    for (const auto& node : compiled.value().ir.nodes)
                        if (node.parameter_id == parameter.id && node.id.starts_with(slot.id + "::")) slot_id = slot.id;
                }
                std::string semantic = "none";
                for (const auto& output : compiled.value().descriptor.outputs)
                    if (output.output == rt::material_surface_output::base_color && output.connected)
                        for (const auto& node : compiled.value().ir.nodes)
                            if (node.id == output.source_node && node.parameter_id == parameter.id)
                                semantic = "baseColor";
                json presentation = nullptr;
                for (const auto& metadata : compiled.value().parameter_presentation)
                    if (metadata.id == parameter.id)
                    {
                        presentation = {
                            {"editorKind", metadata.editor_kind},
                            {"groupId", metadata.group_id},
                            {"groupLabel", metadata.group_label},
                            {"groupOrder", metadata.group_order ? json(*metadata.group_order) : json(nullptr)},
                            {"order", metadata.order ? json(*metadata.order) : json(nullptr)}};
                        break;
                    }
                result["parameters"].push_back(
                    {{"id", std::to_string(parameter.id.representation())},
                     {"name", parameter.name},
                     {"type", type},
                     {"defaultValue", default_value},
                     {"inheritedValue", inherited},
                     {"overridden", overridden},
                     {"localValue", local_value},
                     {"slotId", slot_id},
                     {"semantic", semantic},
                     {"presentation", presentation},
                     {"resolvedValue", resolved},
                     {"valueStatus", value_available ? "resolved" : "unavailable"},
                     {"range", parameter.has_range ? json{{"min", parameter.minimum}, {"max", parameter.maximum}}
                                                   : json(nullptr)}});
                if (type == "texture2d")
                {
                    texture["effectiveStatus"] = texture_status;
                    result["parameters"].back()["texture"] = std::move(texture);
                }
            }
            if (!overrides.empty())
                throw std::runtime_error("Instance contains stale parameter overrides; repair the saved definition");
            for (const auto& entry : local)
            {
                if (entry.value("kind", std::string{}) == "function") continue;
                const auto id = entry.value("parameterId", std::string{});
                if (!std::ranges::any_of(result["parameters"],
                                         [&](const auto& parameter) { return parameter["id"] == id; }))
                    result["diagnostics"].push_back(
                        {{"code", "stale-entity-override"},
                         {"message", "Saved entity override no longer matches a parameter: " + id}});
            }
            std::size_t trials = 0;
            if (!discovery.slot_id.empty() &&
                !std::ranges::any_of(compiled.value().descriptor.function_slots,
                                     [&](const auto& slot) { return slot.id == discovery.slot_id; }))
                throw std::runtime_error("Unknown Material Function slot");
            for (const auto& slot : compiled.value().descriptor.function_slots)
            {
                if (!discovery.slot_id.empty() && slot.id != discovery.slot_id) continue;
                json alternatives = json::array();
                std::set<std::string> seen;
                bool truncated = false;
                std::vector<const rt::material_function_source*> candidates;
                for (const auto& candidate : functions)
                    if (seen.insert(candidate.identity).second) candidates.push_back(&candidate);
                std::ranges::sort(candidates,
                                  [](const auto* left, const auto* right) { return left->identity < right->identity; });
                std::string discovery_revision;
                if (!discovery.slot_id.empty())
                {
                    const auto identity =
                        json{{"definition", state.revision()}, {"overrides", local}, {"slot", slot.id}}.dump();
                    discovery_revision = assets::to_string(
                        assets::hash_bytes(std::as_bytes(std::span(identity.data(), identity.size()))));
                    if (!discovery.revision.empty() && discovery.revision != discovery_revision)
                        throw std::runtime_error("Stale function discovery revision; restart discovery");
                    if (discovery.offset > candidates.size())
                        throw std::runtime_error("Function discovery offset is out of range");
                }
                std::size_t next = discovery.slot_id.empty() ? 0 : discovery.offset;
                const auto limit = discovery.slot_id.empty() ? alternative_limit : discovery.limit;
                for (; next < candidates.size(); ++next)
                {
                    const auto& candidate = *candidates[next];
                    if (alternatives.size() >= limit || trials >= 256)
                    {
                        truncated = true;
                        break;
                    }
                    ++trials;
                    auto trial = selections;
                    std::erase_if(trial, [&](const auto& entry) { return entry.slot_id == slot.id; });
                    trial.push_back({.slot_id = slot.id, .function_path = candidate.path});
                    if (rt::compile_material_graph_json(authored.value().graph_json, functions, trial))
                        alternatives.push_back({{"guid", candidate.identity}, {"pathHint", candidate.path}});
                }
                if (!discovery.slot_id.empty())
                    result["functionDiscovery"] = {{"guid", source.guid},
                                                   {"slotId", slot.id},
                                                   {"definitionRevision", state.revision()},
                                                   {"discoveryRevision", discovery_revision},
                                                   {"alternatives", alternatives},
                                                   {"nextOffset", truncated ? json(next) : json(nullptr)}};
                json inputs = json::array(), outputs = json::array();
                for (const auto& pin : slot.inputs)
                    inputs.push_back(pin_json(pin));
                for (const auto& pin : slot.outputs)
                    outputs.push_back(pin_json(pin));
                result["functionSlots"].push_back(
                    {{"id", slot.id},
                     {"name", slot.name},
                     {"defaultFunctionGuid", state.aliases.at(slot.default_function_path)},
                     {"selectedFunctionGuid", slot.selected_function_identity},
                     {"inheritedFunctionGuid", inherited_functions.contains(slot.id)
                                                   ? inherited_functions.at(slot.id)
                                                   : state.aliases.at(slot.default_function_path)},
                     {"overridden", std::ranges::any_of(local,
                                                        [&](const auto& entry) {
                                                            return entry.value("kind", std::string{}) == "function" &&
                                                                   entry.value("slotId", std::string{}) == slot.id;
                                                        })},
                     {"inputs", inputs},
                     {"outputs", outputs},
                     {"alternatives", alternatives},
                     {"alternativesTruncated", truncated}});
            }
            for (const auto& diagnostic : compiled.value().diagnostics)
                result["diagnostics"].push_back({{"code", diagnostic.code}, {"message", diagnostic.message}});
            result["definitionReady"] = true;
        }
    }
    catch (const std::exception& error)
    {
        result["diagnostics"].push_back({{"code", "material-inspection-failed"}, {"message", error.what()}});
    }
    result["definitionRevision"] = state.revision();
    for (const auto& [guid, dependency] : state.sources)
        if (guid != source.guid) result["dependencies"].push_back({{"guid", guid}, {"pathHint", dependency.path}});
    return result.dump();
}
} // namespace arc::editor
