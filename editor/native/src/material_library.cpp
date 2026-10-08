#include <arc/editor/material_library.h>

#include <arc/editor/editor_interaction.h>
#include <arc/editor/material_preview_realizer.h>
#include <arc/diagnostics/diagnostics.h>
#include <arc/render/primitives.h>
#include <arc/render_tools/material_asset.h>
#include <arc/render/texture.h>
#include <arc/scene/scene.h>

#include <algorithm>
#include <array>
#include <cctype>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <optional>
#include <set>
#include <sstream>

#include <nlohmann/json.hpp>

namespace arc::editor
{
namespace
{

std::filesystem::path canonical_key(const std::filesystem::path& path);
render::texture_handle ensure_texture(editor_material_library& library, render::renderer& renderer,
                                      const std::filesystem::path& path, render::texture_semantic semantic);

std::string read_material_text(const std::filesystem::path& path)
{
    std::ifstream stream(path, std::ios::binary);
    if (!stream) return {};
    std::ostringstream output;
    output << stream.rdbuf();
    return output.str();
}

std::filesystem::path resolve_instance_reference_path(const std::filesystem::path& instance_path,
                                                      const std::filesystem::path& asset_root,
                                                      std::string_view path_hint)
{
    if (path_hint.empty()) return {};
    std::filesystem::path hinted{path_hint};
    if (hinted.is_absolute()) return hinted.lexically_normal();

    const std::array candidates{
        (asset_root / hinted).lexically_normal(),
        (instance_path.parent_path() / hinted).lexically_normal(),
    };
    std::error_code ec;
    for (const auto& candidate : candidates)
        if (std::filesystem::exists(candidate, ec) && !ec) return candidate;

    for (auto current = instance_path.parent_path(); !current.empty(); current = current.parent_path())
    {
        const auto candidate = (current / hinted).lexically_normal();
        ec.clear();
        if (std::filesystem::exists(candidate, ec) && !ec) return candidate;
        const auto parent = current.parent_path();
        if (parent == current) break;
    }
    return (asset_root / hinted).lexically_normal();
}

bool collect_material_function_paths(std::string_view graph_json, std::vector<std::string>& paths)
{
    const auto graph = nlohmann::json::parse(graph_json, nullptr, false);
    if (graph.is_discarded() || !graph.is_object() || !graph.contains("nodes") || !graph["nodes"].is_array())
        return false;
    for (const auto& node : graph["nodes"])
    {
        if (!node.is_object()) continue;
        const auto type = node.value("type", std::string{});
        if (type != "functionCall" && type != "functionSlot") continue;
        const auto values = node.value("values", nlohmann::json::object());
        const auto path = values.value("path", std::string{});
        if (!path.empty()) paths.push_back(path);
    }
    return true;
}

bool load_instance_function_sources(const std::filesystem::path& parent_path, const std::filesystem::path& asset_root,
                                    const render::tools::material_authoring_document& parent,
                                    const render::tools::material_instance_authoring_document& instance,
                                    std::vector<render::tools::material_function_source>& functions,
                                    std::string& message)
{
    std::vector<std::string> pending;
    if (!collect_material_function_paths(parent.graph_json, pending))
    {
        message = "Parent Material graph is malformed while resolving Material Functions";
        return false;
    }
    for (const auto& override_value : instance.function_overrides)
        pending.push_back(override_value.function.path_hint);

    std::set<std::string> visited;
    for (std::size_t index = 0; index < pending.size(); ++index)
    {
        const auto authored_path = pending[index];
        const auto source_path = resolve_instance_reference_path(parent_path, asset_root, authored_path);
        auto key = canonical_key(source_path).generic_string();
        std::transform(key.begin(), key.end(), key.begin(),
                       [](unsigned char value) { return static_cast<char>(std::tolower(value)); });
        if (!visited.insert(key).second) continue;

        const auto source = read_material_text(source_path);
        if (source.empty() || !render::tools::is_material_function_json(source))
        {
            message = "Material Function could not be loaded: " + source_path.generic_string();
            return false;
        }

        std::string identity = key;
        for (const auto& override_value : instance.function_overrides)
        {
            const auto selected =
                resolve_instance_reference_path(parent_path, asset_root, override_value.function.path_hint);
            if (canonical_key(selected) == canonical_key(source_path))
            {
                identity = override_value.function.guid;
                break;
            }
        }

        functions.push_back({.path = source_path.generic_string(), .identity = std::move(identity), .source = source});

        const auto document = nlohmann::json::parse(source, nullptr, false);
        if (document.is_discarded() || !document.is_object()) continue;
        if (document.contains("graph")) collect_material_function_paths(document["graph"].dump(), pending);
    }
    return true;
}

std::optional<render::material_parameter_value>
instance_parameter_value(editor_material_library& library, render::renderer& renderer,
                         const std::filesystem::path& asset_root, const render::shader_parameter_descriptor& parameter,
                         std::string_view value_json)
{
    const auto value = nlohmann::json::parse(value_json, nullptr, false);
    if (value.is_discarded()) return std::nullopt;
    const auto number_array = [&](std::size_t count) -> std::optional<std::vector<float>>
    {
        if (!value.is_array() || value.size() != count) return std::nullopt;
        std::vector<float> result;
        result.reserve(count);
        for (const auto& component : value)
        {
            if (!component.is_number()) return std::nullopt;
            result.push_back(component.get<float>());
        }
        return result;
    };

    switch (parameter.type)
    {
        case render::shader_parameter_type::boolean:
            if (value.is_boolean()) return value.get<bool>();
            break;
        case render::shader_parameter_type::int32:
            if (value.is_number_integer()) return static_cast<std::int32_t>(value.get<std::int64_t>());
            break;
        case render::shader_parameter_type::uint32:
            if (value.is_number_unsigned()) return static_cast<std::uint32_t>(value.get<std::uint64_t>());
            if (value.is_number_integer() && value.get<std::int64_t>() >= 0)
                return static_cast<std::uint32_t>(value.get<std::int64_t>());
            break;
        case render::shader_parameter_type::float32:
            if (value.is_number()) return value.get<float>();
            break;
        case render::shader_parameter_type::float2:
            if (const auto values = number_array(2u)) return math::vector2f{(*values)[0], (*values)[1]};
            break;
        case render::shader_parameter_type::float3:
            if (const auto values = number_array(3u)) return math::vector3f{(*values)[0], (*values)[1], (*values)[2]};
            break;
        case render::shader_parameter_type::float4:
            if (const auto values = number_array(4u))
                return math::vector4f{(*values)[0], (*values)[1], (*values)[2], (*values)[3]};
            break;
        case render::shader_parameter_type::texture_2d:
            if (value.is_string())
            {
                const auto path = resolve_instance_reference_path({}, asset_root, value.get<std::string>());
                return render::resource_handle{
                    ensure_texture(library, renderer, path, render::texture_semantic::generic_color)};
            }
            break;
        default:
            break;
    }
    return std::nullopt;
}

std::filesystem::path canonical_key(const std::filesystem::path& path)
{
    std::error_code ec;
    const auto absolute = std::filesystem::absolute(path, ec);
    return ec ? path.lexically_normal() : absolute.lexically_normal();
}

render::texture_handle ensure_texture(editor_material_library& library, render::renderer& renderer,
                                      const std::filesystem::path& path,
                                      render::texture_semantic semantic = render::texture_semantic::generic_color)
{
    if (path.empty()) return {};
    auto key = canonical_key(path);
    key += "#" + std::to_string(static_cast<unsigned>(semantic));
    for (const auto& [texture_path, handle] : library.textures)
    {
        if (texture_path == key) return handle;
    }

    auto texture_result = render::load_texture_asset(path);
    if (!texture_result.succeeded())
    {
        arc::diagnostics::warn("editor.materials", "Texture asset could not be loaded: " + path.string() + " (" +
                                                       texture_result.message + ")");
        return {};
    }

    texture_result.texture.semantic = semantic;
    texture_result.texture.color_space = render::required_color_space(semantic);
    if (texture_result.texture.format == render::texture_format::rgba8_srgb ||
        texture_result.texture.format == render::texture_format::rgba8_unorm)
    {
        texture_result.texture.format = texture_result.texture.color_space == render::texture_color_space::srgb
                                            ? render::texture_format::rgba8_srgb
                                            : render::texture_format::rgba8_unorm;
    }
    const auto handle = renderer.create_texture(std::move(texture_result.texture));
    library.textures.push_back({key, handle});
    return handle;
}

render::texture_handle ensure_packed_terrain_surface(editor_material_library& library, render::renderer& renderer,
                                                     const std::filesystem::path& asset_root,
                                                     const material_asset& asset, std::size_t layer_index)
{
    const auto& paths = asset.terrain_layers[layer_index];
    if (!paths.packed_aorh.empty())
        return ensure_texture(library, renderer, resolve_material_texture_path(asset_root, paths.packed_aorh),
                              render::texture_semantic::metallic_roughness);

    auto cache_key = canonical_key(asset.path);
    cache_key += ".terrain-aorh-" + std::to_string(layer_index);
    for (const auto& [texture_path, handle] : library.textures)
    {
        if (texture_path == cache_key) return handle;
    }

    const auto load_channel = [&](const std::string& relative_path,
                                  std::string_view channel) -> std::optional<render::texture_data>
    {
        if (relative_path.empty()) return std::nullopt;
        const auto path = resolve_material_texture_path(asset_root, relative_path);
        auto loaded = render::load_texture_asset(path);
        if (!loaded.succeeded() || !loaded.texture.has_pixels())
        {
            arc::diagnostics::warn("editor.materials", "Terrain " + std::string(channel) +
                                                           " map could not be packed: " + path.generic_string());
            return std::nullopt;
        }
        return std::move(loaded.texture);
    };

    auto ao = load_channel(paths.ao, "AO");
    auto roughness = load_channel(paths.roughness, "roughness");
    auto height = load_channel(paths.height, "height");
    const render::texture_data* reference = ao ? &*ao : roughness ? &*roughness : height ? &*height : nullptr;
    if (reference == nullptr) return {};

    const auto pixel_count = static_cast<std::size_t>(reference->width) * reference->height;
    const auto channel_usable = [&](const std::optional<render::texture_data>& source)
    {
        return source && source->width == reference->width && source->height == reference->height &&
               source->pixels.size() >= pixel_count * 4u;
    };
    if ((ao && !channel_usable(ao)) || (roughness && !channel_usable(roughness)) || (height && !channel_usable(height)))
        arc::diagnostics::warn("editor.materials",
                               "Terrain AORH source dimensions differ; mismatched channels use explicit defaults");

    render::texture_data packed;
    packed.name = asset.name + " " + asset.material.terrain_layers[layer_index].name + " AORH";
    packed.source_path = cache_key;
    packed.width = reference->width;
    packed.height = reference->height;
    packed.format = render::texture_format::rgba8_unorm;
    packed.color_space = render::texture_color_space::linear;
    packed.semantic = render::texture_semantic::metallic_roughness;
    packed.pixels.resize(pixel_count * 4u);
    const auto channel_value =
        [&](const std::optional<render::texture_data>& source, std::size_t pixel, std::uint8_t fallback)
    { return channel_usable(source) ? std::to_integer<std::uint8_t>(source->pixels[pixel * 4u]) : fallback; };
    const auto roughness_fallback = static_cast<std::uint8_t>(
        std::clamp(asset.material.terrain_layers[layer_index].roughness, 0.0f, 1.0f) * 255.0f + 0.5f);
    for (std::size_t pixel = 0; pixel < pixel_count; ++pixel)
    {
        packed.pixels[pixel * 4u + 0u] = std::byte{channel_value(ao, pixel, 255u)};
        packed.pixels[pixel * 4u + 1u] = std::byte{channel_value(roughness, pixel, roughness_fallback)};
        packed.pixels[pixel * 4u + 2u] = std::byte{channel_value(height, pixel, 128u)};
        packed.pixels[pixel * 4u + 3u] = std::byte{255u};
    }

    const auto handle = renderer.create_texture(std::move(packed));
    if (handle.valid()) library.textures.push_back({std::move(cache_key), handle});
    return handle;
}

void resolve_texture_handles(editor_material_library& library, render::renderer& renderer,
                             const std::filesystem::path& asset_root, material_asset& asset)
{
    if (asset.material.domain == render::material_domain::terrain)
    {
        for (std::size_t layer_index = 0; layer_index < asset.terrain_layers.size(); ++layer_index)
        {
            const auto& paths = asset.terrain_layers[layer_index];
            auto& layer = asset.material.terrain_layers[layer_index];
            layer.base_color_texture =
                ensure_texture(library, renderer, resolve_material_texture_path(asset_root, paths.base_color),
                               render::texture_semantic::base_color);
            layer.normal_texture =
                ensure_texture(library, renderer, resolve_material_texture_path(asset_root, paths.normal),
                               render::texture_semantic::normal);
            layer.packed_surface_texture =
                ensure_packed_terrain_surface(library, renderer, asset_root, asset, layer_index);
        }
        return;
    }

    asset.material.base_color_texture =
        ensure_texture(library, renderer, resolve_material_texture_path(asset_root, asset.textures.base_color),
                       render::texture_semantic::base_color);
    asset.material.metallic_roughness_texture =
        ensure_texture(library, renderer, resolve_material_texture_path(asset_root, asset.textures.metallic_roughness),
                       render::texture_semantic::metallic_roughness);
    asset.material.normal_texture =
        ensure_texture(library, renderer, resolve_material_texture_path(asset_root, asset.textures.normal),
                       render::texture_semantic::normal);
    asset.material.occlusion_texture =
        ensure_texture(library, renderer, resolve_material_texture_path(asset_root, asset.textures.ao),
                       render::texture_semantic::occlusion);
    asset.material.emissive_texture =
        ensure_texture(library, renderer, resolve_material_texture_path(asset_root, asset.textures.emissive),
                       render::texture_semantic::emissive);
    asset.material.clear_coat_texture =
        ensure_texture(library, renderer, resolve_material_texture_path(asset_root, asset.textures.clear_coat),
                       render::texture_semantic::clear_coat);
    asset.material.clear_coat_roughness_texture = ensure_texture(
        library, renderer, resolve_material_texture_path(asset_root, asset.textures.clear_coat_roughness),
        render::texture_semantic::clear_coat);
    asset.material.clear_coat_normal_texture =
        ensure_texture(library, renderer, resolve_material_texture_path(asset_root, asset.textures.clear_coat_normal),
                       render::texture_semantic::normal);
    asset.material.anisotropy_texture =
        ensure_texture(library, renderer, resolve_material_texture_path(asset_root, asset.textures.anisotropy),
                       render::texture_semantic::anisotropy);
    asset.material.subsurface_texture =
        ensure_texture(library, renderer, resolve_material_texture_path(asset_root, asset.textures.subsurface),
                       render::texture_semantic::thickness);
    asset.material.thickness_texture =
        ensure_texture(library, renderer, resolve_material_texture_path(asset_root, asset.textures.thickness),
                       render::texture_semantic::thickness);
    asset.material.transmission_texture =
        ensure_texture(library, renderer, resolve_material_texture_path(asset_root, asset.textures.transmission),
                       render::texture_semantic::transmission);
}

editor_material_record* find_record(editor_material_library& library, const std::filesystem::path& path)
{
    const auto key = canonical_key(path);
    for (auto& record : library.materials)
    {
        if (canonical_key(record.path) == key) return &record;
    }
    return nullptr;
}

} // namespace

void resolve_material_runtime_textures(editor_material_library& library, render::renderer& renderer,
                                       const std::filesystem::path& asset_root,
                                       const std::filesystem::path& material_path,
                                       const std::vector<std::string>& texture_sources,
                                       render::material_descriptor& material)
{
    material.runtime_textures.clear();
    material.runtime_textures.resize(texture_sources.size());
    for (std::size_t slot = 0; slot < texture_sources.size(); ++slot)
    {
        if (texture_sources[slot].empty()) continue;
        std::filesystem::path source(texture_sources[slot]);
        if (!source.is_absolute())
        {
            const auto beside_material = (material_path.parent_path() / source).lexically_normal();
            std::error_code ec;
            if (std::filesystem::exists(beside_material, ec) && !ec)
                source = beside_material;
            else if (!asset_root.empty() && source.begin() != source.end() &&
                     source.begin()->string() == asset_root.filename().string())
                source = (asset_root.parent_path() / source).lexically_normal();
            else
                source = (asset_root / source).lexically_normal();
        }
        material.runtime_textures[slot] =
            ensure_texture(library, renderer, source, render::texture_semantic::generic_color);
    }
}

bool is_material_asset_path(const std::filesystem::path& path)
{
    auto ext = path.extension().string();
    std::transform(ext.begin(), ext.end(), ext.begin(),
                   [](unsigned char ch) { return static_cast<char>(std::tolower(ch)); });
    return ext == ".arcmat" || ext == ".arcmatinst";
}

bool is_texture_asset_path(const std::filesystem::path& path)
{
    return render::is_supported_texture_asset(path);
}

bool assign_texture_to_material_slot(material_editor_state& editor, material_texture_slot slot,
                                     const std::filesystem::path& asset_root, const std::filesystem::path& texture_path,
                                     std::string* message)
{
    if (!editor.open)
    {
        if (message) *message = "material editor is not open";
        return false;
    }

    std::filesystem::path relative_path = texture_path;
    if (texture_path.is_absolute())
    {
        std::error_code ec;
        relative_path = std::filesystem::relative(texture_path, asset_root, ec);
        if (ec) relative_path = texture_path.filename();
    }
    relative_path = relative_path.lexically_normal();
    if (!is_texture_asset_path(relative_path))
    {
        if (message) *message = "asset is not a supported texture";
        return false;
    }

    auto& textures = editor.working.textures;
    const auto value = relative_path.generic_string();
    switch (slot)
    {
        case material_texture_slot::base_color:
            textures.base_color = value;
            break;
        case material_texture_slot::metallic_roughness:
            textures.metallic_roughness = value;
            break;
        case material_texture_slot::normal:
            textures.normal = value;
            break;
        case material_texture_slot::ao:
            textures.ao = value;
            break;
        case material_texture_slot::emissive:
            textures.emissive = value;
            break;
        case material_texture_slot::height:
            textures.height = value;
            break;
        case material_texture_slot::clear_coat:
            textures.clear_coat = value;
            break;
        case material_texture_slot::clear_coat_roughness:
            textures.clear_coat_roughness = value;
            break;
        case material_texture_slot::clear_coat_normal:
            textures.clear_coat_normal = value;
            break;
        case material_texture_slot::anisotropy:
            textures.anisotropy = value;
            break;
        case material_texture_slot::subsurface:
            textures.subsurface = value;
            break;
        case material_texture_slot::thickness:
            textures.thickness = value;
            break;
        case material_texture_slot::transmission:
            textures.transmission = value;
            break;
    }

    editor.dirty = true;
    if (message) *message = "assigned texture " + value;
    return true;
}

bool create_default_material_asset(const std::filesystem::path& path, const std::filesystem::path& asset_root,
                                   std::string& message)
{
    auto asset = make_default_material_asset(path.stem().string());
    asset.path = path;
    return save_material_asset(asset, asset_root, message);
}

render::material_handle load_material_for_editor(editor_material_library& library, render::renderer& renderer,
                                                 const std::filesystem::path& asset_root,
                                                 const std::filesystem::path& path, material_asset* out_asset)
{
    auto extension = path.extension().string();
    std::transform(extension.begin(), extension.end(), extension.begin(),
                   [](unsigned char ch) { return static_cast<char>(std::tolower(ch)); });
    if (extension != ".arcmatinst")
    {
        if (auto* record = find_record(library, path))
        {
            if (out_asset) *out_asset = record->asset;
            return record->material;
        }
    }

    if (extension == ".arcmatinst")
    {
        const auto source = read_material_text(path);
        auto authored = render::tools::parse_material_instance_authoring_json(source);
        if (!authored)
        {
            arc::diagnostics::error("editor.materials", "Failed to load Material Instance '" + path.string() +
                                                            "': " + authored.error().message);
            return {};
        }

        const auto parent_path = resolve_instance_reference_path(path, asset_root, authored.value().parent.path_hint);
        const auto parent_source = read_material_text(parent_path);
        auto parent_authored = render::tools::parse_material_authoring_json(parent_source);
        if (!parent_authored)
        {
            arc::diagnostics::error("editor.materials", "Material Instance parent is invalid: " + parent_path.string());
            return {};
        }

        material_asset parent_asset;
        const auto parent = load_material_for_editor(library, renderer, asset_root, parent_path, &parent_asset);
        if (!parent.valid())
        {
            arc::diagnostics::error("editor.materials",
                                    "Material Instance parent could not be realized: " + parent_path.string());
            return {};
        }

        std::vector<render::tools::material_function_source> functions;
        std::vector<render::tools::material_function_slot_override> slot_overrides;
        slot_overrides.reserve(authored.value().function_overrides.size());
        for (const auto& override_value : authored.value().function_overrides)
            slot_overrides.push_back(
                {.slot_id = override_value.slot_id, .function_path = override_value.function.path_hint});

        material_preview_descriptor_result realized;
        if (parent_authored.value().graph_json.empty())
        {
            if (!slot_overrides.empty())
            {
                arc::diagnostics::error("editor.materials",
                                        "Handwritten Material parents cannot expose Material Function Slots");
                return {};
            }
            realized.material = parent_asset.material;
            realized.succeeded = true;
        }
        else
        {
            std::string function_message;
            if (!load_instance_function_sources(parent_path, asset_root, parent_authored.value(), authored.value(),
                                                functions, function_message))
            {
                arc::diagnostics::error("editor.materials", function_message);
                return {};
            }
            realized =
                realize_material_preview_descriptor(parent_source, authored.value().name, functions, slot_overrides);
            if (!realized.succeeded)
            {
                arc::diagnostics::error("editor.materials",
                                        "Material Instance specialization failed: " + realized.message);
                return {};
            }
            resolve_material_runtime_textures(library, renderer, asset_root, parent_path, realized.texture_sources,
                                              realized.material);
        }

        if (!realized.material.runtime_program)
        {
            arc::diagnostics::error("editor.materials",
                                    "Material Instance requires a compiled runtime parameter layout");
            return {};
        }

        render::material_instance_descriptor instance;
        instance.parent = parent;
        instance.name = authored.value().name;
        const auto& layout = realized.material.runtime_program->parameters;
        for (const auto& override_value : authored.value().parameter_overrides)
        {
            const auto parameter =
                std::ranges::find(layout, override_value.parameter_id, &render::shader_parameter_descriptor::id);
            if (parameter == layout.end()) continue;
            auto value = instance_parameter_value(library, renderer, asset_root, *parameter, override_value.value_json);
            if (!value) continue;
            instance.overrides.push_back(
                {.id = override_value.parameter_id, .name = parameter->name, .value = std::move(*value)});
        }
        for (const auto& function_override : authored.value().function_overrides)
        {
            for (const auto& input_override : function_override.input_overrides)
            {
                const auto stable_name = "slot::" + function_override.slot_id + "::" + function_override.function.guid +
                                         "::" + input_override.pin_id;
                const auto parameter_id = render::make_shader_parameter_id(stable_name);
                const auto parameter =
                    std::ranges::find(layout, parameter_id, &render::shader_parameter_descriptor::id);
                if (parameter == layout.end()) continue;
                auto value =
                    instance_parameter_value(library, renderer, asset_root, *parameter, input_override.value_json);
                if (!value) continue;
                instance.overrides.push_back({.id = parameter_id, .name = parameter->name, .value = std::move(*value)});
            }
        }

        render::material_definition_descriptor definition;
        definition.material = realized.material;
        definition.parameter_layout = layout;
        auto resolved = render::resolve_material_instance(definition, instance);
        if (!resolved)
        {
            arc::diagnostics::error("editor.materials",
                                    "Material Instance could not be resolved: " + resolved.error().message);
            return {};
        }

        auto asset = parent_asset;
        asset.name = authored.value().name;
        asset.path = path;
        asset.material = std::move(resolved).value();
        render::material_handle handle{};
        if (auto* record = find_record(library, path))
        {
            if (renderer.material_alive(record->material))
            {
                handle = record->material;
                if (!renderer.update_material(handle, asset.material)) return {};
            }
            else
            {
                handle = renderer.create_material(asset.material);
                record->material = handle;
            }
            record->asset = asset;
        }
        else
        {
            handle = renderer.create_material(asset.material);
            library.materials.push_back({canonical_key(path), asset, handle});
        }
        if (out_asset) *out_asset = asset;
        return handle;
    }

    material_asset asset;
    std::string message;
    if (!load_material_asset(path, asset_root, asset, message))
    {
        arc::diagnostics::error("editor.materials", "Failed to load material '" + path.string() + "': " + message);
        return {};
    }

    auto realized = load_material_preview_descriptor(path);
    const bool legacy_document = !realized.succeeded && realized.message.starts_with("Legacy material field '");
    if (!realized.succeeded && !legacy_document)
    {
        arc::diagnostics::error("editor.materials",
                                "Failed to realize material '" + path.string() + "': " + realized.message);
        return {};
    }
    if (legacy_document)
    {
        arc::diagnostics::info("editor.materials", "Loading legacy material descriptor for '" + path.string() +
                                                       "' until native material serialization is graph-only");
    }
    else
    {
        asset.material = std::move(realized.material);
        resolve_material_runtime_textures(library, renderer, asset_root, path, realized.texture_sources,
                                          asset.material);
        for (const auto& diagnostic : realized.diagnostics)
            arc::diagnostics::info("editor.materials",
                                   "Material realization note for '" + path.string() + "': " + diagnostic);
    }

    resolve_texture_handles(library, renderer, asset_root, asset);
    const auto handle = renderer.create_material(asset.material);
    library.materials.push_back({canonical_key(path), asset, handle});
    if (out_asset) *out_asset = asset;
    return handle;
}

bool open_material_editor(material_editor_state& editor, editor_material_library& library, render::renderer& renderer,
                          const std::filesystem::path& asset_root, const std::filesystem::path& path,
                          std::string& message)
{
    material_asset asset;
    const auto handle = load_material_for_editor(library, renderer, asset_root, path, &asset);
    if (!handle.valid())
    {
        message = "material could not be loaded";
        return false;
    }

    editor.open = true;
    editor.dirty = false;
    editor.working = asset;
    editor.saved = asset;
    editor.material = handle;
    if (!editor.preview_sphere.valid())
        editor.preview_sphere = renderer.create_mesh(render::make_uv_sphere_mesh(0.75f, 48, 24));
    if (!editor.preview_plane.valid()) editor.preview_plane = renderer.create_mesh(render::make_plane_mesh(1.6f));
    if (!editor.preview_cube.valid()) editor.preview_cube = renderer.create_mesh(render::make_cube_mesh(1.1f));
    message = "opened material editor";
    return true;
}

bool save_material_editor(material_editor_state& editor, editor_material_library& library, render::renderer& renderer,
                          const std::filesystem::path& asset_root, std::string& message)
{
    if (!editor.open)
    {
        message = "material editor is not open";
        return false;
    }

    editor.working.material.name = editor.working.name;
    if (!save_material_asset(editor.working, asset_root, message)) return false;

    resolve_texture_handles(library, renderer, asset_root, editor.working);
    if (editor.material.valid())
        renderer.update_material(editor.material, editor.working.material);
    else
        editor.material = renderer.create_material(editor.working.material);

    if (auto* record = find_record(library, editor.working.path))
    {
        record->asset = editor.working;
        record->material = editor.material;
    }
    else
    {
        library.materials.push_back({canonical_key(editor.working.path), editor.working, editor.material});
    }

    editor.saved = editor.working;
    editor.dirty = false;
    return true;
}

bool update_material_editor_live_material(material_editor_state& editor, editor_material_library& library,
                                          render::renderer& renderer, const std::filesystem::path& asset_root,
                                          std::string* message)
{
    if (!editor.open)
    {
        if (message) *message = "material editor is not open";
        return false;
    }

    editor.working.material.name = editor.working.name;
    resolve_texture_handles(library, renderer, asset_root, editor.working);
    if (editor.material.valid())
        renderer.update_material(editor.material, editor.working.material);
    else
        editor.material = renderer.create_material(editor.working.material);

    if (auto* record = find_record(library, editor.working.path))
    {
        record->asset = editor.working;
        record->material = editor.material;
    }

    if (message) *message = "updated live material";
    return true;
}

bool apply_material_to_selected(ecs::world& scene, ecs::entity selected, render::material_handle material)
{
    if (!material.valid()) return false;
    auto* mesh = scene.try_get<scene::mesh_renderer_component>(selected);
    if (!mesh) return false;
    mesh->material = material;
    return true;
}

bool apply_material_asset_to_entity(editor_material_library& library, render::renderer& renderer,
                                    const std::filesystem::path& asset_root, const std::filesystem::path& material_path,
                                    ecs::world& scene, ecs::entity entity, std::string* message)
{
    if (!is_material_asset_path(material_path))
    {
        if (message) *message = "asset is not a material";
        return false;
    }

    auto* mesh = scene.try_get<scene::mesh_renderer_component>(entity);
    if (!mesh)
    {
        if (message) *message = "target entity has no mesh renderer";
        return false;
    }

    const auto resolved_path = material_path.is_absolute() ? material_path : asset_root / material_path;
    material_asset asset;
    const auto material = load_material_for_editor(library, renderer, asset_root, resolved_path, &asset);
    if (!material.valid())
    {
        if (message) *message = "material could not be loaded";
        return false;
    }

    mesh->material = material;
    if (message) *message = "applied material " + asset.name;
    return true;
}

ecs::entity apply_material_asset_to_viewport_hit(editor_material_library& library, render::renderer& renderer,
                                                 const std::filesystem::path& asset_root,
                                                 const std::filesystem::path& material_path, ecs::world& scene,
                                                 const editor_ray& ray, ecs::entity& selected, std::string* message)
{
    const auto picked = pick_bounded_entity(scene, ray);
    if (!picked.valid())
    {
        if (message) *message = "material drop did not hit a renderable entity";
        return {};
    }

    if (!apply_material_asset_to_entity(library, renderer, asset_root, material_path, scene, picked, message))
        return {};

    select_entity(scene, picked, selected);
    return picked;
}

} // namespace arc::editor
