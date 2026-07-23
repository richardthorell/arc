#include <arc/editor/material_asset.h>

#include <algorithm>
#include <cctype>
#include <charconv>
#include <fstream>
#include <optional>
#include <sstream>

namespace arc::editor
{
namespace
{

std::string read_text_file(const std::filesystem::path& path)
{
    std::ifstream stream(path, std::ios::binary);
    if (!stream)
        return {};
    std::ostringstream buffer;
    buffer << stream.rdbuf();
    return buffer.str();
}

std::string escape_json(std::string_view value)
{
    std::string result;
    result.reserve(value.size());
    for (const char ch : value)
    {
        switch (ch)
        {
        case '\\':
            result += "\\\\";
            break;
        case '"':
            result += "\\\"";
            break;
        case '\n':
            result += "\\n";
            break;
        case '\r':
            break;
        default:
            result += ch;
            break;
        }
    }
    return result;
}

std::string lowercase(std::string value)
{
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char ch) {
        return static_cast<char>(std::tolower(ch));
    });
    return value;
}

std::string blend_mode_to_string(render::material_alpha_mode mode)
{
    switch (mode)
    {
    case render::material_alpha_mode::opaque:
        return "opaque";
    case render::material_alpha_mode::masked:
        return "masked";
    case render::material_alpha_mode::blend:
        return "blend";
    }
    return "opaque";
}

render::material_alpha_mode blend_mode_from_string(std::string value)
{
    value = lowercase(std::move(value));
    if (value == "masked" || value == "mask")
        return render::material_alpha_mode::masked;
    if (value == "blend" || value == "transparent")
        return render::material_alpha_mode::blend;
    return render::material_alpha_mode::opaque;
}

std::size_t find_key(std::string_view json, std::string_view key)
{
    return json.find('"' + std::string(key) + '"');
}

std::optional<std::string> string_value(std::string_view json, std::string_view key)
{
    const auto key_pos = find_key(json, key);
    if (key_pos == std::string_view::npos)
        return std::nullopt;
    const auto colon = json.find(':', key_pos);
    const auto begin = json.find('"', colon + 1);
    if (colon == std::string_view::npos || begin == std::string_view::npos)
        return std::nullopt;
    std::string result;
    for (std::size_t index = begin + 1; index < json.size(); ++index)
    {
        const char ch = json[index];
        if (ch == '\\' && index + 1 < json.size())
        {
            result += json[index + 1];
            ++index;
            continue;
        }
        if (ch == '"')
            return result;
        result += ch;
    }
    return std::nullopt;
}

std::optional<float> float_value(std::string_view json, std::string_view key)
{
    const auto key_pos = find_key(json, key);
    if (key_pos == std::string_view::npos)
        return std::nullopt;
    const auto colon = json.find(':', key_pos);
    if (colon == std::string_view::npos)
        return std::nullopt;
    std::size_t begin = colon + 1;
    while (begin < json.size() && std::isspace(static_cast<unsigned char>(json[begin])))
        ++begin;
    std::size_t end = begin;
    while (end < json.size() && (std::isdigit(static_cast<unsigned char>(json[end])) || json[end] == '-' || json[end] == '+' || json[end] == '.' || json[end] == 'e' || json[end] == 'E'))
        ++end;
    float result{};
    const auto parsed = std::from_chars(json.data() + begin, json.data() + end, result);
    if (parsed.ec != std::errc{})
        return std::nullopt;
    return result;
}

std::optional<bool> bool_value(std::string_view json, std::string_view key)
{
    const auto key_pos = find_key(json, key);
    if (key_pos == std::string_view::npos)
        return std::nullopt;
    const auto colon = json.find(':', key_pos);
    if (colon == std::string_view::npos)
        return std::nullopt;
    const auto value = json.substr(colon + 1, 8);
    if (value.find("true") != std::string_view::npos)
        return true;
    if (value.find("false") != std::string_view::npos)
        return false;
    return std::nullopt;
}

std::optional<std::string> object_for_key(std::string_view json, std::string_view key)
{
    const auto key_pos = find_key(json, key);
    if (key_pos == std::string_view::npos)
        return std::nullopt;
    const auto object_begin = json.find('{', key_pos);
    if (object_begin == std::string_view::npos)
        return std::nullopt;

    int depth = 0;
    bool in_string = false;
    for (std::size_t index = object_begin; index < json.size(); ++index)
    {
        const char ch = json[index];
        if (ch == '"' && (index == 0 || json[index - 1] != '\\'))
            in_string = !in_string;
        if (in_string)
            continue;
        if (ch == '{')
            ++depth;
        else if (ch == '}')
        {
            --depth;
            if (depth == 0)
                return std::string(json.substr(object_begin, index - object_begin + 1));
        }
    }
    return std::nullopt;
}

void assign_vec3(std::string_view object, const char* x, const char* y, const char* z, math::vector3f& value)
{
    if (auto parsed = float_value(object, x))
        value[0] = *parsed;
    if (auto parsed = float_value(object, y))
        value[1] = *parsed;
    if (auto parsed = float_value(object, z))
        value[2] = *parsed;
}

void assign_vec4(std::string_view object, const char* r, const char* g, const char* b, const char* a, math::vector4f& value)
{
    if (auto parsed = float_value(object, r))
        value[0] = *parsed;
    if (auto parsed = float_value(object, g))
        value[1] = *parsed;
    if (auto parsed = float_value(object, b))
        value[2] = *parsed;
    if (auto parsed = float_value(object, a))
        value[3] = *parsed;
}

} // namespace

material_asset make_default_material_asset(std::string name)
{
    material_asset asset;
    asset.name = std::move(name);
    asset.material.name = asset.name;
    asset.material.base_color = { 0.78f, 0.80f, 0.84f, 1.0f };
    asset.material.roughness = 0.62f;
    return asset;
}

bool load_material_asset(
    const std::filesystem::path& path,
    const std::filesystem::path& asset_root,
    material_asset& out_asset,
    std::string& message)
{
    const auto json = read_text_file(path);
    if (json.empty())
    {
        message = "material file could not be read";
        return false;
    }

    material_asset asset = make_default_material_asset(path.stem().string());
    asset.path = path;
    asset.version = static_cast<int>(float_value(json, "version").value_or(1.0f));
    asset.name = string_value(json, "name").value_or(asset.name);
    asset.shader = string_value(json, "shader").value_or(asset.shader);
    asset.domain = string_value(json, "domain").value_or(asset.domain);
    asset.material.domain = lowercase(asset.domain) == "terrain"
        ? render::material_domain::terrain : render::material_domain::surface;
    asset.material.name = asset.name;
    asset.material.alpha_mode = blend_mode_from_string(string_value(json, "blendMode").value_or("opaque"));
    asset.material.double_sided = bool_value(json, "doubleSided").value_or(asset.material.double_sided);
    asset.graph_reserved = json.find("\"graph\"") != std::string::npos;

    if (auto surface = object_for_key(json, "surface"))
    {
        if (auto base = object_for_key(*surface, "baseColor"))
            assign_vec4(*base, "r", "g", "b", "a", asset.material.base_color);
        if (auto parsed = float_value(*surface, "metallic"))
            asset.material.metallic = *parsed;
        if (auto parsed = float_value(*surface, "roughness"))
            asset.material.roughness = *parsed;
        if (auto parsed = float_value(*surface, "alphaCutoff"))
            asset.material.alpha_cutoff = *parsed;
        if (auto emissive = object_for_key(*surface, "emissive"))
            assign_vec3(*emissive, "r", "g", "b", asset.material.emissive_factor);
        if (auto parsed = float_value(*surface, "emissiveStrength"))
            asset.material.emissive_strength = *parsed;
        if (auto parsed = float_value(*surface, "normalScale"))
            asset.material.normal_scale = *parsed;
        if (auto parsed = float_value(*surface, "aoStrength"))
            asset.material.occlusion_strength = *parsed;
    }

    if (auto advanced = object_for_key(json, "advanced"))
    {
        if (auto parsed = float_value(*advanced, "clearCoat"))
            asset.material.clear_coat_factor = *parsed;
        if (auto parsed = float_value(*advanced, "clearCoatRoughness"))
            asset.material.clear_coat_roughness = *parsed;
        if (auto parsed = float_value(*advanced, "sheen"))
            asset.material.sheen_factor = *parsed;
        if (auto parsed = float_value(*advanced, "transmission"))
            asset.material.transmission_factor = *parsed;
        if (auto parsed = float_value(*advanced, "subsurface"))
            asset.material.subsurface_factor = *parsed;
        if (auto parsed = float_value(*advanced, "anisotropy"))
            asset.material.anisotropy_factor = *parsed;
        if (auto parsed = float_value(*advanced, "anisotropyRotation"))
            asset.material.anisotropy_rotation = *parsed;
        if (auto parsed = float_value(*advanced, "parallaxHeightScale"))
            asset.material.parallax_height_scale = *parsed;
    }

    if (auto textures = object_for_key(json, "textures"))
    {
        asset.textures.base_color = string_value(*textures, "baseColor").value_or("");
        asset.textures.metallic_roughness = string_value(*textures, "metallicRoughness").value_or("");
        asset.textures.normal = string_value(*textures, "normal").value_or("");
        asset.textures.ao = string_value(*textures, "ao").value_or("");
        asset.textures.emissive = string_value(*textures, "emissive").value_or("");
        asset.textures.height = string_value(*textures, "height").value_or("");
    }

    if (auto terrain_layers = object_for_key(json, "terrainLayers"))
    {
        for (std::size_t layer_index = 0; layer_index < asset.terrain_layers.size(); ++layer_index)
        {
            const auto key = "layer" + std::to_string(layer_index);
            const auto layer = object_for_key(*terrain_layers, key);
            if (!layer)
                continue;
            auto& desc = asset.material.terrain_layers[layer_index];
            auto& paths = asset.terrain_layers[layer_index];
            desc.name = string_value(*layer, "name").value_or(desc.name);
            paths.base_color = string_value(*layer, "baseColor").value_or("");
            paths.normal = string_value(*layer, "normal").value_or("");
            paths.roughness = string_value(*layer, "roughnessTexture").value_or("");
            paths.ao = string_value(*layer, "ao").value_or("");
            paths.height = string_value(*layer, "height").value_or("");
            paths.packed_aorh = string_value(*layer, "packedAorh").value_or("");
            if (auto tint = object_for_key(*layer, "tint"))
                assign_vec4(*tint, "r", "g", "b", "a", desc.tint);
            desc.world_scale = float_value(*layer, "worldScale").value_or(desc.world_scale);
            desc.roughness = float_value(*layer, "roughness").value_or(desc.roughness);
        }
    }

    (void)asset_root;
    out_asset = std::move(asset);
    message = "loaded material asset";
    return true;
}

bool save_material_asset(
    const material_asset& asset,
    const std::filesystem::path& asset_root,
    std::string& message)
{
    if (asset.path.empty())
    {
        message = "material asset has no path";
        return false;
    }

    std::filesystem::create_directories(asset.path.parent_path());
    std::ofstream stream(asset.path, std::ios::binary);
    if (!stream)
    {
        message = "material file could not be opened for writing";
        return false;
    }

    const auto write_texture = [&](const char* key, const std::string& value, bool comma) {
        stream << "    \"" << key << "\": \"" << escape_json(value) << "\"" << (comma ? "," : "") << "\n";
    };

    stream << "{\n";
    stream << "  \"version\": " << asset.version << ",\n";
    stream << "  \"name\": \"" << escape_json(asset.name) << "\",\n";
    stream << "  \"shader\": \"" << escape_json(asset.shader) << "\",\n";
    stream << "  \"domain\": \"" << escape_json(asset.domain) << "\",\n";
    stream << "  \"blendMode\": \"" << blend_mode_to_string(asset.material.alpha_mode) << "\",\n";
    stream << "  \"doubleSided\": " << (asset.material.double_sided ? "true" : "false") << ",\n";
    stream << "  \"surface\": {\n";
    stream << "    \"baseColor\": { \"r\": " << asset.material.base_color[0] << ", \"g\": " << asset.material.base_color[1] << ", \"b\": " << asset.material.base_color[2] << ", \"a\": " << asset.material.base_color[3] << " },\n";
    stream << "    \"metallic\": " << asset.material.metallic << ",\n";
    stream << "    \"roughness\": " << asset.material.roughness << ",\n";
    stream << "    \"normalScale\": " << asset.material.normal_scale << ",\n";
    stream << "    \"aoStrength\": " << asset.material.occlusion_strength << ",\n";
    stream << "    \"emissive\": { \"r\": " << asset.material.emissive_factor[0] << ", \"g\": " << asset.material.emissive_factor[1] << ", \"b\": " << asset.material.emissive_factor[2] << " },\n";
    stream << "    \"emissiveStrength\": " << asset.material.emissive_strength << ",\n";
    stream << "    \"alphaCutoff\": " << asset.material.alpha_cutoff << "\n";
    stream << "  },\n";
    stream << "  \"textures\": {\n";
    write_texture("baseColor", asset.textures.base_color, true);
    write_texture("metallicRoughness", asset.textures.metallic_roughness, true);
    write_texture("normal", asset.textures.normal, true);
    write_texture("ao", asset.textures.ao, true);
    write_texture("emissive", asset.textures.emissive, true);
    write_texture("height", asset.textures.height, false);
    stream << "  },\n";
    if (asset.material.domain == render::material_domain::terrain || lowercase(asset.domain) == "terrain")
    {
        stream << "  \"terrainLayers\": {\n";
        for (std::size_t layer_index = 0; layer_index < asset.terrain_layers.size(); ++layer_index)
        {
            const auto& desc = asset.material.terrain_layers[layer_index];
            const auto& paths = asset.terrain_layers[layer_index];
            stream << "    \"layer" << layer_index << "\": { "
                << "\"name\": \"" << escape_json(desc.name) << "\", "
                << "\"baseColor\": \"" << escape_json(paths.base_color) << "\", "
                << "\"normal\": \"" << escape_json(paths.normal) << "\", "
                << "\"roughnessTexture\": \"" << escape_json(paths.roughness) << "\", "
                << "\"ao\": \"" << escape_json(paths.ao) << "\", "
                << "\"height\": \"" << escape_json(paths.height) << "\", "
                << "\"packedAorh\": \"" << escape_json(paths.packed_aorh) << "\", "
                << "\"tint\": { \"r\": " << desc.tint[0] << ", \"g\": " << desc.tint[1]
                << ", \"b\": " << desc.tint[2] << ", \"a\": " << desc.tint[3] << " }, "
                << "\"worldScale\": " << desc.world_scale << ", \"roughness\": " << desc.roughness << " }"
                << (layer_index + 1u < asset.terrain_layers.size() ? "," : "") << "\n";
        }
        stream << "  },\n";
    }
    stream << "  \"advanced\": {\n";
    stream << "    \"clearCoat\": " << asset.material.clear_coat_factor << ",\n";
    stream << "    \"clearCoatRoughness\": " << asset.material.clear_coat_roughness << ",\n";
    stream << "    \"sheen\": " << asset.material.sheen_factor << ",\n";
    stream << "    \"transmission\": " << asset.material.transmission_factor << ",\n";
    stream << "    \"subsurface\": " << asset.material.subsurface_factor << ",\n";
    stream << "    \"anisotropy\": " << asset.material.anisotropy_factor << ",\n";
    stream << "    \"anisotropyRotation\": " << asset.material.anisotropy_rotation << ",\n";
    stream << "    \"parallaxHeightScale\": " << asset.material.parallax_height_scale << "\n";
    stream << "  },\n";
    stream << "  \"graph\": null\n";
    stream << "}\n";

    (void)asset_root;
    message = "saved material asset";
    return true;
}

std::filesystem::path resolve_material_texture_path(
    const std::filesystem::path& asset_root,
    const std::string& relative_path)
{
    if (relative_path.empty())
        return {};
    std::filesystem::path path(relative_path);
    if (path.is_absolute())
        return path;
    return asset_root / path;
}

} // namespace arc::editor
