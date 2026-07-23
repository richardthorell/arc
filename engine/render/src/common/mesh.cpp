#include <arc/render/mesh.h>
#include <arc/render/texture.h>

#if defined(ARC_RENDER_HAS_UFBX)
#include <ufbx.h>
#endif

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <cmath>
#include <map>
#include <memory>
#include <optional>
#include <regex>
#include <span>
#include <sstream>
#include <string>
#include <string_view>
#include <unordered_map>

namespace arc::render
{
namespace
{

std::string lowercase(std::string value)
{
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char ch) {
        return static_cast<char>(std::tolower(ch));
    });
    return value;
}

struct buffer_view
{
    std::size_t byte_offset{};
    std::size_t byte_length{};
    std::size_t byte_stride{};
};

struct accessor
{
    std::size_t buffer_view_index{};
    std::size_t byte_offset{};
    std::uint32_t component_type{};
    std::size_t count{};
    std::string type;
};

struct primitive
{
    std::size_t position_accessor{};
    std::size_t normal_accessor{};
    std::size_t texcoord_accessor{};
    std::size_t index_accessor{};
    std::size_t material_index{ material_texture_indices::invalid };
    bool has_normal{};
    bool has_texcoord{};
};

std::uint32_t read_u32_le(const std::vector<std::byte>& bytes, std::size_t offset)
{
    if (offset + sizeof(std::uint32_t) > bytes.size())
        return 0;

    std::uint32_t result{};
    std::memcpy(&result, bytes.data() + offset, sizeof(result));
    return result;
}

std::optional<std::string> read_file(const std::filesystem::path& path, std::vector<std::byte>& bytes)
{
    std::ifstream file(path, std::ios::binary);
    if (!file)
        return "failed to open mesh file";

    file.seekg(0, std::ios::end);
    const auto size = file.tellg();
    if (size <= 0)
        return "mesh file is empty";

    bytes.resize(static_cast<std::size_t>(size));
    file.seekg(0, std::ios::beg);
    file.read(reinterpret_cast<char*>(bytes.data()), size);
    if (!file)
        return "failed to read mesh file";

    return std::nullopt;
}

std::string trim_json_padding(std::string value)
{
    while (!value.empty() && value.back() == ' ')
        value.pop_back();
    return value;
}

mesh_load_result read_glb_chunks(const std::vector<std::byte>& bytes, std::string& json, std::span<const std::byte>& bin)
{
    if (bytes.size() < 20)
        return { .message = "GLB file is too small" };
    if (read_u32_le(bytes, 0) != 0x46546C67)
        return { .message = "file is not a GLB mesh" };
    if (read_u32_le(bytes, 4) != 2)
        return { .message = "only GLB version 2 is supported" };

    const std::uint32_t total_length = read_u32_le(bytes, 8);
    if (total_length > bytes.size())
        return { .message = "GLB length exceeds file size" };

    std::size_t cursor = 12;
    while (cursor + 8 <= total_length)
    {
        const std::uint32_t chunk_length = read_u32_le(bytes, cursor);
        const std::uint32_t chunk_type = read_u32_le(bytes, cursor + 4);
        cursor += 8;
        if (cursor + chunk_length > bytes.size())
            return { .message = "GLB chunk exceeds file size" };

        if (chunk_type == 0x4E4F534A)
        {
            json.assign(
                reinterpret_cast<const char*>(bytes.data() + cursor),
                reinterpret_cast<const char*>(bytes.data() + cursor + chunk_length));
            json = trim_json_padding(std::move(json));
        }
        else if (chunk_type == 0x004E4942)
        {
            bin = std::span<const std::byte>(bytes.data() + cursor, chunk_length);
        }
        cursor += chunk_length;
    }

    if (json.empty())
        return { .message = "GLB JSON chunk is missing" };
    if (bin.empty())
        return { .message = "GLB binary chunk is missing" };
    return {};
}

std::optional<std::string> extract_array(std::string_view json, std::string_view key)
{
    const auto key_pos = json.find('"' + std::string(key) + '"');
    if (key_pos == std::string_view::npos)
        return std::nullopt;

    const auto array_begin = json.find('[', key_pos);
    if (array_begin == std::string_view::npos)
        return std::nullopt;

    int depth = 0;
    for (std::size_t index = array_begin; index < json.size(); ++index)
    {
        if (json[index] == '[')
            ++depth;
        else if (json[index] == ']')
        {
            --depth;
            if (depth == 0)
                return std::string(json.substr(array_begin + 1, index - array_begin - 1));
        }
    }

    return std::nullopt;
}

std::vector<std::string> extract_objects(std::string_view array)
{
    std::vector<std::string> result;
    int depth = 0;
    std::size_t object_begin = std::string_view::npos;

    for (std::size_t index = 0; index < array.size(); ++index)
    {
        if (array[index] == '{')
        {
            if (depth == 0)
                object_begin = index;
            ++depth;
        }
        else if (array[index] == '}')
        {
            --depth;
            if (depth == 0 && object_begin != std::string_view::npos)
            {
                result.emplace_back(array.substr(object_begin, index - object_begin + 1));
                object_begin = std::string_view::npos;
            }
        }
    }

    return result;
}

std::optional<std::size_t> parse_size(std::string_view object, std::string_view key)
{
    const std::regex pattern("\"" + std::string(key) + "\"\\s*:\\s*(\\d+)");
    std::cmatch match;
    if (!std::regex_search(object.data(), object.data() + object.size(), match, pattern))
        return std::nullopt;
    return static_cast<std::size_t>(std::stoull(match[1].str()));
}

std::optional<std::string> parse_string(std::string_view object, std::string_view key)
{
    const std::regex pattern("\"" + std::string(key) + "\"\\s*:\\s*\"([^\"]+)\"");
    std::cmatch match;
    if (!std::regex_search(object.data(), object.data() + object.size(), match, pattern))
        return std::nullopt;
    return match[1].str();
}

std::optional<float> parse_float(std::string_view object, std::string_view key)
{
    const std::regex pattern("\"" + std::string(key) + "\"\\s*:\\s*(-?\\d+(?:\\.\\d+)?(?:[eE][+-]?\\d+)?)");
    std::cmatch match;
    if (!std::regex_search(object.data(), object.data() + object.size(), match, pattern))
        return std::nullopt;
    return std::stof(match[1].str());
}

std::optional<bool> parse_bool(std::string_view object, std::string_view key)
{
    const std::regex pattern("\"" + std::string(key) + "\"\\s*:\\s*(true|false)");
    std::cmatch match;
    if (!std::regex_search(object.data(), object.data() + object.size(), match, pattern))
        return std::nullopt;
    return match[1].str() == "true";
}

std::optional<std::string> extract_object_for_key(std::string_view json, std::string_view key)
{
    const auto key_pos = json.find('"' + std::string(key) + '"');
    if (key_pos == std::string_view::npos)
        return std::nullopt;

    const auto object_begin = json.find('{', key_pos);
    if (object_begin == std::string_view::npos)
        return std::nullopt;

    int depth = 0;
    for (std::size_t index = object_begin; index < json.size(); ++index)
    {
        if (json[index] == '{')
            ++depth;
        else if (json[index] == '}')
        {
            --depth;
            if (depth == 0)
                return std::string(json.substr(object_begin, index - object_begin + 1));
        }
    }
    return std::nullopt;
}

std::optional<std::vector<float>> parse_float_array(std::string_view object, std::string_view key)
{
    const auto key_pos = object.find('"' + std::string(key) + '"');
    if (key_pos == std::string_view::npos)
        return std::nullopt;

    const auto begin = object.find('[', key_pos);
    const auto end = object.find(']', begin);
    if (begin == std::string_view::npos || end == std::string_view::npos || end <= begin)
        return std::nullopt;

    std::vector<float> values;
    const std::string body(object.substr(begin + 1, end - begin - 1));
    const std::regex number("-?\\d+(?:\\.\\d+)?(?:[eE][+-]?\\d+)?");
    for (std::sregex_iterator it(body.begin(), body.end(), number), last; it != last; ++it)
        values.push_back(std::stof((*it)[0].str()));
    return values;
}

std::size_t parse_texture_index(std::string_view object, std::string_view key)
{
    const auto texture_object = extract_object_for_key(object, key);
    if (!texture_object)
        return material_texture_indices::invalid;
    return parse_size(*texture_object, "index").value_or(material_texture_indices::invalid);
}

std::vector<buffer_view> parse_buffer_views(std::string_view json)
{
    std::vector<buffer_view> result;
    const auto array = extract_array(json, "bufferViews");
    if (!array)
        return result;

    for (const auto& object : extract_objects(*array))
    {
        result.push_back({
            .byte_offset = parse_size(object, "byteOffset").value_or(0),
            .byte_length = parse_size(object, "byteLength").value_or(0),
            .byte_stride = parse_size(object, "byteStride").value_or(0)
        });
    }
    return result;
}

std::vector<accessor> parse_accessors(std::string_view json)
{
    std::vector<accessor> result;
    const auto array = extract_array(json, "accessors");
    if (!array)
        return result;

    for (const auto& object : extract_objects(*array))
    {
        result.push_back({
            .buffer_view_index = parse_size(object, "bufferView").value_or(static_cast<std::size_t>(-1)),
            .byte_offset = parse_size(object, "byteOffset").value_or(0),
            .component_type = static_cast<std::uint32_t>(parse_size(object, "componentType").value_or(0)),
            .count = parse_size(object, "count").value_or(0),
            .type = parse_string(object, "type").value_or("")
        });
    }
    return result;
}

std::optional<primitive> parse_first_primitive(std::string_view json)
{
    const auto meshes = extract_array(json, "meshes");
    if (!meshes)
        return std::nullopt;
    const auto primitives = extract_array(*meshes, "primitives");
    if (!primitives)
        return std::nullopt;
    const auto objects = extract_objects(*primitives);
    if (objects.empty())
        return std::nullopt;

    const auto& object = objects.front();
    primitive result{};
    result.position_accessor = parse_size(object, "POSITION").value_or(static_cast<std::size_t>(-1));
    result.normal_accessor = parse_size(object, "NORMAL").value_or(0);
    result.texcoord_accessor = parse_size(object, "TEXCOORD_0").value_or(0);
    result.index_accessor = parse_size(object, "indices").value_or(static_cast<std::size_t>(-1));
    result.material_index = parse_size(object, "material").value_or(material_texture_indices::invalid);
    result.has_normal = object.find("\"NORMAL\"") != std::string::npos;
    result.has_texcoord = object.find("\"TEXCOORD_0\"") != std::string::npos;
    return result;
}

std::vector<std::size_t> parse_texture_sources(std::string_view json)
{
    std::vector<std::size_t> result;
    const auto array = extract_array(json, "textures");
    if (!array)
        return result;

    for (const auto& object : extract_objects(*array))
        result.push_back(parse_size(object, "source").value_or(material_texture_indices::invalid));
    return result;
}

std::vector<texture_data> parse_images(std::string_view json, const std::vector<buffer_view>& views, std::span<const std::byte> bin)
{
    std::vector<texture_data> result;
    const auto array = extract_array(json, "images");
    if (!array)
        return result;

    for (const auto& object : extract_objects(*array))
    {
        texture_data texture;
        texture.name = parse_string(object, "name").value_or("");
        texture.mime_type = parse_string(object, "mimeType").value_or("");
        const auto view_index = parse_size(object, "bufferView");
        if (view_index && *view_index < views.size())
        {
            const auto& view = views[*view_index];
            if (view.byte_offset <= bin.size() && view.byte_length <= bin.size() - view.byte_offset)
            {
                texture.encoded.assign(
                    bin.begin() + static_cast<std::ptrdiff_t>(view.byte_offset),
                    bin.begin() + static_cast<std::ptrdiff_t>(view.byte_offset + view.byte_length));
            }
        }
        result.push_back(std::move(texture));
    }
    return result;
}

std::size_t texture_image_index(const std::vector<std::size_t>& texture_sources, std::size_t texture_index)
{
    if (texture_index == material_texture_indices::invalid || texture_index >= texture_sources.size())
        return material_texture_indices::invalid;
    return texture_sources[texture_index];
}

std::vector<material_import> parse_materials(std::string_view json, const std::vector<std::size_t>& texture_sources)
{
    std::vector<material_import> result;
    const auto array = extract_array(json, "materials");
    if (!array)
        return result;

    for (const auto& object : extract_objects(*array))
    {
        material_import import;
        import.material.name = parse_string(object, "name").value_or("");
        import.material.double_sided = parse_bool(object, "doubleSided").value_or(false);
        import.material.alpha_cutoff = parse_float(object, "alphaCutoff").value_or(0.5f);
        const auto alpha_mode = parse_string(object, "alphaMode").value_or("OPAQUE");
        if (alpha_mode == "MASK")
            import.material.alpha_mode = material_alpha_mode::masked;
        else if (alpha_mode == "BLEND")
            import.material.alpha_mode = material_alpha_mode::blend;

        if (const auto pbr = extract_object_for_key(object, "pbrMetallicRoughness"))
        {
            if (const auto base_color = parse_float_array(*pbr, "baseColorFactor"); base_color && base_color->size() >= 4)
            {
                import.material.base_color = math::vector4f{
                    (*base_color)[0],
                    (*base_color)[1],
                    (*base_color)[2],
                    (*base_color)[3]
                };
            }
            import.material.metallic = parse_float(*pbr, "metallicFactor").value_or(import.material.metallic);
            import.material.roughness = parse_float(*pbr, "roughnessFactor").value_or(import.material.roughness);
            import.textures.base_color = texture_image_index(texture_sources, parse_texture_index(*pbr, "baseColorTexture"));
            import.textures.metallic_roughness = texture_image_index(texture_sources, parse_texture_index(*pbr, "metallicRoughnessTexture"));
        }

        import.textures.normal = texture_image_index(texture_sources, parse_texture_index(object, "normalTexture"));
        import.textures.occlusion = texture_image_index(texture_sources, parse_texture_index(object, "occlusionTexture"));
        import.textures.emissive = texture_image_index(texture_sources, parse_texture_index(object, "emissiveTexture"));
        if (const auto normal_texture = extract_object_for_key(object, "normalTexture"))
            import.material.normal_scale = parse_float(*normal_texture, "scale").value_or(1.0f);
        if (const auto occlusion_texture = extract_object_for_key(object, "occlusionTexture"))
            import.material.occlusion_strength = parse_float(*occlusion_texture, "strength").value_or(1.0f);
        if (const auto emissive = parse_float_array(object, "emissiveFactor"); emissive && emissive->size() >= 3)
            import.material.emissive_factor = math::vector3f{ (*emissive)[0], (*emissive)[1], (*emissive)[2] };

        result.push_back(std::move(import));
    }
    return result;
}

std::size_t component_count(std::string_view type)
{
    if (type == "SCALAR")
        return 1;
    if (type == "VEC2")
        return 2;
    if (type == "VEC3")
        return 3;
    if (type == "VEC4")
        return 4;
    return 0;
}

std::size_t component_size(std::uint32_t component_type)
{
    switch (component_type)
    {
    case 5123:
        return sizeof(std::uint16_t);
    case 5125:
        return sizeof(std::uint32_t);
    case 5126:
        return sizeof(float);
    default:
        return 0;
    }
}

math::vector2f vertex_uv(const mesh_vertex& vertex) noexcept
{
    return { vertex.texcoord[0], vertex.texcoord[1] };
}

math::vector3f vertex_position(const mesh_vertex& vertex) noexcept
{
    return { vertex.position[0], vertex.position[1], vertex.position[2] };
}

math::vector3f vertex_normal(const mesh_vertex& vertex) noexcept
{
    return { vertex.normal[0], vertex.normal[1], vertex.normal[2] };
}

math::vector3f vertex_tangent(const mesh_vertex& vertex) noexcept
{
    return { vertex.tangent[0], vertex.tangent[1], vertex.tangent[2] };
}

math::vector3f safe_normalize(const math::vector3f& value, math::vector3f fallback) noexcept
{
    if (math::length_squared(value) <= 0.000001f)
        return fallback;
    return math::normalize(value);
}

void accumulate_tangent(mesh_vertex& vertex, const math::vector3f& tangent) noexcept
{
    vertex.tangent[0] += tangent[0];
    vertex.tangent[1] += tangent[1];
    vertex.tangent[2] += tangent[2];
}

void generate_tangents(mesh_data& mesh)
{
    if (mesh.vertices.empty() || mesh.indices.size() < 3)
        return;

    for (auto& vertex : mesh.vertices)
        vertex.tangent[0] = vertex.tangent[1] = vertex.tangent[2] = 0.0f;

    for (std::size_t index = 0; index + 2 < mesh.indices.size(); index += 3)
    {
        const auto i0 = mesh.indices[index + 0];
        const auto i1 = mesh.indices[index + 1];
        const auto i2 = mesh.indices[index + 2];
        if (i0 >= mesh.vertices.size() || i1 >= mesh.vertices.size() || i2 >= mesh.vertices.size())
            continue;

        auto& v0 = mesh.vertices[i0];
        auto& v1 = mesh.vertices[i1];
        auto& v2 = mesh.vertices[i2];
        const auto p0 = vertex_position(v0);
        const auto p1 = vertex_position(v1);
        const auto p2 = vertex_position(v2);
        const auto uv0 = vertex_uv(v0);
        const auto uv1 = vertex_uv(v1);
        const auto uv2 = vertex_uv(v2);
        const auto edge1 = math::sub(p1, p0);
        const auto edge2 = math::sub(p2, p0);
        const auto duv1 = math::sub(uv1, uv0);
        const auto duv2 = math::sub(uv2, uv0);
        const float determinant = duv1[0] * duv2[1] - duv2[0] * duv1[1];
        if (std::abs(determinant) <= 0.000001f)
            continue;

        const float inv_det = 1.0f / determinant;
        const auto tangent = math::mul(math::sub(math::mul(edge1, duv2[1]), math::mul(edge2, duv1[1])), inv_det);
        accumulate_tangent(v0, tangent);
        accumulate_tangent(v1, tangent);
        accumulate_tangent(v2, tangent);
    }

    for (auto& vertex : mesh.vertices)
    {
        const auto normal = safe_normalize(vertex_normal(vertex), { 0.0f, 1.0f, 0.0f });
        auto tangent = vertex_tangent(vertex);
        tangent = math::sub(tangent, math::mul(normal, math::dot(normal, tangent)));
        tangent = safe_normalize(tangent, { 1.0f, 0.0f, 0.0f });
        vertex.normal[0] = normal[0];
        vertex.normal[1] = normal[1];
        vertex.normal[2] = normal[2];
        vertex.tangent[0] = tangent[0];
        vertex.tangent[1] = tangent[1];
        vertex.tangent[2] = tangent[2];
        vertex.tangent[3] = 1.0f;
    }
}

const std::byte* accessor_data(
    const accessor& value,
    const std::vector<buffer_view>& views,
    std::span<const std::byte> bin,
    std::size_t& stride)
{
    if (value.buffer_view_index >= views.size())
        return nullptr;

    const auto& view = views[value.buffer_view_index];
    const auto element_size = component_size(value.component_type) * component_count(value.type);
    stride = view.byte_stride == 0 ? element_size : view.byte_stride;
    const auto offset = view.byte_offset + value.byte_offset;
    if (offset >= bin.size() || offset + element_size > bin.size())
        return nullptr;
    return bin.data() + offset;
}

bool read_vec_f32(
    const accessor& value,
    const std::vector<buffer_view>& views,
    std::span<const std::byte> bin,
    std::size_t index,
    float* out,
    std::size_t count)
{
    if (value.component_type != 5126 || component_count(value.type) < count)
        return false;

    std::size_t stride{};
    const std::byte* base = accessor_data(value, views, bin, stride);
    if (!base || index >= value.count)
        return false;

    std::memcpy(out, base + index * stride, sizeof(float) * count);
    return true;
}

std::optional<std::uint32_t> read_index(
    const accessor& value,
    const std::vector<buffer_view>& views,
    std::span<const std::byte> bin,
    std::size_t index)
{
    std::size_t stride{};
    const std::byte* base = accessor_data(value, views, bin, stride);
    if (!base || index >= value.count || value.type != "SCALAR")
        return std::nullopt;

    if (value.component_type == 5123)
    {
        std::uint16_t result{};
        std::memcpy(&result, base + index * stride, sizeof(result));
        return result;
    }
    if (value.component_type == 5125)
    {
        std::uint32_t result{};
        std::memcpy(&result, base + index * stride, sizeof(result));
        return result;
    }
    return std::nullopt;
}

std::string string_from_path(const std::filesystem::path& path)
{
    return path.generic_string();
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

std::string sanitize_asset_name(std::string value, std::string fallback)
{
    if (value.empty())
        value = std::move(fallback);
    for (char& ch : value)
    {
        const unsigned char uch = static_cast<unsigned char>(ch);
        if (!std::isalnum(uch) && ch != '-' && ch != '_')
            ch = '_';
    }
    while (!value.empty() && value.front() == '_')
        value.erase(value.begin());
    while (!value.empty() && value.back() == '_')
        value.pop_back();
    return value.empty() ? "asset" : value;
}

std::filesystem::path default_import_directory(
    const std::filesystem::path& source,
    const scene_import_options& options)
{
    if (!options.import_directory.empty())
        return options.import_directory;
    const auto root = options.asset_root.empty() ? std::filesystem::current_path() / "assets" : options.asset_root;
    return root / "imported" / sanitize_asset_name(source.stem().string(), "scene");
}

std::filesystem::path asset_relative_path(
    const std::filesystem::path& asset_root,
    const std::filesystem::path& path)
{
    std::error_code ec;
    const auto relative = std::filesystem::relative(path, asset_root, ec);
    if (!ec && !relative.empty())
        return relative.lexically_normal();
    return path.filename();
}

bool report_progress(
    const scene_import_progress_callback& callback,
    scene_import_stage stage,
    float progress,
    std::string message)
{
    if (!callback)
        return true;
    return callback({ .stage = stage, .progress = progress, .message = std::move(message) });
}

bool is_cancelled(const scene_import_options& options)
{
    return options.cancel_requested && options.cancel_requested->load();
}

const char* blend_mode_name(material_alpha_mode mode) noexcept
{
    switch (mode)
    {
    case material_alpha_mode::opaque:
        return "opaque";
    case material_alpha_mode::masked:
        return "masked";
    case material_alpha_mode::blend:
        return "blend";
    }
    return "opaque";
}

bool write_material_asset(
    const std::filesystem::path& path,
    const material_import& imported,
    std::string& diagnostic)
{
    std::error_code ec;
    std::filesystem::create_directories(path.parent_path(), ec);
    if (ec)
    {
        diagnostic = "failed to create material folder: " + ec.message();
        return false;
    }

    std::ofstream stream(path, std::ios::binary);
    if (!stream)
    {
        diagnostic = "failed to write material asset: " + path.string();
        return false;
    }

    const auto& material = imported.material;
    const auto write_texture = [&](const char* key, const std::string& value, bool comma) {
        stream << "    \"" << key << "\": \"" << escape_json(value) << "\"" << (comma ? "," : "") << "\n";
    };

    stream << "{\n";
    stream << "  \"version\": 1,\n";
    stream << "  \"name\": \"" << escape_json(material.name) << "\",\n";
    stream << "  \"shader\": \"arc/default_phong\",\n";
    stream << "  \"domain\": \"surface\",\n";
    stream << "  \"blendMode\": \"" << blend_mode_name(material.alpha_mode) << "\",\n";
    stream << "  \"doubleSided\": " << (material.double_sided ? "true" : "false") << ",\n";
    stream << "  \"surface\": {\n";
    stream << "    \"baseColor\": { \"r\": " << material.base_color[0] << ", \"g\": " << material.base_color[1] << ", \"b\": " << material.base_color[2] << ", \"a\": " << material.base_color[3] << " },\n";
    stream << "    \"metallic\": " << material.metallic << ",\n";
    stream << "    \"roughness\": " << material.roughness << ",\n";
    stream << "    \"normalScale\": " << material.normal_scale << ",\n";
    stream << "    \"aoStrength\": " << material.occlusion_strength << ",\n";
    stream << "    \"emissive\": { \"r\": " << material.emissive_factor[0] << ", \"g\": " << material.emissive_factor[1] << ", \"b\": " << material.emissive_factor[2] << " },\n";
    stream << "    \"emissiveStrength\": " << material.emissive_strength << ",\n";
    stream << "    \"alphaCutoff\": " << material.alpha_cutoff << "\n";
    stream << "  },\n";
    stream << "  \"textures\": {\n";
    write_texture("baseColor", imported.texture_paths.base_color, true);
    write_texture("metallicRoughness", imported.texture_paths.metallic_roughness, true);
    write_texture("normal", imported.texture_paths.normal, true);
    write_texture("ao", imported.texture_paths.occlusion, true);
    write_texture("emissive", imported.texture_paths.emissive, true);
    write_texture("height", "", false);
    stream << "  },\n";
    stream << "  \"advanced\": {\n";
    stream << "    \"clearCoat\": " << material.clear_coat_factor << ",\n";
    stream << "    \"clearCoatRoughness\": " << material.clear_coat_roughness << ",\n";
    stream << "    \"sheen\": " << material.sheen_factor << ",\n";
    stream << "    \"transmission\": " << material.transmission_factor << ",\n";
    stream << "    \"subsurface\": " << material.subsurface_factor << ",\n";
    stream << "    \"anisotropy\": " << material.anisotropy_factor << ",\n";
    stream << "    \"anisotropyRotation\": " << material.anisotropy_rotation << ",\n";
    stream << "    \"parallaxHeightScale\": " << material.parallax_height_scale << "\n";
    stream << "  },\n";
    stream << "  \"graph\": null\n";
    stream << "}\n";
    return true;
}

#if defined(ARC_RENDER_HAS_UFBX)

std::string to_string(ufbx_string value)
{
    return value.data && value.length != 0 ? std::string(value.data, value.length) : std::string{};
}

math::vector3f to_vec3(ufbx_vec3 value)
{
    return math::vector3f{ static_cast<float>(value.x), static_cast<float>(value.y), static_cast<float>(value.z) };
}

math::quatf to_quat(ufbx_quat value)
{
    return math::quatf{
        static_cast<float>(value.x),
        static_cast<float>(value.y),
        static_cast<float>(value.z),
        static_cast<float>(value.w)
    };
}

void assign_map_texture(
    material_texture_indices& indices,
    material_texture_paths& paths,
    std::size_t texture_index,
    const std::string& texture_path,
    std::size_t material_texture_indices::* index_member,
    std::string material_texture_paths::* path_member)
{
    if (texture_index == material_texture_indices::invalid)
        return;
    indices.*index_member = texture_index;
    paths.*path_member = texture_path;
}

std::filesystem::path first_existing_texture_path(
    const std::filesystem::path& source_folder,
    const std::filesystem::path& asset_root,
    const ufbx_texture* texture)
{
    if (!texture)
        return {};

    std::vector<std::filesystem::path> candidates;
    const auto filename = to_string(texture->filename);
    const auto absolute = to_string(texture->absolute_filename);
    const auto relative = to_string(texture->relative_filename);
    if (!absolute.empty())
        candidates.emplace_back(absolute);
    if (!filename.empty())
        candidates.emplace_back(filename);
    if (!relative.empty())
        candidates.emplace_back(source_folder / relative);
    if (!relative.empty())
        candidates.emplace_back(asset_root / relative);
    if (!filename.empty())
        candidates.emplace_back(source_folder / filename);
    if (!filename.empty())
        candidates.emplace_back(asset_root / filename);

    for (const auto& candidate : candidates)
    {
        std::error_code ec;
        if (!candidate.empty() && std::filesystem::exists(candidate, ec))
            return candidate;
    }
    return {};
}

std::string texture_extension(const ufbx_texture* texture, const std::filesystem::path& source)
{
    if (!source.extension().empty())
        return source.extension().string();
    if (texture)
    {
        auto relative = std::filesystem::path(to_string(texture->relative_filename));
        if (!relative.extension().empty())
            return relative.extension().string();
        auto filename = std::filesystem::path(to_string(texture->filename));
        if (!filename.extension().empty())
            return filename.extension().string();
    }
    return ".bin";
}

std::size_t import_texture(
    const ufbx_texture* texture,
    scene_import_result& result,
    const scene_import_options& options,
    const std::filesystem::path& source_folder,
    const std::filesystem::path& texture_folder,
    std::unordered_map<const ufbx_texture*, std::size_t>& texture_indices,
    std::string& relative_out)
{
    if (!texture)
        return material_texture_indices::invalid;
    if (const auto found = texture_indices.find(texture); found != texture_indices.end())
    {
        relative_out = result.textures[found->second].source_path.generic_string();
        return found->second;
    }

    std::error_code ec;
    std::filesystem::create_directories(texture_folder, ec);
    if (ec)
    {
        result.diagnostics.push_back("failed to create texture import folder: " + ec.message());
        return material_texture_indices::invalid;
    }

    const auto source_path = first_existing_texture_path(source_folder, options.asset_root, texture);
    const auto base_name = sanitize_asset_name(to_string(texture->name), "texture");
    const auto extension = texture_extension(texture, source_path);
    const auto destination = texture_folder / (base_name + extension);

    bool wrote_file = false;
    if (texture->content.data && texture->content.size != 0)
    {
        std::ofstream out(destination, std::ios::binary);
        if (out)
        {
            out.write(reinterpret_cast<const char*>(texture->content.data), static_cast<std::streamsize>(texture->content.size));
            wrote_file = static_cast<bool>(out);
        }
    }
    else if (!source_path.empty())
    {
        std::filesystem::copy_file(source_path, destination, std::filesystem::copy_options::overwrite_existing, ec);
        wrote_file = !ec;
    }

    if (!wrote_file)
    {
        result.diagnostics.push_back("texture could not be copied or extracted: " + base_name);
        return material_texture_indices::invalid;
    }

    auto loaded = load_texture_asset(destination);
    if (!loaded.succeeded())
    {
        result.diagnostics.push_back("imported texture could not be loaded: " + destination.filename().string() + " (" + loaded.message + ")");
        return material_texture_indices::invalid;
    }

    loaded.texture.source_path = asset_relative_path(options.asset_root, destination);
    relative_out = loaded.texture.source_path.generic_string();
    const auto index = result.textures.size();
    result.textures.push_back(std::move(loaded.texture));
    texture_indices[texture] = index;
    return index;
}

std::size_t import_material(
    const ufbx_material* material,
    scene_import_result& result,
    const scene_import_options& options,
    const std::filesystem::path& source_folder,
    const std::filesystem::path& import_folder,
    std::unordered_map<const ufbx_material*, std::size_t>& material_indices,
    std::unordered_map<const ufbx_texture*, std::size_t>& texture_indices)
{
    if (!material)
        return material_texture_indices::invalid;
    if (const auto found = material_indices.find(material); found != material_indices.end())
        return found->second;

    material_import imported;
    imported.material.name = to_string(material->name);
    if (imported.material.name.empty())
        imported.material.name = "Imported Material";

    const auto& pbr = material->pbr;
    if (pbr.base_color.has_value || pbr.base_color.value_components >= 3)
    {
        imported.material.base_color[0] = static_cast<float>(pbr.base_color.value_vec3.x);
        imported.material.base_color[1] = static_cast<float>(pbr.base_color.value_vec3.y);
        imported.material.base_color[2] = static_cast<float>(pbr.base_color.value_vec3.z);
    }
    else if (material->fbx.diffuse_color.has_value || material->fbx.diffuse_color.value_components >= 3)
    {
        imported.material.base_color[0] = static_cast<float>(material->fbx.diffuse_color.value_vec3.x);
        imported.material.base_color[1] = static_cast<float>(material->fbx.diffuse_color.value_vec3.y);
        imported.material.base_color[2] = static_cast<float>(material->fbx.diffuse_color.value_vec3.z);
    }

    if (pbr.opacity.has_value)
    {
        imported.material.base_color[3] = static_cast<float>(pbr.opacity.value_real);
        if (imported.material.base_color[3] < 0.999f)
            imported.material.alpha_mode = material_alpha_mode::blend;
    }
    imported.material.metallic = pbr.metalness.has_value ? static_cast<float>(pbr.metalness.value_real) : imported.material.metallic;
    imported.material.roughness = pbr.roughness.has_value ? static_cast<float>(pbr.roughness.value_real) : imported.material.roughness;
    if (pbr.emission_color.has_value || pbr.emission_color.value_components >= 3)
    {
        imported.material.emissive_factor = math::vector3f{
            static_cast<float>(pbr.emission_color.value_vec3.x),
            static_cast<float>(pbr.emission_color.value_vec3.y),
            static_cast<float>(pbr.emission_color.value_vec3.z)
        };
    }
    if (pbr.emission_factor.has_value)
        imported.material.emissive_strength = static_cast<float>(pbr.emission_factor.value_real);
    imported.material.double_sided = material->features.double_sided.enabled;

    const auto texture_folder = import_folder / "textures";
    const auto import_map = [&](const ufbx_material_map& map, std::size_t material_texture_indices::* index_member, std::string material_texture_paths::* path_member) {
        if (!map.texture || !map.texture_enabled)
            return;
        std::string relative;
        const auto index = import_texture(map.texture, result, options, source_folder, texture_folder, texture_indices, relative);
        assign_map_texture(imported.textures, imported.texture_paths, index, relative, index_member, path_member);
    };
    import_map(pbr.base_color, &material_texture_indices::base_color, &material_texture_paths::base_color);
    import_map(pbr.metalness, &material_texture_indices::metallic_roughness, &material_texture_paths::metallic_roughness);
    import_map(pbr.roughness, &material_texture_indices::metallic_roughness, &material_texture_paths::metallic_roughness);
    import_map(pbr.normal_map, &material_texture_indices::normal, &material_texture_paths::normal);
    import_map(pbr.ambient_occlusion, &material_texture_indices::occlusion, &material_texture_paths::occlusion);
    import_map(pbr.emission_color, &material_texture_indices::emissive, &material_texture_paths::emissive);

    const auto material_folder = import_folder / "materials";
    imported.asset_path = material_folder / (sanitize_asset_name(imported.material.name, "material") + ".arcmat");
    std::string diagnostic;
    if (!write_material_asset(imported.asset_path, imported, diagnostic))
        result.diagnostics.push_back(diagnostic);

    const auto index = result.materials.size();
    result.materials.push_back(std::move(imported));
    material_indices[material] = index;
    return index;
}

void normalize3(float* values)
{
    const float length = std::sqrt(values[0] * values[0] + values[1] * values[1] + values[2] * values[2]);
    if (length <= 0.000001f)
    {
        values[0] = 0.0f;
        values[1] = 1.0f;
        values[2] = 0.0f;
        return;
    }
    values[0] /= length;
    values[1] /= length;
    values[2] /= length;
}

scene_import_result load_fbx_scene_asset(
    const std::filesystem::path& path,
    const scene_import_options& options,
    const scene_import_progress_callback& progress)
{
    scene_import_result result;
    result.import_directory = default_import_directory(path, options);
    result.manifest_path = result.import_directory / "import.json";
    if (!report_progress(progress, scene_import_stage::loading, 0.0f, "Loading FBX"))
        return { .message = "FBX import cancelled" };

    ufbx_load_opts load_options{};
    load_options.generate_missing_normals = true;
    load_options.normalize_normals = true;
    load_options.use_blender_pbr_material = true;
    load_options.ignore_animation = true;
    load_options.load_external_files = true;
    load_options.ignore_missing_external_files = true;
    if (options.normalize_axes)
        load_options.target_axes = ufbx_axes_right_handed_y_up;
    if (options.normalize_units)
        load_options.target_unit_meters = 1.0f;

    struct progress_user
    {
        const scene_import_progress_callback* callback{};
        const scene_import_options* options{};
    } progress_data{ &progress, &options };
    load_options.progress_cb.fn = [](void* user, const ufbx_progress* fbx_progress) -> ufbx_progress_result {
        auto* data = static_cast<progress_user*>(user);
        if (is_cancelled(*data->options))
            return UFBX_PROGRESS_CANCEL;
        float amount = 0.0f;
        if (fbx_progress && fbx_progress->bytes_total != 0)
            amount = static_cast<float>(static_cast<double>(fbx_progress->bytes_read) / static_cast<double>(fbx_progress->bytes_total)) * 0.35f;
        return report_progress(*data->callback, scene_import_stage::loading, amount, "Loading FBX")
            ? UFBX_PROGRESS_CONTINUE
            : UFBX_PROGRESS_CANCEL;
    };
    load_options.progress_cb.user = &progress_data;
    load_options.progress_interval_hint = 64 * 1024;

    ufbx_error error{};
    ufbx_scene* scene = ufbx_load_file(path.string().c_str(), &load_options, &error);
    if (!scene)
    {
        char buffer[512]{};
        ufbx_format_error(buffer, sizeof(buffer), &error);
        result.message = std::string("FBX import failed: ") + buffer;
        return result;
    }

    struct scene_deleter
    {
        void operator()(ufbx_scene* value) const noexcept { ufbx_free_scene(value); }
    };
    std::unique_ptr<ufbx_scene, scene_deleter> scene_guard(scene);

    std::filesystem::create_directories(result.import_directory);
    std::unordered_map<const ufbx_material*, std::size_t> material_indices;
    std::unordered_map<const ufbx_texture*, std::size_t> texture_indices;
    const auto source_folder = path.parent_path();

    if (!report_progress(progress, scene_import_stage::building_materials, 0.38f, "Importing materials"))
        return { .message = "FBX import cancelled" };

    const auto node_count = std::max<std::size_t>(1, scene->nodes.count);
    for (std::size_t node_index = 0; node_index < scene->nodes.count; ++node_index)
    {
        if (is_cancelled(options))
            return { .message = "FBX import cancelled" };
        const ufbx_node* node = scene->nodes.data[node_index];
        if (!node || !node->mesh || !node->visible)
            continue;

        const ufbx_mesh* mesh = node->mesh;
        const float mesh_progress = 0.45f + 0.45f * static_cast<float>(node_index) / static_cast<float>(node_count);
        if (!report_progress(progress, scene_import_stage::building_meshes, mesh_progress, "Building meshes"))
            return { .message = "FBX import cancelled" };

        ufbx_transform node_transform = ufbx_matrix_to_transform(&node->node_to_world);
        ufbx_matrix normal_to_node = ufbx_matrix_for_normals(&node->geometry_to_node);

        for (std::size_t part_index = 0; part_index < mesh->material_parts.count; ++part_index)
        {
            const ufbx_mesh_part& part = mesh->material_parts.data[part_index];
            if (part.num_triangles == 0)
                continue;

            mesh_data imported_mesh;
            imported_mesh.name = to_string(node->name);
            if (imported_mesh.name.empty())
                imported_mesh.name = to_string(mesh->name);
            if (imported_mesh.name.empty())
                imported_mesh.name = "FBX Mesh";
            if (mesh->material_parts.count > 1)
                imported_mesh.name += " Part " + std::to_string(part_index);
            imported_mesh.vertices.reserve(part.num_triangles * 3);
            imported_mesh.indices.reserve(part.num_triangles * 3);
            const bool has_imported_tangents = mesh->vertex_tangent.exists;

            const ufbx_material* material = nullptr;
            if (part.index < node->materials.count)
                material = node->materials.data[part.index];
            else if (part.index < mesh->materials.count)
                material = mesh->materials.data[part.index];
            const auto material_index = import_material(material, result, options, source_folder, result.import_directory, material_indices, texture_indices);
            imported_mesh.material_index = material_index;

            for (std::size_t face_index = 0; face_index < part.num_faces; ++face_index)
            {
                const ufbx_face face = mesh->faces.data[part.face_indices.data[face_index]];
                if (face.num_indices < 3)
                    continue;
                std::vector<std::uint32_t> tri_indices(face.num_indices * 3);
                const auto tri_count = ufbx_triangulate_face(tri_indices.data(), tri_indices.size(), mesh, face);
                for (std::size_t vertex_index = 0; vertex_index < tri_count * 3; ++vertex_index)
                {
                    const std::uint32_t ix = tri_indices[vertex_index];
                    const ufbx_vec3 position = ufbx_transform_position(&node->geometry_to_node, ufbx_get_vertex_vec3(&mesh->vertex_position, ix));
                    const ufbx_vec3 normal = mesh->vertex_normal.exists
                        ? ufbx_transform_direction(&normal_to_node, ufbx_get_vertex_vec3(&mesh->vertex_normal, ix))
                        : ufbx_vec3{ 0.0, 1.0, 0.0 };
                    const ufbx_vec2 uv = mesh->vertex_uv.exists ? ufbx_get_vertex_vec2(&mesh->vertex_uv, ix) : ufbx_vec2{};
                    const ufbx_vec4 color = mesh->vertex_color.exists ? ufbx_get_vertex_vec4(&mesh->vertex_color, ix) : ufbx_vec4{ 1.0, 1.0, 1.0, 1.0 };

                    mesh_vertex vertex;
                    vertex.position[0] = static_cast<float>(position.x);
                    vertex.position[1] = static_cast<float>(position.y);
                    vertex.position[2] = static_cast<float>(position.z);
                    vertex.normal[0] = static_cast<float>(normal.x);
                    vertex.normal[1] = static_cast<float>(normal.y);
                    vertex.normal[2] = static_cast<float>(normal.z);
                    normalize3(vertex.normal);
                    if (mesh->vertex_tangent.exists)
                    {
                        const ufbx_vec3 tangent = ufbx_transform_direction(&normal_to_node, ufbx_get_vertex_vec3(&mesh->vertex_tangent, ix));
                        vertex.tangent[0] = static_cast<float>(tangent.x);
                        vertex.tangent[1] = static_cast<float>(tangent.y);
                        vertex.tangent[2] = static_cast<float>(tangent.z);
                        vertex.tangent[3] = 1.0f;
                        if (mesh->vertex_bitangent.exists)
                        {
                            const ufbx_vec3 bitangent = ufbx_transform_direction(&normal_to_node, ufbx_get_vertex_vec3(&mesh->vertex_bitangent, ix));
                            const math::vector3f n{ vertex.normal[0], vertex.normal[1], vertex.normal[2] };
                            const math::vector3f t{ vertex.tangent[0], vertex.tangent[1], vertex.tangent[2] };
                            const math::vector3f b{ static_cast<float>(bitangent.x), static_cast<float>(bitangent.y), static_cast<float>(bitangent.z) };
                            vertex.tangent[3] = math::dot(math::cross(n, t), b) < 0.0f ? -1.0f : 1.0f;
                        }
                    }
                    vertex.texcoord[0] = static_cast<float>(uv.x);
                    vertex.texcoord[1] = mesh->vertex_uv.exists ? 1.0f - static_cast<float>(uv.y) : 0.0f;
                    if (mesh->vertex_tangent.exists)
                        vertex.tangent[3] = -vertex.tangent[3];
                    vertex.color[0] = static_cast<float>(color.x);
                    vertex.color[1] = static_cast<float>(color.y);
                    vertex.color[2] = static_cast<float>(color.z);
                    vertex.color[3] = static_cast<float>(color.w);

                    imported_mesh.indices.push_back(static_cast<std::uint32_t>(imported_mesh.vertices.size()));
                    imported_mesh.vertices.push_back(vertex);
                }
            }

            if (imported_mesh.vertices.empty() || imported_mesh.indices.empty())
                continue;

            if (!has_imported_tangents)
                generate_tangents(imported_mesh);
            const auto mesh_index = result.meshes.size();
            result.meshes.push_back(std::move(imported_mesh));
            result.nodes.push_back({
                .name = result.meshes.back().name,
                .mesh_index = mesh_index,
                .material_index = material_index,
                .position = to_vec3(node_transform.translation),
                .rotation = to_quat(node_transform.rotation),
                .scale = to_vec3(node_transform.scale)
            });
        }
    }

    if (!report_progress(progress, scene_import_stage::finalizing, 0.95f, "Writing import manifest"))
        return { .message = "FBX import cancelled" };

    std::ofstream manifest(result.manifest_path, std::ios::binary);
    if (manifest)
    {
        manifest << "{\n";
        manifest << "  \"version\": 1,\n";
        manifest << "  \"source\": \"" << escape_json(path.string()) << "\",\n";
        manifest << "  \"meshes\": " << result.meshes.size() << ",\n";
        manifest << "  \"materials\": " << result.materials.size() << ",\n";
        manifest << "  \"textures\": " << result.textures.size() << ",\n";
        manifest << "  \"nodes\": [\n";
        for (std::size_t index = 0; index < result.nodes.size(); ++index)
        {
            manifest << "    \"" << escape_json(result.nodes[index].name) << "\"";
            manifest << (index + 1 == result.nodes.size() ? "\n" : ",\n");
        }
        manifest << "  ],\n";
        manifest << "  \"diagnostics\": [\n";
        for (std::size_t index = 0; index < result.diagnostics.size(); ++index)
        {
            manifest << "    \"" << escape_json(result.diagnostics[index]) << "\"";
            manifest << (index + 1 == result.diagnostics.size() ? "\n" : ",\n");
        }
        manifest << "  ]\n";
        manifest << "}\n";
    }
    else
    {
        result.diagnostics.push_back("failed to write import manifest");
    }

    report_progress(progress, scene_import_stage::finalizing, 1.0f, "Import complete");
    result.message = result.succeeded()
        ? "loaded FBX scene"
        : "FBX scene contained no static renderable meshes";
    return result;
}

#endif

} // namespace

mesh_load_result load_gltf_mesh(const std::filesystem::path& path)
{
    std::vector<std::byte> bytes;
    if (const auto error = read_file(path, bytes))
        return { .message = *error };

    std::string json;
    std::span<const std::byte> bin;
    if (auto result = read_glb_chunks(bytes, json, bin); !result.message.empty())
        return result;

    const auto views = parse_buffer_views(json);
    const auto accessors = parse_accessors(json);
    const auto primitive = parse_first_primitive(json);
    if (!primitive)
        return { .message = "GLB contains no mesh primitive" };
    if (primitive->position_accessor >= accessors.size() || primitive->index_accessor >= accessors.size())
        return { .message = "GLB mesh primitive references missing accessors" };

    const auto& positions = accessors[primitive->position_accessor];
    if (positions.component_type != 5126 || positions.type != "VEC3")
        return { .message = "GLB POSITION accessor must be float VEC3" };

    mesh_data mesh;
    mesh.name = path.stem().string();
    mesh.material_index = primitive->material_index;
    mesh.vertices.resize(positions.count);

    for (std::size_t index = 0; index < positions.count; ++index)
    {
        if (!read_vec_f32(positions, views, bin, index, mesh.vertices[index].position, 3))
            return { .message = "failed to read POSITION data" };
        if (primitive->has_normal)
        {
            if (primitive->normal_accessor >= accessors.size() ||
                !read_vec_f32(accessors[primitive->normal_accessor], views, bin, index, mesh.vertices[index].normal, 3))
            {
                return { .message = "failed to read NORMAL data" };
            }
        }
        if (primitive->has_texcoord)
        {
            if (primitive->texcoord_accessor >= accessors.size() ||
                !read_vec_f32(accessors[primitive->texcoord_accessor], views, bin, index, mesh.vertices[index].texcoord, 2))
            {
                return { .message = "failed to read TEXCOORD_0 data" };
            }
        }
    }

    const auto& indices = accessors[primitive->index_accessor];
    mesh.indices.reserve(indices.count);
    for (std::size_t index = 0; index < indices.count; ++index)
    {
        const auto value = read_index(indices, views, bin, index);
        if (!value)
            return { .message = "failed to read index data" };
        mesh.indices.push_back(*value);
    }
    generate_tangents(mesh);

    const auto texture_sources = parse_texture_sources(json);
    auto textures = parse_images(json, views, bin);
    auto materials = parse_materials(json, texture_sources);
    return {
        .mesh = std::move(mesh),
        .textures = std::move(textures),
        .materials = std::move(materials),
        .message = "loaded GLB mesh"
    };
}

job_future<mesh_load_result> load_gltf_mesh_async(
    job_system& jobs,
    std::filesystem::path path,
    cancellation_token cancellation)
{
    return jobs.submit_future({
        .name = "render.load_gltf",
        .priority = job_priority::normal,
        .affinity = job_affinity::io_thread,
        .cancellation = cancellation
    }, [path = std::move(path)] {
        return load_gltf_mesh(path);
    });
}

scene_import_result load_scene_asset(
    const std::filesystem::path& path,
    const scene_import_options& options,
    scene_import_progress_callback progress)
{
    const auto extension = lowercase(path.extension().string());
    if (extension == ".glb")
    {
        if (!report_progress(progress, scene_import_stage::loading, 0.05f, "Loading GLB"))
            return { .message = "GLB import cancelled" };
        auto mesh_result = load_gltf_mesh(path);
        scene_import_result result;
        result.import_directory = default_import_directory(path, options);
        result.manifest_path = result.import_directory / "import.json";
        result.message = mesh_result.message;
        if (!mesh_result.succeeded())
            return result;

        result.meshes.push_back(std::move(mesh_result.mesh));
        result.textures = std::move(mesh_result.textures);
        result.materials = std::move(mesh_result.materials);
        result.nodes.push_back({
            .name = result.meshes.front().name.empty() ? path.stem().string() : result.meshes.front().name,
            .mesh_index = 0,
            .material_index = result.meshes.front().material_index
        });
        report_progress(progress, scene_import_stage::finalizing, 1.0f, "Import complete");
        result.message = "loaded GLB scene";
        return result;
    }

    if (extension == ".gltf")
    {
        return {
            .diagnostics = { "text glTF import is not wired yet; export GLB for this importer" },
            .message = "text glTF scene import is not supported yet"
        };
    }

    if (extension == ".fbx")
    {
#if defined(ARC_RENDER_HAS_UFBX)
        return load_fbx_scene_asset(path, options, progress);
#else
        return {
            .diagnostics = { "FBX importer is not available; enable ARC_FETCH_UFBX or ARC_USE_SYSTEM_UFBX." },
            .message = "FBX scene import is unavailable in this build"
        };
#endif
    }

    return { .message = "unsupported scene asset extension: " + extension };
}

scene_import_result load_scene_asset(const std::filesystem::path& path)
{
    scene_import_options options;
    return load_scene_asset(path, options);
}

job_future<scene_import_result> load_scene_asset_async(
    job_system& jobs,
    std::filesystem::path path,
    scene_import_options options,
    scene_import_progress_callback progress,
    cancellation_token cancellation)
{
    return jobs.submit_future({
        .name = "render.import_scene",
        .priority = job_priority::normal,
        .affinity = job_affinity::io_thread,
        .cancellation = cancellation
    }, [
        path = std::move(path),
        options = std::move(options),
        progress = std::move(progress)
    ]() mutable {
        return load_scene_asset(path, options, std::move(progress));
    });
}

} // namespace arc::render
