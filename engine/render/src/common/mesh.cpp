#include <arc/render/mesh.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <optional>
#include <regex>
#include <span>
#include <string>
#include <string_view>

namespace arc::render
{
namespace
{

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
    result.has_normal = object.find("\"NORMAL\"") != std::string::npos;
    result.has_texcoord = object.find("\"TEXCOORD_0\"") != std::string::npos;
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

    return { .mesh = std::move(mesh), .message = "loaded GLB mesh" };
}

} // namespace arc::render
