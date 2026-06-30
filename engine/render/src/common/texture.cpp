#include <arc/render/texture.h>

#include <algorithm>
#include <cctype>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <initializer_list>
#include <string_view>

namespace arc::render
{
namespace
{

constexpr std::uint32_t fourcc(char a, char b, char c, char d) noexcept
{
    return static_cast<std::uint32_t>(static_cast<unsigned char>(a)) |
        (static_cast<std::uint32_t>(static_cast<unsigned char>(b)) << 8u) |
        (static_cast<std::uint32_t>(static_cast<unsigned char>(c)) << 16u) |
        (static_cast<std::uint32_t>(static_cast<unsigned char>(d)) << 24u);
}

std::uint32_t read_u32(const std::vector<std::byte>& bytes, std::size_t offset) noexcept
{
    if (offset + sizeof(std::uint32_t) > bytes.size())
        return 0;
    std::uint32_t value{};
    std::memcpy(&value, bytes.data() + offset, sizeof(value));
    return value;
}

std::string lowercase(std::string value)
{
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char ch) {
        return static_cast<char>(std::tolower(ch));
    });
    return value;
}

bool contains_any(std::string_view value, std::initializer_list<std::string_view> needles) noexcept
{
    for (const auto needle : needles)
    {
        if (value.find(needle) != std::string_view::npos)
            return true;
    }
    return false;
}

texture_format with_srgb(texture_format format) noexcept
{
    switch (format)
    {
    case texture_format::rgba8_unorm:
        return texture_format::rgba8_srgb;
    case texture_format::bc1_rgba_unorm:
        return texture_format::bc1_rgba_srgb;
    case texture_format::bc2_rgba_unorm:
        return texture_format::bc2_rgba_srgb;
    case texture_format::bc3_rgba_unorm:
        return texture_format::bc3_rgba_srgb;
    case texture_format::bc7_rgba_unorm:
        return texture_format::bc7_rgba_srgb;
    default:
        return format;
    }
}

texture_format with_linear(texture_format format) noexcept
{
    switch (format)
    {
    case texture_format::rgba8_srgb:
        return texture_format::rgba8_unorm;
    case texture_format::bc1_rgba_srgb:
        return texture_format::bc1_rgba_unorm;
    case texture_format::bc2_rgba_srgb:
        return texture_format::bc2_rgba_unorm;
    case texture_format::bc3_rgba_srgb:
        return texture_format::bc3_rgba_unorm;
    case texture_format::bc7_rgba_srgb:
        return texture_format::bc7_rgba_unorm;
    default:
        return format;
    }
}

void apply_filename_color_space(texture_data& texture, const std::filesystem::path& path)
{
    const auto name = lowercase((path.filename().string() + " " + texture.name));
    if (contains_any(name, { "normal", "_n.", "_nor", "roughness", "metallic", "metalness", "metallicroughness", "occlusion", "_ao", "ambientocclusion" }))
    {
        texture.format = with_linear(texture.format);
        return;
    }
    if (contains_any(name, { "basecolor", "base_color", "albedo", "diffuse", "emissive", "emission" }))
        texture.format = with_srgb(texture.format);
}

std::string mime_type_for_path(const std::filesystem::path& path)
{
    const auto ext = lowercase(path.extension().string());
    if (ext == ".png")
        return "image/png";
    if (ext == ".jpg" || ext == ".jpeg")
        return "image/jpeg";
    if (ext == ".tga")
        return "image/tga";
    if (ext == ".hdr")
        return "image/vnd.radiance";
    if (ext == ".dds")
        return "image/vnd-ms.dds";
    return "application/octet-stream";
}

std::vector<std::byte> read_binary_file(const std::filesystem::path& path)
{
    std::ifstream stream(path, std::ios::binary);
    if (!stream)
        return {};
    stream.seekg(0, std::ios::end);
    const auto size = stream.tellg();
    if (size <= 0)
        return {};
    stream.seekg(0, std::ios::beg);
    std::vector<std::byte> bytes(static_cast<std::size_t>(size));
    stream.read(reinterpret_cast<char*>(bytes.data()), size);
    return stream ? bytes : std::vector<std::byte>{};
}

bool format_block_info(texture_format format, std::uint32_t& block_width, std::uint32_t& block_bytes) noexcept
{
    block_width = 4;
    switch (format)
    {
    case texture_format::bc1_rgba_unorm:
    case texture_format::bc1_rgba_srgb:
    case texture_format::bc4_r_unorm:
        block_bytes = 8;
        return true;
    case texture_format::bc2_rgba_unorm:
    case texture_format::bc2_rgba_srgb:
    case texture_format::bc3_rgba_unorm:
    case texture_format::bc3_rgba_srgb:
    case texture_format::bc5_rg_unorm:
    case texture_format::bc6h_rgb_ufloat:
    case texture_format::bc7_rgba_unorm:
    case texture_format::bc7_rgba_srgb:
        block_bytes = 16;
        return true;
    default:
        block_width = 1;
        block_bytes = 0;
        return false;
    }
}

std::size_t mip_payload_size(texture_format format, std::uint32_t width, std::uint32_t height)
{
    std::uint32_t block_width{};
    std::uint32_t block_bytes{};
    if (format_block_info(format, block_width, block_bytes))
    {
        const auto blocks_x = std::max(1u, (width + block_width - 1u) / block_width);
        const auto blocks_y = std::max(1u, (height + block_width - 1u) / block_width);
        return static_cast<std::size_t>(blocks_x) * blocks_y * block_bytes;
    }

    if (format == texture_format::rgba8_unorm || format == texture_format::rgba8_srgb)
        return static_cast<std::size_t>(width) * height * 4u;
    if (format == texture_format::rgba16f)
        return static_cast<std::size_t>(width) * height * 8u;
    if (format == texture_format::rgba32f)
        return static_cast<std::size_t>(width) * height * 16u;
    return 0;
}

bool map_dxgi_format(std::uint32_t dxgi, texture_format& format, bool& compressed) noexcept
{
    compressed = true;
    switch (dxgi)
    {
    case 71:
        format = texture_format::bc1_rgba_unorm;
        return true;
    case 72:
        format = texture_format::bc1_rgba_srgb;
        return true;
    case 74:
        format = texture_format::bc2_rgba_unorm;
        return true;
    case 75:
        format = texture_format::bc2_rgba_srgb;
        return true;
    case 77:
        format = texture_format::bc3_rgba_unorm;
        return true;
    case 78:
        format = texture_format::bc3_rgba_srgb;
        return true;
    case 80:
        format = texture_format::bc4_r_unorm;
        return true;
    case 83:
        format = texture_format::bc5_rg_unorm;
        return true;
    case 95:
        format = texture_format::bc6h_rgb_ufloat;
        return true;
    case 98:
        format = texture_format::bc7_rgba_unorm;
        return true;
    case 99:
        format = texture_format::bc7_rgba_srgb;
        return true;
    case 28:
        format = texture_format::rgba8_unorm;
        compressed = false;
        return true;
    case 29:
        format = texture_format::rgba8_srgb;
        compressed = false;
        return true;
    case 10:
        format = texture_format::rgba16f;
        compressed = false;
        return true;
    case 2:
        format = texture_format::rgba32f;
        compressed = false;
        return true;
    default:
        return false;
    }
}

bool map_legacy_format(
    std::uint32_t flags,
    std::uint32_t four_cc,
    std::uint32_t rgb_bit_count,
    std::uint32_t r_mask,
    std::uint32_t g_mask,
    std::uint32_t b_mask,
    std::uint32_t a_mask,
    texture_format& format,
    bool& compressed,
    bool& has_dx10_header) noexcept
{
    constexpr std::uint32_t ddpf_fourcc = 0x00000004;
    constexpr std::uint32_t ddpf_rgb = 0x00000040;

    has_dx10_header = false;
    compressed = true;
    if ((flags & ddpf_fourcc) != 0)
    {
        if (four_cc == fourcc('D', 'X', '1', '0'))
        {
            has_dx10_header = true;
            return true;
        }
        if (four_cc == fourcc('D', 'X', 'T', '1'))
            format = texture_format::bc1_rgba_unorm;
        else if (four_cc == fourcc('D', 'X', 'T', '3'))
            format = texture_format::bc2_rgba_unorm;
        else if (four_cc == fourcc('D', 'X', 'T', '5'))
            format = texture_format::bc3_rgba_unorm;
        else if (four_cc == fourcc('A', 'T', 'I', '1') || four_cc == fourcc('B', 'C', '4', 'U'))
            format = texture_format::bc4_r_unorm;
        else if (four_cc == fourcc('A', 'T', 'I', '2') || four_cc == fourcc('B', 'C', '5', 'U'))
            format = texture_format::bc5_rg_unorm;
        else
            return false;
        return true;
    }

    compressed = false;
    if ((flags & ddpf_rgb) != 0 && rgb_bit_count == 32 &&
        r_mask == 0x000000ff && g_mask == 0x0000ff00 && b_mask == 0x00ff0000 && a_mask == 0xff000000)
    {
        format = texture_format::rgba8_unorm;
        return true;
    }
    return false;
}

} // namespace

texture_load_result parse_dds_texture(const std::vector<std::byte>& bytes, std::string name)
{
    if (bytes.size() < 128)
        return { .message = "DDS file is too small" };
    if (read_u32(bytes, 0) != fourcc('D', 'D', 'S', ' '))
        return { .message = "file is not a DDS texture" };
    if (read_u32(bytes, 4) != 124)
        return { .message = "DDS header size is invalid" };
    if (read_u32(bytes, 76) != 32)
        return { .message = "DDS pixel format size is invalid" };

    texture_format format{};
    bool compressed{};
    bool has_dx10_header{};
    const bool legacy_format = map_legacy_format(
        read_u32(bytes, 80),
        read_u32(bytes, 84),
        read_u32(bytes, 88),
        read_u32(bytes, 92),
        read_u32(bytes, 96),
        read_u32(bytes, 100),
        read_u32(bytes, 104),
        format,
        compressed,
        has_dx10_header);
    if (!legacy_format)
        return { .message = "DDS pixel format is not supported" };

    std::size_t payload_offset = 128;
    std::uint32_t array_layers = 1;
    if (has_dx10_header)
    {
        if (bytes.size() < 148)
            return { .message = "DDS DX10 header is truncated" };
        if (!map_dxgi_format(read_u32(bytes, 128), format, compressed))
            return { .message = "DDS DXGI format is not supported" };
        array_layers = std::max(1u, read_u32(bytes, 140));
        payload_offset = 148;
    }

    const std::uint32_t width = read_u32(bytes, 16);
    const std::uint32_t height = read_u32(bytes, 12);
    const std::uint32_t mip_count = std::max(1u, read_u32(bytes, 28));
    if (width == 0 || height == 0)
        return { .message = "DDS dimensions are invalid" };
    if (array_layers != 1)
        return { .message = "DDS texture arrays are not supported yet" };

    texture_data texture;
    texture.name = std::move(name);
    texture.width = width;
    texture.height = height;
    texture.format = format;
    texture.mime_type = "image/vnd-ms.dds";
    texture.array_layers = array_layers;
    texture.compressed = compressed;
    texture.dds = true;

    std::size_t cursor = payload_offset;
    std::uint32_t mip_width = width;
    std::uint32_t mip_height = height;
    for (std::uint32_t mip = 0; mip < mip_count; ++mip)
    {
        const auto size = mip_payload_size(format, mip_width, mip_height);
        if (size == 0)
            return { .message = "DDS format has no known payload size" };
        if (cursor > bytes.size() || size > bytes.size() - cursor)
            return { .message = "DDS mip payload is truncated" };
        texture.mips.push_back({
            .width = mip_width,
            .height = mip_height,
            .offset = cursor - payload_offset,
            .size = size
        });
        cursor += size;
        mip_width = std::max(1u, mip_width / 2u);
        mip_height = std::max(1u, mip_height / 2u);
    }

    texture.encoded.assign(bytes.begin() + static_cast<std::ptrdiff_t>(payload_offset), bytes.begin() + static_cast<std::ptrdiff_t>(cursor));
    return { .texture = std::move(texture), .message = "loaded DDS texture" };
}

texture_load_result load_texture_asset(const std::filesystem::path& path)
{
    auto bytes = read_binary_file(path);
    if (bytes.empty())
        return { .message = "texture file could not be read" };

    if (lowercase(path.extension().string()) == ".dds")
    {
        auto result = parse_dds_texture(bytes, path.filename().string());
        if (result.succeeded())
        {
            result.texture.name = path.filename().string();
            result.texture.source_path = path;
            apply_filename_color_space(result.texture, path);
        }
        return result;
    }

    texture_data texture;
    texture.name = path.filename().string();
    texture.source_path = path;
    texture.mime_type = mime_type_for_path(path);
    texture.format = lowercase(path.extension().string()) == ".hdr"
        ? texture_format::rgba32f
        : texture_format::rgba8_srgb;
    texture.encoded = std::move(bytes);
    apply_filename_color_space(texture, path);
    return { .texture = std::move(texture), .message = "loaded encoded texture" };
}

bool is_supported_texture_asset(const std::filesystem::path& path)
{
    const auto ext = lowercase(path.extension().string());
    return ext == ".png" || ext == ".jpg" || ext == ".jpeg" || ext == ".tga" || ext == ".hdr" || ext == ".dds";
}

} // namespace arc::render
