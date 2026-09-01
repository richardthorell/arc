from pathlib import Path

p = Path('engine/render/inc/arc/render/texture.h')
s = p.read_text()
anchor = '''struct [[nodiscard]] texture_load_result
{
    texture_data texture;
    std::string message;

    /**
     * @brief Return whether the texture contains usable decoded or encoded data.
     */
    bool succeeded() const noexcept
    {
        return texture.has_pixels() || texture.has_encoded_mips() || !texture.encoded.empty();
    }
};
'''
insert = anchor + '''
/**
 * @brief Lightweight texture source metadata without decoding pixel payloads.
 */
struct [[nodiscard]] texture_asset_info
{
    std::uint32_t width{};
    std::uint32_t height{};
    texture_format format{texture_format::rgba8_unorm};
    std::uint32_t mip_count{};
    std::string message;

    bool succeeded() const noexcept
    {
        return width > 0 && height > 0;
    }
};
'''
if anchor not in s:
    raise SystemExit('texture_load_result anchor missing')
s = s.replace(anchor, insert, 1)
decl = 'texture_load_result load_texture_asset(const std::filesystem::path& path);\n'
repl = '''texture_load_result load_texture_asset(const std::filesystem::path& path);

/**
 * @brief Inspect dimensions, format, and mip metadata without decoding texture pixels.
 */
texture_asset_info inspect_texture_asset(const std::filesystem::path& path);
'''
if decl not in s:
    raise SystemExit('load_texture_asset declaration missing')
s = s.replace(decl, repl, 1)
p.write_text(s)

p = Path('engine/render/src/common/texture.cpp')
s = p.read_text()
anchor = '''std::vector<std::byte> read_binary_file(const std::filesystem::path& path)
{
    std::ifstream stream(path, std::ios::binary);
    if (!stream) return {};
    stream.seekg(0, std::ios::end);
    const auto size = stream.tellg();
    if (size <= 0) return {};
    stream.seekg(0, std::ios::beg);
    std::vector<std::byte> bytes(static_cast<std::size_t>(size));
    stream.read(reinterpret_cast<char*>(bytes.data()), size);
    return stream ? bytes : std::vector<std::byte>{};
}
'''
insert = anchor + '''
std::vector<std::byte> read_binary_prefix(const std::filesystem::path& path, std::size_t maximum_bytes)
{
    std::ifstream stream(path, std::ios::binary);
    if (!stream || maximum_bytes == 0) return {};
    std::vector<std::byte> bytes(maximum_bytes);
    stream.read(reinterpret_cast<char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
    const auto read = stream.gcount();
    if (read <= 0) return {};
    bytes.resize(static_cast<std::size_t>(read));
    return bytes;
}

std::uint32_t generated_mip_count(std::uint32_t width, std::uint32_t height) noexcept
{
    if (width == 0 || height == 0) return 0;
    std::uint32_t count{1};
    while (width > 1 || height > 1)
    {
        width = std::max(1u, width / 2u);
        height = std::max(1u, height / 2u);
        ++count;
    }
    return count;
}
'''
if anchor not in s:
    raise SystemExit('read_binary_file anchor missing')
s = s.replace(anchor, insert, 1)

anchor = '''texture_load_result load_texture_asset(const std::filesystem::path& path)
{
    auto bytes = read_binary_file(path);
    if (bytes.empty()) return {.message = "texture file could not be read"};
    return load_texture_asset_bytes(std::move(bytes), path);
}
'''
inspect = '''texture_asset_info inspect_texture_asset(const std::filesystem::path& path)
{
    const auto extension = lowercase(path.extension().string());
    if (extension == ".dds")
    {
        const auto bytes = read_binary_prefix(path, 148);
        if (bytes.size() < 128) return {.message = "DDS header is truncated"};
        if (read_u32(bytes, 0) != fourcc('D', 'D', 'S', ' ')) return {.message = "file is not a DDS texture"};
        if (read_u32(bytes, 4) != 124 || read_u32(bytes, 76) != 32) return {.message = "DDS header is invalid"};

        texture_format format{};
        bool compressed{};
        bool has_dx10_header{};
        if (!map_legacy_format(read_u32(bytes, 80), read_u32(bytes, 84), read_u32(bytes, 88), read_u32(bytes, 92),
                               read_u32(bytes, 96), read_u32(bytes, 100), read_u32(bytes, 104), format, compressed,
                               has_dx10_header))
            return {.message = "DDS pixel format is not supported"};
        if (has_dx10_header)
        {
            if (bytes.size() < 148) return {.message = "DDS DX10 header is truncated"};
            if (!map_dxgi_format(read_u32(bytes, 128), format, compressed))
                return {.message = "DDS DXGI format is not supported"};
            if (std::max(1u, read_u32(bytes, 140)) != 1u)
                return {.message = "DDS texture arrays are not supported yet"};
        }

        const auto width = read_u32(bytes, 16);
        const auto height = read_u32(bytes, 12);
        if (width == 0 || height == 0) return {.message = "DDS dimensions are invalid"};
        texture_data metadata;
        metadata.name = path.filename().string();
        metadata.format = format;
        apply_filename_color_space(metadata, path);
        return {.width = width,
                .height = height,
                .format = metadata.format,
                .mip_count = std::max(1u, read_u32(bytes, 28)),
                .message = "inspected DDS texture"};
    }

#if defined(ARC_RENDER_HAS_STB)
    int width{};
    int height{};
    int channels{};
    const auto native_path = path.string();
    if (stbi_info(native_path.c_str(), &width, &height, &channels) == 0 || width <= 0 || height <= 0)
        return {.message = "image metadata inspection failed: " +
                           std::string(stbi_failure_reason() ? stbi_failure_reason() : "unknown image error")};
    const bool hdr = stbi_is_hdr(native_path.c_str()) != 0;
    texture_data metadata;
    metadata.name = path.filename().string();
    metadata.format = hdr ? texture_format::rgba32f : texture_format::rgba8_srgb;
    metadata.color_space = hdr ? texture_color_space::linear : texture_color_space::srgb;
    metadata.semantic = hdr ? texture_semantic::environment : texture_semantic::generic_color;
    apply_filename_color_space(metadata, path);
    return {.width = static_cast<std::uint32_t>(width),
            .height = static_cast<std::uint32_t>(height),
            .format = metadata.format,
            .mip_count = hdr ? 0u : generated_mip_count(static_cast<std::uint32_t>(width),
                                                        static_cast<std::uint32_t>(height)),
            .message = "inspected texture metadata"};
#else
    return {.message = "texture metadata inspection requires an image decoder for non-DDS assets"};
#endif
}

''' + anchor
if anchor not in s:
    raise SystemExit('load_texture_asset definition missing')
s = s.replace(anchor, inspect, 1)
p.write_text(s)

p = Path('editor/native/src/arc_host_base.inc')
s = p.read_text()
old = '''                    const auto texture = render::load_texture_asset(absolute_path);
                    if (texture.succeeded())
                    {
                        host_asset.width = texture.texture.width;
                        host_asset.height = texture.texture.height;
                        host_asset.texture_format = texture_format_name(texture.texture.format);
                        host_asset.mip_count = static_cast<std::uint32_t>(texture.texture.mips.size());
                        if (host_asset.streaming_mode == "virtual_tiles")
                            for (const auto& mip : texture.texture.mips)
                            {
                                if (mip.width <= render::virtual_texture_tile_size &&
                                    mip.height <= render::virtual_texture_tile_size)
                                    break;
                                host_asset.tile_count += ((mip.width + render::virtual_texture_tile_size - 1u) /
                                                          render::virtual_texture_tile_size) *
                                                         ((mip.height + render::virtual_texture_tile_size - 1u) /
                                                          render::virtual_texture_tile_size);
                            }
                    }
                    else
                        host_asset.streaming_eligibility_error = texture.message;
'''
new = '''                    const auto texture = render::inspect_texture_asset(absolute_path);
                    if (texture.succeeded())
                    {
                        host_asset.width = texture.width;
                        host_asset.height = texture.height;
                        host_asset.texture_format = texture_format_name(texture.format);
                        host_asset.mip_count = texture.mip_count;
                        if (host_asset.streaming_mode == "virtual_tiles")
                        {
                            auto mip_width = texture.width;
                            auto mip_height = texture.height;
                            for (std::uint32_t mip = 0; mip < texture.mip_count; ++mip)
                            {
                                if (mip_width <= render::virtual_texture_tile_size &&
                                    mip_height <= render::virtual_texture_tile_size)
                                    break;
                                host_asset.tile_count += ((mip_width + render::virtual_texture_tile_size - 1u) /
                                                          render::virtual_texture_tile_size) *
                                                         ((mip_height + render::virtual_texture_tile_size - 1u) /
                                                          render::virtual_texture_tile_size);
                                mip_width = std::max(1u, mip_width / 2u);
                                mip_height = std::max(1u, mip_height / 2u);
                            }
                        }
                    }
                    else
                        host_asset.streaming_eligibility_error = texture.message;
'''
if old not in s:
    raise SystemExit('project asset texture decode block missing')
s = s.replace(old, new, 1)
p.write_text(s)
