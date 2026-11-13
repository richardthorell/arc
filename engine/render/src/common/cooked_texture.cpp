#include <arc/render/cooked_texture.h>
#include <arc/render/texture_artifact.h>

#include <algorithm>
#include <cstring>
#include <limits>
#include <string_view>
#include <utility>

namespace arc::render
{
namespace
{
// The project cooker currently emits ARC_TEXTURE_1 with a length-prefixed
// magic, a fixed header and mip table, followed by length-prefixed pixels.
// Preserve read compatibility until the cooker migrates to indexed .arctex V5.
constexpr std::string_view legacy_magic = "ARC_TEXTURE_1";

class legacy_reader
{
public:
    explicit legacy_reader(std::span<const std::byte> bytes) : bytes_(bytes) {}

    template <class T> bool read(T& value)
    {
        if (cursor_ > bytes_.size() || sizeof(T) > bytes_.size() - cursor_) return false;
        std::memcpy(&value, bytes_.data() + cursor_, sizeof(T));
        cursor_ += sizeof(T);
        return true;
    }

    bool skip(std::size_t length)
    {
        if (cursor_ > bytes_.size() || length > bytes_.size() - cursor_) return false;
        cursor_ += length;
        return true;
    }

    [[nodiscard]] std::size_t cursor() const noexcept
    {
        return cursor_;
    }

private:
    std::span<const std::byte> bytes_;
    std::size_t cursor_{};
};

bool is_legacy_texture(std::span<const std::byte> bytes)
{
    constexpr auto prefix = sizeof(std::uint64_t) + legacy_magic.size();
    if (bytes.size() < prefix) return false;
    std::uint64_t length{};
    std::memcpy(&length, bytes.data(), sizeof(length));
    return length == legacy_magic.size() &&
           std::memcmp(bytes.data() + sizeof(length), legacy_magic.data(), legacy_magic.size()) == 0;
}

core::result<texture_data, std::string> decode_legacy_texture(std::span<const std::byte> bytes)
{
    using result = core::result<texture_data, std::string>;
    legacy_reader reader(bytes);
    std::uint64_t magic_size{};
    if (!reader.read(magic_size) || magic_size != legacy_magic.size() || !reader.skip(legacy_magic.size()))
        return result::failure("Invalid ARC_TEXTURE_1 header");

    texture_data texture;
    std::uint32_t mip_count{};
    const bool valid_header = reader.read(texture.width) && reader.read(texture.height) && reader.read(texture.depth) &&
                              reader.read(texture.array_layers) && reader.read(texture.mip_levels) &&
                              reader.read(texture.dimension) && reader.read(texture.format) &&
                              reader.read(texture.color_space) && reader.read(texture.semantic) &&
                              reader.read(mip_count);
    if (!valid_header)
        return result::failure("Truncated ARC_TEXTURE_1 metadata");

    if (texture.width == 0 || texture.height == 0 || texture.depth == 0 || texture.array_layers == 0 ||
        texture.mip_levels == 0 || mip_count == 0 || mip_count > 32 || mip_count != texture.mip_levels ||
        static_cast<unsigned>(texture.dimension) > static_cast<unsigned>(texture_dimension::cube) ||
        static_cast<unsigned>(texture.color_space) > static_cast<unsigned>(texture_color_space::srgb) ||
        static_cast<unsigned>(texture.semantic) > static_cast<unsigned>(texture_semantic::environment) ||
        !valid_texture_format(texture.format))
        return result::failure("Invalid ARC_TEXTURE_1 dimensions or format");

    if ((texture.dimension == texture_dimension::texture_2d && texture.depth != 1) ||
        (texture.dimension == texture_dimension::texture_3d && texture.array_layers != 1) ||
        (texture.dimension == texture_dimension::cube && (texture.width != texture.height || texture.depth != 1)))
        return result::failure("Unsupported ARC_TEXTURE_1 texture topology");

    const auto layout = texture_format_metadata(texture.format);
    const std::uint64_t face_count = texture.dimension == texture_dimension::cube ? 6u : 1u;
    std::uint32_t expected_width = texture.width;
    std::uint32_t expected_height = texture.height;
    std::uint32_t expected_depth = texture.depth;
    std::uint64_t expected_offset{};
    for (std::uint32_t i = 0; i < mip_count; ++i)
    {
        std::uint32_t width{}, height{};
        std::uint64_t offset{}, size{};
        if (!reader.read(width) || !reader.read(height) || !reader.read(offset) || !reader.read(size))
            return result::failure("Truncated ARC_TEXTURE_1 mip table");
        if (width != expected_width || height != expected_height || offset != expected_offset)
            return result::failure("Invalid ARC_TEXTURE_1 mip ordering or dimensions");

        const auto blocks_x = (static_cast<std::uint64_t>(width) + layout.block_width - 1u) / layout.block_width;
        const auto blocks_y = (static_cast<std::uint64_t>(height) + layout.block_height - 1u) / layout.block_height;
        const std::uint64_t multiplier = static_cast<std::uint64_t>(expected_depth) * texture.array_layers * face_count;
        if (blocks_x > std::numeric_limits<std::uint64_t>::max() / blocks_y ||
            blocks_x * blocks_y > std::numeric_limits<std::uint64_t>::max() / layout.bytes_per_block ||
            blocks_x * blocks_y * layout.bytes_per_block > std::numeric_limits<std::uint64_t>::max() / multiplier ||
            size != blocks_x * blocks_y * layout.bytes_per_block * multiplier ||
            expected_offset > std::numeric_limits<std::uint64_t>::max() - size ||
            offset > std::numeric_limits<std::size_t>::max() || size > std::numeric_limits<std::size_t>::max())
            return result::failure("Invalid ARC_TEXTURE_1 mip payload size");

        texture_mip_data mip;
        mip.width = width;
        mip.height = height;
        mip.offset = static_cast<std::size_t>(offset);
        mip.size = static_cast<std::size_t>(size);
        texture.mips.push_back(mip);
        expected_offset += size;
        expected_width = std::max(1u, expected_width / 2u);
        expected_height = std::max(1u, expected_height / 2u);
        if (texture.dimension == texture_dimension::texture_3d) expected_depth = std::max(1u, expected_depth / 2u);
    }

    std::uint64_t payload_size{};
    if (!reader.read(payload_size) || payload_size != expected_offset || payload_size > bytes.size() - reader.cursor())
        return result::failure("Invalid ARC_TEXTURE_1 payload length");
    if (payload_size != bytes.size() - reader.cursor()) return result::failure("Trailing ARC_TEXTURE_1 payload data");

    texture.compressed = layout.compressed;
    auto& payload = texture.compressed ? texture.encoded : texture.pixels;
    const auto first = bytes.begin() + static_cast<std::ptrdiff_t>(reader.cursor());
    payload.assign(first, bytes.end());
    return result::success(std::move(texture));
}
} // namespace

core::result<texture_data, std::string> decode_cooked_texture_data(std::span<const std::byte> bytes)
{
    using result = core::result<texture_data, std::string>;
    if (is_legacy_texture(bytes)) return decode_legacy_texture(bytes);

    const auto inspected = inspect_texture_artifact(bytes);
    if (!inspected) return result::failure("Invalid .arctex: " + inspected.error().message);
    const auto& index = inspected.value();
    if (index.mode == texture_streaming_mode::virtual_tiles || index.mips.empty())
        return result::failure("Virtual/empty .arctex requires the streaming residency path");

    texture_data texture;
    texture.width = index.width;
    texture.height = index.height;
    texture.depth = index.depth;
    texture.dimension = index.dimension;
    texture.array_layers = index.array_layers;
    texture.mip_levels = index.mip_count;
    texture.format = index.format;
    texture.color_space = index.color_space;
    texture.semantic = index.semantic;
    texture.compressed = texture_format_metadata(index.format).compressed;

    auto& payload = texture.compressed ? texture.encoded : texture.pixels;
    for (std::uint32_t mip = 0; mip < index.mip_count; ++mip)
    {
        auto decoded = read_texture_artifact_mip(bytes, index, mip);
        if (!decoded)
            return result::failure("Corrupted .arctex mip " + std::to_string(mip) + ": " + decoded.error().message);
        if (payload.size() > std::numeric_limits<std::size_t>::max() - decoded.value().size())
            return result::failure("Cooked texture mip size overflow");
        const auto offset = payload.size();
        payload.insert(payload.end(), decoded.value().begin(), decoded.value().end());
        texture.mips.push_back({.width = index.mips[mip].width,
                                .height = index.mips[mip].height,
                                .offset = offset,
                                .size = decoded.value().size()});
    }
    return result::success(std::move(texture));
}
} // namespace arc::render
