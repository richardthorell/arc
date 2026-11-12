#include <arc/render/cooked_texture.h>
#include <arc/render/texture_artifact.h>

#include <limits>
#include <utility>

namespace arc::render
{
core::result<texture_data, std::string> decode_cooked_texture_data(std::span<const std::byte> bytes)
{
    using result = core::result<texture_data, std::string>;
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
