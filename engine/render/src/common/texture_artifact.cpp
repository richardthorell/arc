#include <arc/render/texture_artifact.h>

#include <algorithm>
#include <cstring>
#include <limits>

namespace arc::render
{
namespace
{

constexpr std::uint64_t artifact_magic = 0x3158455443524141ull; // "AARCTEX1" little endian.
constexpr std::uint32_t header_bytes = 88;
constexpr std::uint32_t mip_entry_bytes = 32;
constexpr std::uint32_t tile_entry_bytes = 44;

texture_artifact_error failure(texture_artifact_error_code code, std::string message)
{
    return {.code = code, .message = std::move(message)};
}

std::uint64_t hash_bytes(std::span<const std::byte> bytes) noexcept
{
    std::uint64_t hash = 1469598103934665603ull;
    for (const auto byte : bytes)
    {
        hash ^= std::to_integer<std::uint8_t>(byte);
        hash *= 1099511628211ull;
    }
    return hash;
}

class byte_writer
{
public:
    template <class T> void value(T value)
    {
        const auto* first = reinterpret_cast<const std::byte*>(&value);
        bytes_.insert(bytes_.end(), first, first + sizeof(T));
    }

    void bytes(std::span<const std::byte> value)
    {
        bytes_.insert(bytes_.end(), value.begin(), value.end());
    }

    void align(std::size_t alignment)
    {
        bytes_.resize((bytes_.size() + alignment - 1u) / alignment * alignment);
    }

    template <class T> void patch(std::size_t offset, T value)
    {
        std::memcpy(bytes_.data() + offset, &value, sizeof(value));
    }

    std::vector<std::byte>& data() noexcept
    {
        return bytes_;
    }

private:
    std::vector<std::byte> bytes_;
};

class byte_reader
{
public:
    explicit byte_reader(std::span<const std::byte> bytes) : bytes_(bytes) {}

    template <class T> bool value(T& result)
    {
        if (cursor_ > bytes_.size() || sizeof(T) > bytes_.size() - cursor_) return false;
        std::memcpy(&result, bytes_.data() + cursor_, sizeof(T));
        cursor_ += sizeof(T);
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

struct format_layout
{
    std::uint32_t unit_width{1};
    std::uint32_t unit_height{1};
    std::uint32_t unit_bytes{};
};

format_layout layout_for(texture_format format) noexcept
{
    switch (format)
    {
        case texture_format::bc1_rgba_unorm:
        case texture_format::bc1_rgba_srgb:
        case texture_format::bc4_r_unorm:
            return {4, 4, 8};
        case texture_format::bc2_rgba_unorm:
        case texture_format::bc2_rgba_srgb:
        case texture_format::bc3_rgba_unorm:
        case texture_format::bc3_rgba_srgb:
        case texture_format::bc5_rg_unorm:
        case texture_format::bc6h_rgb_ufloat:
        case texture_format::bc7_rgba_unorm:
        case texture_format::bc7_rgba_srgb:
            return {4, 4, 16};
        case texture_format::rgba8_unorm:
        case texture_format::rgba8_srgb:
            return {1, 1, 4};
        case texture_format::rgba16f:
            return {1, 1, 8};
        case texture_format::rgba32f:
            return {1, 1, 16};
    }
    return {};
}

std::span<const std::byte> mip_payload(const texture_data& texture, const texture_mip_data& mip) noexcept
{
    const auto& storage = texture.has_encoded_mips() ? texture.encoded : texture.pixels;
    if (mip.offset > storage.size() || mip.size > storage.size() - mip.offset) return {};
    return std::span(storage).subspan(mip.offset, mip.size);
}

std::vector<std::byte> extract_tile(std::span<const std::byte> source, std::uint32_t width, std::uint32_t height,
                                    texture_format format, std::uint32_t tile_x, std::uint32_t tile_y)
{
    const auto layout = layout_for(format);
    if (layout.unit_bytes == 0 || source.empty()) return {};

    const std::uint32_t source_units_x = std::max(1u, (width + layout.unit_width - 1u) / layout.unit_width);
    const std::uint32_t source_units_y = std::max(1u, (height + layout.unit_height - 1u) / layout.unit_height);
    const std::uint32_t interior_units = virtual_texture_tile_size / layout.unit_width;
    const std::uint32_t border_units = virtual_texture_tile_border / layout.unit_width;
    const std::uint32_t output_units = interior_units + border_units * 2u;
    const auto expected_source = static_cast<std::size_t>(source_units_x) * source_units_y * layout.unit_bytes;
    if (source.size() < expected_source) return {};

    std::vector<std::byte> result(static_cast<std::size_t>(output_units) * output_units * layout.unit_bytes);
    const std::int64_t origin_x = static_cast<std::int64_t>(tile_x * interior_units) - border_units;
    const std::int64_t origin_y = static_cast<std::int64_t>(tile_y * interior_units) - border_units;
    const auto wrap = [](std::int64_t value, std::uint32_t size)
    {
        const auto signed_size = static_cast<std::int64_t>(size);
        value %= signed_size;
        if (value < 0) value += signed_size;
        return static_cast<std::uint32_t>(value);
    };
    for (std::uint32_t y = 0; y < output_units; ++y)
        for (std::uint32_t x = 0; x < output_units; ++x)
        {
            const auto source_x = wrap(origin_x + x, source_units_x);
            const auto source_y = wrap(origin_y + y, source_units_y);
            const auto source_offset =
                (static_cast<std::size_t>(source_y) * source_units_x + source_x) * layout.unit_bytes;
            const auto destination_offset =
                (static_cast<std::size_t>(y) * output_units + x) * layout.unit_bytes;
            std::memcpy(result.data() + destination_offset, source.data() + source_offset, layout.unit_bytes);
        }
    return result;
}

struct pending_payload
{
    std::vector<std::byte> bytes;
    std::uint64_t offset{};
    std::uint64_t hash{};
};

bool valid_range(std::uint64_t offset, std::uint32_t size, std::uint64_t lower, std::size_t total) noexcept
{
    return offset >= lower && offset % texture_artifact_alignment == 0 && offset <= total && size <= total - offset;
}

} // namespace

texture_artifact_bytes_result encode_texture_artifact(const texture_data& texture, texture_streaming_mode mode)
{
    if (texture.dimension != texture_dimension::texture_2d || texture.array_layers != 1 || texture.width == 0 ||
        texture.height == 0 || texture.mips.empty())
        return texture_artifact_bytes_result::failure(failure(
            texture_artifact_error_code::unsupported_texture,
            "streamed texture artifacts require one ordinary 2D texture with a complete mip payload"));
    if (layout_for(texture.format).unit_bytes == 0)
        return texture_artifact_bytes_result::failure(
            failure(texture_artifact_error_code::unsupported_texture, "texture format is not pageable"));

    std::uint32_t tail_first = static_cast<std::uint32_t>(texture.mips.size() - 1u);
    for (std::uint32_t mip = 0; mip < texture.mips.size(); ++mip)
        if (texture.mips[mip].width <= virtual_texture_tile_size &&
            texture.mips[mip].height <= virtual_texture_tile_size)
        {
            tail_first = mip;
            break;
        }

    std::vector<pending_payload> mip_payloads;
    mip_payloads.reserve(texture.mips.size());
    for (const auto& mip : texture.mips)
    {
        const auto bytes = mip_payload(texture, mip);
        if (bytes.empty() || bytes.size() > std::numeric_limits<std::uint32_t>::max())
            return texture_artifact_bytes_result::failure(
                failure(texture_artifact_error_code::out_of_bounds, "texture mip payload is invalid"));
        pending_payload pending{.bytes = std::vector<std::byte>(bytes.begin(), bytes.end())};
        pending.hash = hash_bytes(pending.bytes);
        mip_payloads.push_back(std::move(pending));
    }

    struct tile_description
    {
        texture_artifact_tile_range range;
        pending_payload payload;
    };
    std::vector<tile_description> tile_payloads;
    if (mode == texture_streaming_mode::virtual_tiles)
        for (std::uint32_t mip_index = 0; mip_index < tail_first; ++mip_index)
        {
            const auto& mip = texture.mips[mip_index];
            const auto source = mip_payload(texture, mip);
            const std::uint32_t tiles_x = (mip.width + virtual_texture_tile_size - 1u) / virtual_texture_tile_size;
            const std::uint32_t tiles_y = (mip.height + virtual_texture_tile_size - 1u) / virtual_texture_tile_size;
            for (std::uint32_t y = 0; y < tiles_y; ++y)
                for (std::uint32_t x = 0; x < tiles_x; ++x)
                {
                    auto bytes = extract_tile(source, mip.width, mip.height, texture.format, x, y);
                    if (bytes.empty() || bytes.size() > std::numeric_limits<std::uint32_t>::max())
                        return texture_artifact_bytes_result::failure(failure(
                            texture_artifact_error_code::invalid_data, "virtual texture tile extraction failed"));
                    tile_description tile;
                    tile.range = {.mip = mip_index,
                                  .x = x,
                                  .y = y,
                                  .width = virtual_texture_tile_size + virtual_texture_tile_border * 2u,
                                  .height = virtual_texture_tile_size + virtual_texture_tile_border * 2u,
                                  .stored_size = static_cast<std::uint32_t>(bytes.size()),
                                  .decoded_size = static_cast<std::uint32_t>(bytes.size())};
                    tile.payload.bytes = std::move(bytes);
                    tile.payload.hash = hash_bytes(tile.payload.bytes);
                    tile.range.content_hash = tile.payload.hash;
                    tile_payloads.push_back(std::move(tile));
                }
        }

    const std::uint64_t table_end = header_bytes + static_cast<std::uint64_t>(mip_payloads.size()) * mip_entry_bytes +
                                    static_cast<std::uint64_t>(tile_payloads.size()) * tile_entry_bytes;
    if (table_end > std::numeric_limits<std::size_t>::max())
        return texture_artifact_bytes_result::failure(
            failure(texture_artifact_error_code::size_overflow, "texture artifact index is too large"));

    std::uint64_t cursor = (table_end + texture_artifact_alignment - 1u) / texture_artifact_alignment *
                           texture_artifact_alignment;
    for (auto& payload : mip_payloads)
    {
        payload.offset = cursor;
        cursor += payload.bytes.size();
        cursor = (cursor + texture_artifact_alignment - 1u) / texture_artifact_alignment * texture_artifact_alignment;
    }
    for (auto& tile : tile_payloads)
    {
        tile.payload.offset = cursor;
        tile.range.offset = cursor;
        cursor += tile.payload.bytes.size();
        cursor = (cursor + texture_artifact_alignment - 1u) / texture_artifact_alignment * texture_artifact_alignment;
    }
    if (cursor > std::numeric_limits<std::size_t>::max())
        return texture_artifact_bytes_result::failure(
            failure(texture_artifact_error_code::size_overflow, "texture artifact payload is too large"));

    byte_writer output;
    output.value(artifact_magic);
    output.value(texture_artifact_schema_version);
    output.value(header_bytes);
    output.value(static_cast<std::uint32_t>(mode));
    output.value(static_cast<std::uint32_t>(texture.format));
    output.value(static_cast<std::uint32_t>(texture.color_space));
    output.value(static_cast<std::uint32_t>(texture.semantic));
    output.value(texture.width);
    output.value(texture.height);
    output.value(static_cast<std::uint32_t>(texture.mips.size()));
    output.value(tail_first);
    output.value(virtual_texture_tile_size);
    output.value(virtual_texture_tile_border);
    output.value(static_cast<std::uint32_t>(mip_payloads.size()));
    output.value(static_cast<std::uint32_t>(tile_payloads.size()));
    output.value(table_end);
    output.value(cursor);
    output.value(std::uint64_t{});
    for (std::uint32_t mip = 0; mip < texture.mips.size(); ++mip)
    {
        output.value(texture.mips[mip].width);
        output.value(texture.mips[mip].height);
        output.value(mip_payloads[mip].offset);
        output.value(static_cast<std::uint32_t>(mip_payloads[mip].bytes.size()));
        output.value(static_cast<std::uint32_t>(mip_payloads[mip].bytes.size()));
        output.value(mip_payloads[mip].hash);
    }
    for (const auto& tile : tile_payloads)
    {
        output.value(tile.range.mip);
        output.value(tile.range.x);
        output.value(tile.range.y);
        output.value(tile.range.width);
        output.value(tile.range.height);
        output.value(tile.range.offset);
        output.value(tile.range.stored_size);
        output.value(tile.range.decoded_size);
        output.value(tile.range.content_hash);
    }
    const auto table_hash = hash_bytes(std::span(output.data()).subspan(header_bytes, table_end - header_bytes));
    output.patch(80, table_hash);
    output.align(texture_artifact_alignment);
    for (const auto& payload : mip_payloads)
    {
        output.bytes(payload.bytes);
        output.align(texture_artifact_alignment);
    }
    for (const auto& tile : tile_payloads)
    {
        output.bytes(tile.payload.bytes);
        output.align(texture_artifact_alignment);
    }
    output.data().resize(static_cast<std::size_t>(cursor));
    return texture_artifact_bytes_result::success(std::move(output.data()));
}

texture_artifact_index_result inspect_texture_artifact(std::span<const std::byte> bytes)
{
    byte_reader input(bytes);
    std::uint64_t magic{};
    std::uint32_t schema{};
    std::uint32_t declared_header{};
    std::uint32_t mode{};
    std::uint32_t format{};
    std::uint32_t color_space{};
    std::uint32_t semantic{};
    std::uint32_t mip_entries{};
    std::uint32_t tile_entries{};
    std::uint64_t table_hash{};
    texture_artifact_index result;
    if (!input.value(magic) || !input.value(schema) || !input.value(declared_header) || !input.value(mode) ||
        !input.value(format) || !input.value(color_space) || !input.value(semantic) || !input.value(result.width) ||
        !input.value(result.height) || !input.value(result.mip_count) || !input.value(result.tail_first_mip) ||
        !input.value(result.tile_size) || !input.value(result.tile_border) || !input.value(mip_entries) ||
        !input.value(tile_entries) || !input.value(result.table_end) || !input.value(result.artifact_size) ||
        !input.value(table_hash))
        return texture_artifact_index_result::failure(
            failure(texture_artifact_error_code::invalid_data, "texture artifact header is truncated"));
    if (magic != artifact_magic)
        return texture_artifact_index_result::failure(
            failure(texture_artifact_error_code::invalid_data, "texture artifact magic is invalid"));
    if (schema != texture_artifact_schema_version || declared_header != header_bytes)
        return texture_artifact_index_result::failure(
            failure(texture_artifact_error_code::unsupported_version, "texture artifact schema is unsupported"));
    if (mode > static_cast<std::uint32_t>(texture_streaming_mode::virtual_tiles) ||
        format > static_cast<std::uint32_t>(texture_format::bc7_rgba_srgb) ||
        color_space > static_cast<std::uint32_t>(texture_color_space::srgb) ||
        semantic > static_cast<std::uint32_t>(texture_semantic::environment) || result.width == 0 ||
        result.height == 0 || result.mip_count == 0 || mip_entries != result.mip_count ||
        result.tail_first_mip >= result.mip_count || result.table_end < header_bytes ||
        result.table_end > bytes.size() || result.artifact_size != bytes.size())
        return texture_artifact_index_result::failure(
            failure(texture_artifact_error_code::invalid_data, "texture artifact metadata is invalid"));
    const auto expected_table = static_cast<std::uint64_t>(header_bytes) +
                                static_cast<std::uint64_t>(mip_entries) * mip_entry_bytes +
                                static_cast<std::uint64_t>(tile_entries) * tile_entry_bytes;
    if (result.table_end != expected_table ||
        hash_bytes(bytes.subspan(header_bytes, result.table_end - header_bytes)) != table_hash)
        return texture_artifact_index_result::failure(
            failure(texture_artifact_error_code::integrity_failure, "texture artifact index hash is invalid"));

    result.schema_version = schema;
    result.mode = static_cast<texture_streaming_mode>(mode);
    result.format = static_cast<texture_format>(format);
    result.color_space = static_cast<texture_color_space>(color_space);
    result.semantic = static_cast<texture_semantic>(semantic);
    result.mips.reserve(mip_entries);
    for (std::uint32_t mip = 0; mip < mip_entries; ++mip)
    {
        texture_artifact_mip_range range;
        if (!input.value(range.width) || !input.value(range.height) || !input.value(range.offset) ||
            !input.value(range.stored_size) || !input.value(range.decoded_size) || !input.value(range.content_hash) ||
            range.width == 0 || range.height == 0 ||
            !valid_range(range.offset, range.stored_size, result.table_end, bytes.size()))
            return texture_artifact_index_result::failure(
                failure(texture_artifact_error_code::out_of_bounds, "texture artifact mip range is invalid"));
        result.mips.push_back(range);
    }
    result.tiles.reserve(tile_entries);
    for (std::uint32_t tile = 0; tile < tile_entries; ++tile)
    {
        texture_artifact_tile_range range;
        if (!input.value(range.mip) || !input.value(range.x) || !input.value(range.y) || !input.value(range.width) ||
            !input.value(range.height) || !input.value(range.offset) || !input.value(range.stored_size) ||
            !input.value(range.decoded_size) || !input.value(range.content_hash) || range.mip >= result.tail_first_mip ||
            range.width != result.tile_size + result.tile_border * 2u ||
            range.height != result.tile_size + result.tile_border * 2u ||
            !valid_range(range.offset, range.stored_size, result.table_end, bytes.size()))
            return texture_artifact_index_result::failure(
                failure(texture_artifact_error_code::out_of_bounds, "texture artifact tile range is invalid"));
        result.tiles.push_back(range);
    }
    return texture_artifact_index_result::success(std::move(result));
}

texture_artifact_bytes_result read_texture_artifact_mip(std::span<const std::byte> bytes,
                                                         const texture_artifact_index& index, std::uint32_t mip)
{
    if (mip >= index.mips.size())
        return texture_artifact_bytes_result::failure(
            failure(texture_artifact_error_code::out_of_bounds, "texture artifact mip index is out of range"));
    const auto& range = index.mips[mip];
    if (!valid_range(range.offset, range.stored_size, index.table_end, bytes.size()))
        return texture_artifact_bytes_result::failure(
            failure(texture_artifact_error_code::out_of_bounds, "texture artifact mip bytes are unavailable"));
    const auto payload = bytes.subspan(static_cast<std::size_t>(range.offset), range.stored_size);
    if (hash_bytes(payload) != range.content_hash)
        return texture_artifact_bytes_result::failure(
            failure(texture_artifact_error_code::integrity_failure, "texture artifact mip hash is invalid"));
    return texture_artifact_bytes_result::success(std::vector<std::byte>(payload.begin(), payload.end()));
}

texture_artifact_bytes_result read_texture_artifact_tile(std::span<const std::byte> bytes,
                                                          const texture_artifact_index& index, std::uint32_t tile)
{
    if (tile >= index.tiles.size())
        return texture_artifact_bytes_result::failure(
            failure(texture_artifact_error_code::out_of_bounds, "texture artifact tile index is out of range"));
    const auto& range = index.tiles[tile];
    if (!valid_range(range.offset, range.stored_size, index.table_end, bytes.size()))
        return texture_artifact_bytes_result::failure(
            failure(texture_artifact_error_code::out_of_bounds, "texture artifact tile bytes are unavailable"));
    const auto payload = bytes.subspan(static_cast<std::size_t>(range.offset), range.stored_size);
    if (hash_bytes(payload) != range.content_hash)
        return texture_artifact_bytes_result::failure(
            failure(texture_artifact_error_code::integrity_failure, "texture artifact tile hash is invalid"));
    return texture_artifact_bytes_result::success(std::vector<std::byte>(payload.begin(), payload.end()));
}

} // namespace arc::render
