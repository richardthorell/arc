#include <arc/render/texture_artifact.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <unordered_set>

namespace arc::render
{
namespace
{

constexpr std::uint64_t artifact_magic = 0x3158455443524141ull; // "AARCTEX1" little endian.
constexpr std::uint32_t header_bytes = 188;
constexpr std::uint32_t mip_entry_bytes = 36;
constexpr std::uint32_t tile_entry_bytes = 44;
constexpr std::size_t table_hash_offset = 172;
constexpr std::size_t header_hash_offset = 180;

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

std::span<const std::byte> mip_payload(const texture_data& texture, const texture_mip_data& mip) noexcept
{
    const auto& storage = texture.has_encoded_mips() ? texture.encoded : texture.pixels;
    if (mip.offset > storage.size() || mip.size > storage.size() - mip.offset) return {};
    return std::span(storage).subspan(mip.offset, mip.size);
}

std::vector<std::byte> extract_tile(std::span<const std::byte> source, std::uint32_t width, std::uint32_t height,
                                    texture_format format, std::uint32_t tile_x, std::uint32_t tile_y)
{
    const auto layout = texture_format_metadata(format);
    if (layout.bytes_per_block == 0 || source.empty()) return {};

    const std::uint32_t source_units_x = std::max(1u, (width + layout.block_width - 1u) / layout.block_width);
    const std::uint32_t source_units_y = std::max(1u, (height + layout.block_height - 1u) / layout.block_height);
    const std::uint32_t interior_units = virtual_texture_tile_size / layout.block_width;
    const std::uint32_t border_units = virtual_texture_tile_border / layout.block_width;
    const std::uint32_t output_units = interior_units + border_units * 2u;
    const auto expected_source = static_cast<std::size_t>(source_units_x) * source_units_y * layout.bytes_per_block;
    if (source.size() < expected_source) return {};

    std::vector<std::byte> result(static_cast<std::size_t>(output_units) * output_units * layout.bytes_per_block);
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
                (static_cast<std::size_t>(source_y) * source_units_x + source_x) * layout.bytes_per_block;
            const auto destination_offset = (static_cast<std::size_t>(y) * output_units + x) * layout.bytes_per_block;
            std::memcpy(result.data() + destination_offset, source.data() + source_offset, layout.bytes_per_block);
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

std::uint32_t face_count_for(texture_dimension dimension) noexcept
{
    return dimension == texture_dimension::cube ? 6u : 1u;
}

bool valid_texture_topology(texture_dimension dimension, std::uint32_t width, std::uint32_t height, std::uint32_t depth,
                            std::uint32_t array_layers, std::uint32_t face_count) noexcept
{
    if (width == 0 || height == 0 || depth == 0 || array_layers == 0 || face_count != face_count_for(dimension))
        return false;
    switch (dimension)
    {
        case texture_dimension::texture_2d:
            return depth == 1;
        case texture_dimension::texture_3d:
            return array_layers == 1 && face_count == 1;
        case texture_dimension::cube:
            return width == height && depth == 1 && face_count == 6;
    }
    return false;
}

std::uint64_t payload_bytes_2d(std::uint32_t width, std::uint32_t height, texture_format format) noexcept
{
    const auto layout = texture_format_metadata(format);
    if (layout.bytes_per_block == 0 || width == 0 || height == 0) return 0;
    const auto units_x = (static_cast<std::uint64_t>(width) + layout.block_width - 1u) / layout.block_width;
    const auto units_y = (static_cast<std::uint64_t>(height) + layout.block_height - 1u) / layout.block_height;
    if (units_x > std::numeric_limits<std::uint64_t>::max() / units_y ||
        units_x * units_y > std::numeric_limits<std::uint64_t>::max() / layout.bytes_per_block)
        return 0;
    return units_x * units_y * layout.bytes_per_block;
}

std::uint64_t payload_bytes(std::uint32_t width, std::uint32_t height, std::uint32_t depth, std::uint32_t array_layers,
                            std::uint32_t face_count, texture_format format) noexcept
{
    const auto slice_bytes = payload_bytes_2d(width, height, format);
    if (slice_bytes == 0 || depth == 0 || array_layers == 0 || face_count == 0) return 0;
    std::uint64_t multiplier = depth;
    if (multiplier > std::numeric_limits<std::uint64_t>::max() / array_layers) return 0;
    multiplier *= array_layers;
    if (multiplier > std::numeric_limits<std::uint64_t>::max() / face_count) return 0;
    multiplier *= face_count;
    if (slice_bytes > std::numeric_limits<std::uint64_t>::max() / multiplier) return 0;
    return slice_bytes * multiplier;
}

std::uint32_t complete_mip_count(std::uint32_t width, std::uint32_t height, std::uint32_t depth,
                                 texture_dimension dimension) noexcept
{
    std::uint32_t count = 1;
    while (width > 1 || height > 1 || (dimension == texture_dimension::texture_3d && depth > 1))
    {
        width = std::max(1u, width / 2u);
        height = std::max(1u, height / 2u);
        if (dimension == texture_dimension::texture_3d) depth = std::max(1u, depth / 2u);
        ++count;
    }
    return count;
}

bool finite_metadata(const texture_artifact_metadata& metadata) noexcept
{
    return metadata.source_mip_count > 0 &&
           static_cast<std::uint32_t>(metadata.mip_policy) <= static_cast<std::uint32_t>(texture_mip_policy::none) &&
           std::isfinite(metadata.anisotropy) && metadata.anisotropy >= 1.0f && std::isfinite(metadata.lod_bias) &&
           std::isfinite(metadata.minimum_lod) && std::isfinite(metadata.maximum_lod) &&
           metadata.minimum_lod <= metadata.maximum_lod && std::isfinite(metadata.alpha_coverage_threshold) &&
           metadata.alpha_coverage_threshold >= 0.0f && metadata.alpha_coverage_threshold <= 1.0f;
}

bool ranges_do_not_overlap(std::span<const texture_artifact_mip_range> mips,
                           std::span<const texture_artifact_tile_range> tiles) noexcept
{
    struct interval
    {
        std::uint64_t first{};
        std::uint64_t last{};
    };
    std::vector<interval> ranges;
    ranges.reserve(mips.size() + tiles.size());
    for (const auto& range : mips)
        ranges.push_back({range.offset, range.offset + range.stored_size});
    for (const auto& range : tiles)
        ranges.push_back({range.offset, range.offset + range.stored_size});
    std::sort(ranges.begin(), ranges.end(), [](const auto& lhs, const auto& rhs) { return lhs.first < rhs.first; });
    for (std::size_t index = 1; index < ranges.size(); ++index)
        if (ranges[index].first < ranges[index - 1].last) return false;
    return true;
}

} // namespace

texture_artifact_bytes_result encode_texture_artifact(const texture_data& texture, texture_streaming_mode mode,
                                                      texture_artifact_metadata metadata)
{
    const auto face_count = face_count_for(texture.dimension);
    if (!valid_texture_topology(texture.dimension, texture.width, texture.height, texture.depth, texture.array_layers,
                                face_count) ||
        texture.mips.empty())
        return texture_artifact_bytes_result::failure(
            failure(texture_artifact_error_code::unsupported_texture, "texture artifact topology is unsupported"));
    if (mode == texture_streaming_mode::virtual_tiles &&
        (texture.dimension != texture_dimension::texture_2d || texture.depth != 1 || texture.array_layers != 1))
        return texture_artifact_bytes_result::failure(
            failure(texture_artifact_error_code::unsupported_texture,
                    "virtual texture artifacts currently require one 2D layer"));
    if (mode == texture_streaming_mode::virtual_tiles)
    {
        const auto format = texture_format_metadata(texture.format);
        if (format.bytes_per_block == 0 || virtual_texture_tile_size % format.block_width != 0 ||
            virtual_texture_tile_size % format.block_height != 0 ||
            virtual_texture_tile_border % format.block_width != 0 ||
            virtual_texture_tile_border % format.block_height != 0)
            return texture_artifact_bytes_result::failure(
                failure(texture_artifact_error_code::unsupported_texture,
                        "virtual texture tile geometry is not aligned to the texture format block size"));
    }
    if (texture_format_metadata(texture.format).bytes_per_block == 0)
        return texture_artifact_bytes_result::failure(
            failure(texture_artifact_error_code::unsupported_texture, "texture format is not pageable"));
    const auto expected_mips = complete_mip_count(texture.width, texture.height, texture.depth, texture.dimension);
    if (texture.mips.front().width != texture.width || texture.mips.front().height != texture.height ||
        ((mode == texture_streaming_mode::streamed_mips || mode == texture_streaming_mode::virtual_tiles) &&
         texture.mips.size() != expected_mips))
        return texture_artifact_bytes_result::failure(
            failure(texture_artifact_error_code::invalid_data, "texture artifact mip chain is incomplete"));

    std::uint32_t expected_width = texture.width;
    std::uint32_t expected_height = texture.height;
    std::uint32_t expected_depth = texture.depth;
    for (const auto& mip : texture.mips)
    {
        const auto expected_bytes = payload_bytes(expected_width, expected_height, expected_depth, texture.array_layers,
                                                  face_count, texture.format);
        if (mip.width != expected_width || mip.height != expected_height || expected_bytes == 0 ||
            expected_bytes > std::numeric_limits<std::size_t>::max() || mip.size != expected_bytes)
            return texture_artifact_bytes_result::failure(
                failure(texture_artifact_error_code::invalid_data, "texture artifact mip layout is invalid"));
        expected_width = std::max(1u, expected_width / 2u);
        expected_height = std::max(1u, expected_height / 2u);
        if (texture.dimension == texture_dimension::texture_3d) expected_depth = std::max(1u, expected_depth / 2u);
    }

    std::uint32_t tail_first = static_cast<std::uint32_t>(texture.mips.size() - 1u);
    expected_depth = texture.depth;
    for (std::uint32_t mip = 0; mip < texture.mips.size(); ++mip)
    {
        if (texture.mips[mip].width <= virtual_texture_tile_size &&
            texture.mips[mip].height <= virtual_texture_tile_size && expected_depth <= virtual_texture_tile_size)
        {
            tail_first = mip;
            break;
        }
        if (texture.dimension == texture_dimension::texture_3d) expected_depth = std::max(1u, expected_depth / 2u);
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

    std::uint64_t cursor =
        (table_end + texture_artifact_alignment - 1u) / texture_artifact_alignment * texture_artifact_alignment;
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
    output.value(static_cast<std::uint32_t>(texture.dimension));
    output.value(texture.depth);
    output.value(texture.array_layers);
    output.value(face_count);
    output.value(texture.width);
    output.value(texture.height);
    output.value(static_cast<std::uint32_t>(texture.mips.size()));
    output.value(tail_first);
    output.value(virtual_texture_tile_size);
    output.value(virtual_texture_tile_border);
    output.value(static_cast<std::uint32_t>(mip_payloads.size()));
    output.value(static_cast<std::uint32_t>(tile_payloads.size()));
    if (metadata.source_width == 0) metadata.source_width = texture.width;
    if (metadata.source_height == 0) metadata.source_height = texture.height;
    if (metadata.source_mip_count == 0) metadata.source_mip_count = static_cast<std::uint32_t>(texture.mips.size());
    if (metadata.resolved_max_size == 0)
        metadata.resolved_max_size = std::max({texture.width, texture.height, texture.depth});
    if (!finite_metadata(metadata))
        return texture_artifact_bytes_result::failure(
            failure(texture_artifact_error_code::invalid_data, "texture artifact import metadata is invalid"));
    output.value(metadata.source_width);
    output.value(metadata.source_height);
    output.value(metadata.source_mip_count);
    output.value(metadata.requested_max_size);
    output.value(metadata.resolved_max_size);
    output.value(static_cast<std::uint32_t>(metadata.power_of_two));
    output.value(static_cast<std::uint32_t>(metadata.compression));
    output.value(static_cast<std::uint32_t>(metadata.mip_policy));
    output.value(static_cast<std::uint32_t>(metadata.min_filter));
    output.value(static_cast<std::uint32_t>(metadata.mag_filter));
    output.value(static_cast<std::uint32_t>(metadata.mip_filter));
    output.value(static_cast<std::uint32_t>(metadata.wrap_u));
    output.value(static_cast<std::uint32_t>(metadata.wrap_v));
    output.value(metadata.anisotropy);
    output.value(metadata.lod_bias);
    output.value(metadata.minimum_lod);
    output.value(metadata.maximum_lod);
    output.value(metadata.alpha_coverage_threshold);
    std::uint32_t processing_flags{};
    if (metadata.generated_mips) processing_flags |= 1u << 0u;
    if (metadata.resized) processing_flags |= 1u << 1u;
    if (metadata.power_of_two_adjusted) processing_flags |= 1u << 2u;
    if (metadata.normal_mips_renormalized) processing_flags |= 1u << 3u;
    if (metadata.alpha_coverage_preserved) processing_flags |= 1u << 4u;
    if (metadata.source_mips_preserved) processing_flags |= 1u << 5u;
    output.value(processing_flags);
    output.value(table_end);
    output.value(cursor);
    output.value(std::uint64_t{});
    output.value(std::uint64_t{});
    expected_depth = texture.depth;
    for (std::uint32_t mip = 0; mip < texture.mips.size(); ++mip)
    {
        output.value(texture.mips[mip].width);
        output.value(texture.mips[mip].height);
        output.value(expected_depth);
        output.value(mip_payloads[mip].offset);
        output.value(static_cast<std::uint32_t>(mip_payloads[mip].bytes.size()));
        output.value(static_cast<std::uint32_t>(mip_payloads[mip].bytes.size()));
        output.value(mip_payloads[mip].hash);
        if (texture.dimension == texture_dimension::texture_3d) expected_depth = std::max(1u, expected_depth / 2u);
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
    output.patch(table_hash_offset, table_hash);
    const auto header_hash = hash_bytes(std::span(output.data()).first(header_hash_offset));
    output.patch(header_hash_offset, header_hash);
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
    std::uint32_t dimension{};
    std::uint32_t mip_entries{};
    std::uint32_t tile_entries{};
    std::uint32_t power_of_two{};
    std::uint32_t compression{};
    std::uint32_t mip_policy{};
    std::uint32_t min_filter{};
    std::uint32_t mag_filter{};
    std::uint32_t mip_filter{};
    std::uint32_t wrap_u{};
    std::uint32_t wrap_v{};
    std::uint32_t processing_flags{};
    std::uint64_t table_hash{};
    std::uint64_t header_hash{};
    texture_artifact_index result;
    if (!input.value(magic) || !input.value(schema) || !input.value(declared_header) || !input.value(mode) ||
        !input.value(format) || !input.value(color_space) || !input.value(semantic) || !input.value(dimension) ||
        !input.value(result.depth) || !input.value(result.array_layers) || !input.value(result.face_count) ||
        !input.value(result.width) || !input.value(result.height) || !input.value(result.mip_count) ||
        !input.value(result.tail_first_mip) || !input.value(result.tile_size) || !input.value(result.tile_border) ||
        !input.value(mip_entries) || !input.value(tile_entries) || !input.value(result.metadata.source_width) ||
        !input.value(result.metadata.source_height) || !input.value(result.metadata.source_mip_count) ||
        !input.value(result.metadata.requested_max_size) || !input.value(result.metadata.resolved_max_size) ||
        !input.value(power_of_two) || !input.value(compression) || !input.value(mip_policy) ||
        !input.value(min_filter) || !input.value(mag_filter) || !input.value(mip_filter) || !input.value(wrap_u) ||
        !input.value(wrap_v) || !input.value(result.metadata.anisotropy) || !input.value(result.metadata.lod_bias) ||
        !input.value(result.metadata.minimum_lod) || !input.value(result.metadata.maximum_lod) ||
        !input.value(result.metadata.alpha_coverage_threshold) || !input.value(processing_flags) ||
        !input.value(result.table_end) || !input.value(result.artifact_size) || !input.value(table_hash) ||
        !input.value(header_hash))
        return texture_artifact_index_result::failure(
            failure(texture_artifact_error_code::invalid_data, "texture artifact header is truncated"));
    if (magic != artifact_magic)
        return texture_artifact_index_result::failure(
            failure(texture_artifact_error_code::invalid_data, "texture artifact magic is invalid"));
    if (schema != texture_artifact_schema_version || declared_header != header_bytes)
        return texture_artifact_index_result::failure(
            failure(texture_artifact_error_code::unsupported_version, "texture artifact schema is unsupported"));
    if (hash_bytes(bytes.first(header_hash_offset)) != header_hash)
        return texture_artifact_index_result::failure(
            failure(texture_artifact_error_code::integrity_failure, "texture artifact header hash is invalid"));
    if (mode > static_cast<std::uint32_t>(texture_streaming_mode::virtual_tiles) ||
        format > static_cast<std::uint32_t>(texture_format::eac_rg11_snorm) ||
        color_space > static_cast<std::uint32_t>(texture_color_space::srgb) ||
        semantic > static_cast<std::uint32_t>(texture_semantic::environment) ||
        dimension > static_cast<std::uint32_t>(texture_dimension::cube) ||
        power_of_two > static_cast<std::uint32_t>(texture_power_of_two_policy::resize_up) ||
        compression > static_cast<std::uint32_t>(texture_compression_policy::uncompressed) ||
        mip_policy > static_cast<std::uint32_t>(texture_mip_policy::none) ||
        min_filter > static_cast<std::uint32_t>(texture_filter_mode::linear) ||
        mag_filter > static_cast<std::uint32_t>(texture_filter_mode::linear) ||
        mip_filter > static_cast<std::uint32_t>(texture_mip_filter_mode::linear) ||
        wrap_u > static_cast<std::uint32_t>(texture_address_mode::mirrored_repeat) ||
        wrap_v > static_cast<std::uint32_t>(texture_address_mode::mirrored_repeat))
        return texture_artifact_index_result::failure(
            failure(texture_artifact_error_code::invalid_data, "texture artifact metadata is invalid"));

    result.dimension = static_cast<texture_dimension>(dimension);
    result.metadata.mip_policy = static_cast<texture_mip_policy>(mip_policy);
    if (!valid_texture_topology(result.dimension, result.width, result.height, result.depth, result.array_layers,
                                result.face_count) ||
        result.mip_count == 0 || result.metadata.source_mip_count == 0 || mip_entries != result.mip_count ||
        result.tail_first_mip >= result.mip_count || result.table_end < header_bytes ||
        result.table_end > bytes.size() || result.artifact_size != bytes.size() || !finite_metadata(result.metadata) ||
        ((mode == static_cast<std::uint32_t>(texture_streaming_mode::streamed_mips) ||
          mode == static_cast<std::uint32_t>(texture_streaming_mode::virtual_tiles)) &&
         result.mip_count != complete_mip_count(result.width, result.height, result.depth, result.dimension)) ||
        (mode == static_cast<std::uint32_t>(texture_streaming_mode::virtual_tiles) &&
         (result.dimension != texture_dimension::texture_2d || result.depth != 1 || result.array_layers != 1 ||
          result.face_count != 1)) ||
        (mode != static_cast<std::uint32_t>(texture_streaming_mode::virtual_tiles) && tile_entries != 0) ||
        result.tile_size != virtual_texture_tile_size || result.tile_border != virtual_texture_tile_border)
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
    result.metadata.power_of_two = static_cast<texture_power_of_two_policy>(power_of_two);
    result.metadata.compression = static_cast<texture_compression_policy>(compression);
    result.metadata.min_filter = static_cast<texture_filter_mode>(min_filter);
    result.metadata.mag_filter = static_cast<texture_filter_mode>(mag_filter);
    result.metadata.mip_filter = static_cast<texture_mip_filter_mode>(mip_filter);
    result.metadata.wrap_u = static_cast<texture_address_mode>(wrap_u);
    result.metadata.wrap_v = static_cast<texture_address_mode>(wrap_v);
    result.metadata.generated_mips = (processing_flags & (1u << 0u)) != 0;
    result.metadata.resized = (processing_flags & (1u << 1u)) != 0;
    result.metadata.power_of_two_adjusted = (processing_flags & (1u << 2u)) != 0;
    result.metadata.normal_mips_renormalized = (processing_flags & (1u << 3u)) != 0;
    result.metadata.alpha_coverage_preserved = (processing_flags & (1u << 4u)) != 0;
    result.metadata.source_mips_preserved = (processing_flags & (1u << 5u)) != 0;
    result.mips.reserve(mip_entries);
    std::uint32_t expected_width = result.width;
    std::uint32_t expected_height = result.height;
    std::uint32_t expected_depth = result.depth;
    const auto payload_begin =
        (result.table_end + texture_artifact_alignment - 1u) / texture_artifact_alignment * texture_artifact_alignment;
    for (std::uint32_t mip = 0; mip < mip_entries; ++mip)
    {
        texture_artifact_mip_range range;
        const auto expected_size = payload_bytes(expected_width, expected_height, expected_depth, result.array_layers,
                                                 result.face_count, result.format);
        if (!input.value(range.width) || !input.value(range.height) || !input.value(range.depth) ||
            !input.value(range.offset) || !input.value(range.stored_size) || !input.value(range.decoded_size) ||
            !input.value(range.content_hash) || range.width != expected_width || range.height != expected_height ||
            range.depth != expected_depth || expected_size == 0 ||
            expected_size > std::numeric_limits<std::uint32_t>::max() || range.decoded_size != expected_size ||
            range.stored_size != range.decoded_size ||
            !valid_range(range.offset, range.stored_size, payload_begin, bytes.size()))
            return texture_artifact_index_result::failure(
                failure(texture_artifact_error_code::out_of_bounds, "texture artifact mip range is invalid"));
        result.mips.push_back(range);
        expected_width = std::max(1u, expected_width / 2u);
        expected_height = std::max(1u, expected_height / 2u);
        if (result.dimension == texture_dimension::texture_3d) expected_depth = std::max(1u, expected_depth / 2u);
    }
    result.tiles.reserve(tile_entries);
    std::unordered_set<std::uint64_t> tile_keys;
    for (std::uint32_t tile = 0; tile < tile_entries; ++tile)
    {
        texture_artifact_tile_range range;
        if (!input.value(range.mip) || !input.value(range.x) || !input.value(range.y) || !input.value(range.width) ||
            !input.value(range.height) || !input.value(range.offset) || !input.value(range.stored_size) ||
            !input.value(range.decoded_size) || !input.value(range.content_hash) ||
            range.mip >= result.tail_first_mip || range.width != result.tile_size + result.tile_border * 2u ||
            range.height != result.tile_size + result.tile_border * 2u ||
            range.x >= (result.mips[range.mip].width + result.tile_size - 1u) / result.tile_size ||
            range.y >= (result.mips[range.mip].height + result.tile_size - 1u) / result.tile_size ||
            range.decoded_size != payload_bytes_2d(range.width, range.height, result.format) ||
            range.stored_size != range.decoded_size ||
            !valid_range(range.offset, range.stored_size, payload_begin, bytes.size()) ||
            !tile_keys
                 .insert((static_cast<std::uint64_t>(range.mip) << 48u) | (static_cast<std::uint64_t>(range.y) << 24u) |
                         range.x)
                 .second)
            return texture_artifact_index_result::failure(
                failure(texture_artifact_error_code::out_of_bounds, "texture artifact tile range is invalid"));
        result.tiles.push_back(range);
    }
    std::uint64_t expected_tiles{};
    if (result.mode == texture_streaming_mode::virtual_tiles)
        for (std::uint32_t mip = 0; mip < result.tail_first_mip; ++mip)
            expected_tiles +=
                static_cast<std::uint64_t>((result.mips[mip].width + result.tile_size - 1u) / result.tile_size) *
                ((result.mips[mip].height + result.tile_size - 1u) / result.tile_size);
    if (result.tiles.size() != expected_tiles || !ranges_do_not_overlap(result.mips, result.tiles))
        return texture_artifact_index_result::failure(
            failure(texture_artifact_error_code::invalid_data, "texture artifact payload topology is invalid"));
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
