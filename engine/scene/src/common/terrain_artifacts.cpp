#include <arc/scene/terrain_artifacts.h>

#include <algorithm>
#include <array>
#include <bit>
#include <cstring>
#include <iomanip>
#include <limits>
#include <sstream>
#include <type_traits>

namespace arc::scene
{
namespace
{

constexpr std::array<std::byte, 8> manifest_magic{
    static_cast<std::byte>('A'), static_cast<std::byte>('R'), static_cast<std::byte>('C'), static_cast<std::byte>('T'),
    static_cast<std::byte>('C'), static_cast<std::byte>('M'), static_cast<std::byte>('0'), static_cast<std::byte>('2')};

class key_hash
{
public:
    key_hash(std::uint64_t seed, std::uint64_t prime) noexcept : value_(seed), prime_(prime) {}

    void append_byte(std::uint8_t value) noexcept
    {
        value_ ^= value;
        value_ *= prime_;
    }

    void append_u32(std::uint32_t value) noexcept
    {
        for (std::uint32_t shift = 0; shift < 32u; shift += 8u)
            append_byte(static_cast<std::uint8_t>((value >> shift) & 0xffu));
    }

    void append_u64(std::uint64_t value) noexcept
    {
        for (std::uint32_t shift = 0; shift < 64u; shift += 8u)
            append_byte(static_cast<std::uint8_t>((value >> shift) & 0xffu));
    }

    void append_string(std::string_view value) noexcept
    {
        append_u64(value.size());
        for (const auto character : value)
            append_byte(static_cast<std::uint8_t>(character));
    }

    [[nodiscard]] std::uint64_t value() const noexcept
    {
        return value_;
    }

private:
    std::uint64_t value_{};
    std::uint64_t prime_{};
};

std::uint64_t dependency_fingerprint(const terrain_artifact_dependency_revision& dependency,
                                     std::uint64_t seed) noexcept
{
    key_hash hash(seed, 1099511628211ull);
    hash.append_u64(static_cast<std::uint64_t>(dependency.region.x));
    hash.append_u64(static_cast<std::uint64_t>(dependency.region.z));
    hash.append_u32(static_cast<std::uint32_t>(dependency.domains));
    hash.append_u64(dependency.revision);
    return hash.value();
}

bool same_region(const terrain_region_manifest& lhs, const terrain_region_manifest& rhs) noexcept
{
    return lhs.region == rhs.region;
}

template <class T, bool = std::is_enum_v<T>> struct stored_type_for
{
    using type = T;
};

template <class T> struct stored_type_for<T, true>
{
    using type = std::underlying_type_t<T>;
};

template <class T> using stored_type_for_t = typename stored_type_for<T>::type;

class byte_writer
{
public:
    template <class T> void value(T input)
    {
        static_assert(std::is_integral_v<T> || std::is_enum_v<T> || std::is_floating_point_v<T>);
        using stored_type = stored_type_for_t<T>;
        if constexpr (std::is_floating_point_v<stored_type>)
        {
            using bits_type = std::conditional_t<sizeof(stored_type) == 4, std::uint32_t, std::uint64_t>;
            value(std::bit_cast<bits_type>(static_cast<stored_type>(input))));
        }
        else
        {
            using unsigned_type = std::make_unsigned_t<stored_type>;
            const auto bits = static_cast<unsigned_type>(static_cast<stored_type>(input));
            for (std::size_t index = 0; index < sizeof(stored_type); ++index)
                bytes_.push_back(static_cast<std::byte>((bits >> (index * 8u)) & 0xffu));
        }
    }

    void raw(std::span<const std::byte> value)
    {
        bytes_.insert(bytes_.end(), value.begin(), value.end());
    }

    void string(std::string_view value)
    {
        this->value(static_cast<std::uint32_t>(value.size()));
        raw(std::as_bytes(std::span(value.data(), value.size())));
    }

    [[nodiscard]] std::vector<std::byte> take() &&
    {
        return std::move(bytes_);
    }

private:
    std::vector<std::byte> bytes_;
};

class byte_reader
{
public:
    explicit byte_reader(std::span<const std::byte> bytes) : bytes_(bytes) {}

    template <class T> bool value(T& output)
    {
        static_assert(std::is_integral_v<T> || std::is_enum_v<T> || std::is_floating_point_v<T>);
        if constexpr (std::is_floating_point_v<T>)
        {
            using bits_type = std::conditional_t<sizeof(T) == 4, std::uint32_t, std::uint64_t>;
            bits_type bits{};
            if (!value(bits)) return false;
            output = std::bit_cast<T>(bits);
            return true;
        }
        else
        {
            using stored_type = stored_type_for_t<T>;
            using unsigned_type = std::make_unsigned_t<stored_type>;
            if (remaining() < sizeof(stored_type)) return false;
            unsigned_type bits{};
            for (std::size_t index = 0; index < sizeof(stored_type); ++index)
                bits |= static_cast<unsigned_type>(std::to_integer<std::uint8_t>(bytes_[cursor_ + index]))
                        << (index * 8u);
            cursor_ += sizeof(stored_type);
            output = static_cast<T>(static_cast<stored_type>(bits));
            return true;
        }
    }

    bool string(std::string& output)
    {
        std::uint32_t size{};
        if (!value(size) || remaining() < size) return false;
        output.assign(reinterpret_cast<const char*>(bytes_.data() + cursor_), size);
        cursor_ += size;
        return true;
    }

    [[nodiscard]] std::size_t remaining() const noexcept
    {
        return bytes_.size() - cursor_;
    }

private:
    std::span<const std::byte> bytes_;
    std::size_t cursor_{};
};

terrain_cooked_manifest_error manifest_failure(std::string message)
{
    return {.message = std::move(message)};
}

void write_bounds(byte_writer& writer, const terrain_world_bounds& bounds)
{
    writer.value(bounds.min_x);
    writer.value(bounds.min_y);
    writer.value(bounds.min_z);
    writer.value(bounds.max_x);
    writer.value(bounds.max_y);
    writer.value(bounds.max_z);
}

bool read_bounds(byte_reader& reader, terrain_world_bounds& bounds)
{
    return reader.value(bounds.min_x) && reader.value(bounds.min_y) && reader.value(bounds.min_z) &&
           reader.value(bounds.max_x) && reader.value(bounds.max_y) && reader.value(bounds.max_z);
}

} // namespace

std::string to_string(terrain_content_key key)
{
    std::ostringstream stream;
    stream << std::hex << std::setfill('0') << std::setw(16) << key.high << std::setw(16) << key.low;
    return stream.str();
}

terrain_content_key make_terrain_artifact_key(const terrain_artifact_build_input& input, terrain_artifact_kind kind,
                                              std::uint32_t compiler_version) noexcept
{
    if (input.surface_fingerprint == 0u || input.authoring_revision == 0u || compiler_version == 0u) return {};

    key_hash high(14695981039346656037ull, 1099511628211ull);
    key_hash low(7809847782465536322ull, 14029467366897019727ull);
    const auto append_common = [&](key_hash& hash)
    {
        hash.append_u32(static_cast<std::uint32_t>(kind));
        hash.append_u32(compiler_version);
        hash.append_u64(static_cast<std::uint64_t>(input.region.x));
        hash.append_u64(static_cast<std::uint64_t>(input.region.z));
        hash.append_u64(input.surface_fingerprint);
        hash.append_u64(input.authoring_revision);
        hash.append_u64(input.source_revision);
        hash.append_string(input.target_profile);
        hash.append_u64(input.dependencies.size());
    };
    append_common(high);
    append_common(low);

    std::uint64_t high_xor{};
    std::uint64_t high_sum{};
    std::uint64_t low_xor{};
    std::uint64_t low_sum{};
    for (const auto& dependency : input.dependencies)
    {
        const auto high_part = dependency_fingerprint(dependency, 0xcbf29ce484222325ull);
        const auto low_part = dependency_fingerprint(dependency, 0x6c62272e07bb0142ull);
        high_xor ^= high_part;
        high_sum += high_part;
        low_xor ^= low_part;
        low_sum += low_part;
    }
    high.append_u64(high_xor);
    high.append_u64(high_sum);
    low.append_u64(low_xor);
    low.append_u64(low_sum);

    terrain_content_key result{high.value(), low.value()};
    if (!result.valid()) result.low = 1u;
    return result;
}

bool validate_terrain_cooked_manifest(const terrain_cooked_manifest& manifest) noexcept
{
    if (manifest.contract_version != terrain_cooked_manifest::current_contract_version || !manifest.terrain.valid() ||
        manifest.authoring_revision == 0u)
        return false;

    for (std::size_t index = 0; index < manifest.regions.size(); ++index)
    {
        const auto& region = manifest.regions[index];
        if (!region.bounds.valid() || region.source_revision == 0u || region.compiled_revision == 0u ||
            region.compiled_revision > manifest.authoring_revision)
            return false;
        for (std::size_t other = index + 1u; other < manifest.regions.size(); ++other)
            if (same_region(region, manifest.regions[other])) return false;

        for (std::size_t artifact_index = 0; artifact_index < region.artifacts.size(); ++artifact_index)
        {
            const auto& artifact = region.artifacts[artifact_index];
            if (!artifact.key.valid() || artifact.compiler_version == 0u || artifact.storage_key.empty() ||
                artifact.generation == 0u || artifact.payload_size == 0u)
                return false;
            if ((artifact.metadata_offset > artifact.payload_size) ||
                (artifact.metadata_size > artifact.payload_size - artifact.metadata_offset))
                return false;
            for (std::size_t page_index = 0; page_index < artifact.pages.size(); ++page_index)
            {
                const auto& page = artifact.pages[page_index];
                if (page.index != page_index || page.stored_size == 0u || page.decoded_size == 0u ||
                    page.offset > artifact.payload_size || page.stored_size > artifact.payload_size - page.offset)
                    return false;
            }
            for (std::size_t other = artifact_index + 1u; other < region.artifacts.size(); ++other)
                if (artifact.kind == region.artifacts[other].kind) return false;
        }
    }
    return true;
}

terrain_cooked_manifest_bytes_result encode_terrain_cooked_manifest(const terrain_cooked_manifest& manifest)
{
    if (!validate_terrain_cooked_manifest(manifest))
        return terrain_cooked_manifest_bytes_result::failure(manifest_failure("terrain cooked manifest is invalid"));
    if (manifest.regions.size() > std::numeric_limits<std::uint32_t>::max())
        return terrain_cooked_manifest_bytes_result::failure(manifest_failure("terrain region count is too large"));

    byte_writer writer;
    writer.raw(manifest_magic);
    writer.value(manifest.contract_version);
    writer.value(manifest.terrain.high);
    writer.value(manifest.terrain.low);
    writer.value(manifest.authoring_revision);
    writer.value(static_cast<std::uint32_t>(manifest.regions.size()));
    for (const auto& region : manifest.regions)
    {
        writer.value(region.region.x);
        writer.value(region.region.z);
        write_bounds(writer, region.bounds);
        writer.value(region.source_revision);
        writer.value(region.compiled_revision);
        writer.value(static_cast<std::uint32_t>(region.artifacts.size()));
        for (const auto& artifact : region.artifacts)
        {
            writer.value(artifact.kind);
            writer.value(artifact.key.high);
            writer.value(artifact.key.low);
            writer.value(artifact.compiler_version);
            writer.string(artifact.storage_key);
            writer.value(artifact.generation);
            writer.value(artifact.payload_size);
            writer.value(artifact.metadata_offset);
            writer.value(artifact.metadata_size);
            writer.value(static_cast<std::uint32_t>(artifact.pages.size()));
            for (const auto& page : artifact.pages)
            {
                writer.value(page.index);
                writer.value(page.offset);
                writer.value(page.stored_size);
                writer.value(page.decoded_size);
                writer.value(page.content_hash);
                writer.value(static_cast<std::uint8_t>(page.root ? 1u : 0u));
            }
        }
    }
    return terrain_cooked_manifest_bytes_result::success(std::move(writer).take());
}

terrain_cooked_manifest_result decode_terrain_cooked_manifest(std::span<const std::byte> bytes)
{
    if (bytes.size() < manifest_magic.size() || !std::equal(manifest_magic.begin(), manifest_magic.end(), bytes.begin()))
        return terrain_cooked_manifest_result::failure(manifest_failure("invalid terrain cooked manifest magic"));

    byte_reader reader(bytes.subspan(manifest_magic.size()));
    terrain_cooked_manifest manifest;
    std::uint32_t region_count{};
    if (!reader.value(manifest.contract_version) || !reader.value(manifest.terrain.high) ||
        !reader.value(manifest.terrain.low) || !reader.value(manifest.authoring_revision) || !reader.value(region_count))
        return terrain_cooked_manifest_result::failure(manifest_failure("truncated terrain cooked manifest header"));
    if (manifest.contract_version != terrain_cooked_manifest::current_contract_version)
        return terrain_cooked_manifest_result::failure(manifest_failure("unsupported terrain cooked manifest version"));

    manifest.regions.reserve(region_count);
    for (std::uint32_t region_index = 0; region_index < region_count; ++region_index)
    {
        terrain_region_manifest region;
        std::uint32_t artifact_count{};
        if (!reader.value(region.region.x) || !reader.value(region.region.z) || !read_bounds(reader, region.bounds) ||
            !reader.value(region.source_revision) || !reader.value(region.compiled_revision) ||
            !reader.value(artifact_count))
            return terrain_cooked_manifest_result::failure(manifest_failure("truncated terrain region record"));
        region.artifacts.reserve(artifact_count);
        for (std::uint32_t artifact_index = 0; artifact_index < artifact_count; ++artifact_index)
        {
            terrain_artifact_reference artifact;
            std::uint32_t page_count{};
            if (!reader.value(artifact.kind) || !reader.value(artifact.key.high) || !reader.value(artifact.key.low) ||
                !reader.value(artifact.compiler_version) || !reader.string(artifact.storage_key) ||
                !reader.value(artifact.generation) || !reader.value(artifact.payload_size) ||
                !reader.value(artifact.metadata_offset) || !reader.value(artifact.metadata_size) ||
                !reader.value(page_count))
                return terrain_cooked_manifest_result::failure(manifest_failure("truncated terrain artifact record"));
            artifact.pages.reserve(page_count);
            for (std::uint32_t page_index = 0; page_index < page_count; ++page_index)
            {
                terrain_artifact_page_reference page;
                std::uint8_t root{};
                if (!reader.value(page.index) || !reader.value(page.offset) || !reader.value(page.stored_size) ||
                    !reader.value(page.decoded_size) || !reader.value(page.content_hash) || !reader.value(root))
                    return terrain_cooked_manifest_result::failure(manifest_failure("truncated terrain page record"));
                page.root = root != 0u;
                artifact.pages.push_back(page);
            }
            region.artifacts.push_back(std::move(artifact));
        }
        manifest.regions.push_back(std::move(region));
    }
    if (reader.remaining() != 0u || !validate_terrain_cooked_manifest(manifest))
        return terrain_cooked_manifest_result::failure(manifest_failure("terrain cooked manifest failed validation"));
    return terrain_cooked_manifest_result::success(std::move(manifest));
}

const terrain_artifact_reference* find_terrain_artifact(const terrain_cooked_manifest& manifest,
                                                        terrain_region_id region, terrain_artifact_kind kind) noexcept
{
    const auto found_region = std::find_if(manifest.regions.begin(), manifest.regions.end(),
                                           [region](const auto& candidate) { return candidate.region == region; });
    if (found_region == manifest.regions.end()) return nullptr;
    const auto found_artifact = std::find_if(found_region->artifacts.begin(), found_region->artifacts.end(),
                                             [kind](const auto& candidate) { return candidate.kind == kind; });
    return found_artifact == found_region->artifacts.end() ? nullptr : &*found_artifact;
}

} // namespace arc::scene
