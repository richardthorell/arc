#include <arc/scene/terrain_artifacts.h>

#include <algorithm>
#include <iomanip>
#include <sstream>

namespace arc::scene
{
namespace
{

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

    // Dependency contribution is deliberately order-independent. Build snapshots canonicalize their order, but the
    // content key must remain stable even when another producer presents the same dependency set in a different order.
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
            if (!artifact.key.valid() || artifact.compiler_version == 0u) return false;
            for (std::size_t other = artifact_index + 1u; other < region.artifacts.size(); ++other)
                if (artifact.kind == region.artifacts[other].kind) return false;
        }
    }
    return true;
}

} // namespace arc::scene
