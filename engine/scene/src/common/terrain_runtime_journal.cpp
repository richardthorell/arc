#include <arc/scene/terrain_runtime_journal.h>

#include <bit>
#include <cstdint>

namespace arc::scene
{
namespace
{

class journal_hash
{
public:
    void append_byte(std::uint8_t value) noexcept
    {
        value_ ^= value;
        value_ *= 1099511628211ull;
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

    void append_double(double value) noexcept
    {
        append_u64(std::bit_cast<std::uint64_t>(value));
    }

    void append_string(const std::string& value) noexcept
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
    std::uint64_t value_{14695981039346656037ull};
};

void append_bounds(journal_hash& hash, const terrain_world_bounds& bounds) noexcept
{
    hash.append_double(bounds.min_x);
    hash.append_double(bounds.min_y);
    hash.append_double(bounds.min_z);
    hash.append_double(bounds.max_x);
    hash.append_double(bounds.max_y);
    hash.append_double(bounds.max_z);
}

} // namespace

bool validate_terrain_runtime_journal(const terrain_runtime_journal& journal) noexcept
{
    if (journal.schema_version != terrain_runtime_journal::current_schema_version || journal.base_authoring_revision == 0u)
        return false;

    for (std::size_t index = 0; index < journal.operations.size(); ++index)
    {
        const auto& operation = journal.operations[index];
        if (!operation.id.valid() || operation.schema_version == 0u || !operation.bounds.valid() ||
            operation.canonical_payload.empty())
            return false;
        for (std::size_t other = index + 1u; other < journal.operations.size(); ++other)
            if (operation.id == journal.operations[other].id) return false;
    }
    return true;
}

std::uint64_t terrain_runtime_journal_fingerprint(const terrain_runtime_journal& journal) noexcept
{
    if (!validate_terrain_runtime_journal(journal)) return 0u;

    journal_hash hash;
    hash.append_u32(journal.schema_version);
    hash.append_u64(journal.base_authoring_revision);
    hash.append_u64(journal.operations.size());
    for (const auto& operation : journal.operations)
    {
        hash.append_u64(operation.id.high);
        hash.append_u64(operation.id.low);
        hash.append_u32(static_cast<std::uint32_t>(operation.kind));
        append_bounds(hash, operation.bounds);
        hash.append_u64(operation.seed);
        hash.append_u32(operation.schema_version);
        hash.append_string(operation.canonical_payload);
    }
    return hash.value();
}

} // namespace arc::scene
