#pragma once

#include <arc/scene/terrain_asset.h>

#include <cstdint>
#include <string>
#include <vector>

namespace arc::scene
{

enum class terrain_runtime_operation_kind : std::uint8_t
{
    deform,
    boolean_add,
    boolean_subtract,
    paint_attribute,
    fracture_bond,
    collapse_chunk
};

/** @brief Versioned deterministic operation used by future terrain save-game and replication systems. */
struct terrain_runtime_operation
{
    terrain_stable_id id{};
    terrain_runtime_operation_kind kind{terrain_runtime_operation_kind::deform};
    terrain_world_bounds bounds{};
    std::uint64_t seed{};
    std::uint32_t schema_version{1};
    std::string canonical_payload{"{}"};
};

/** @brief Ordered runtime changes layered over one authored TerrainAsset revision. */
struct terrain_runtime_journal
{
    static constexpr std::uint32_t current_schema_version = 1;

    std::uint32_t schema_version{current_schema_version};
    std::uint64_t base_authoring_revision{};
    std::vector<terrain_runtime_operation> operations;
};

/** @brief Validate stable IDs, bounds, schema versions, payloads, and operation ordering invariants. */
[[nodiscard]] bool validate_terrain_runtime_journal(const terrain_runtime_journal& journal) noexcept;

/** @brief Stable fingerprint suitable for save/network divergence checks; this is not a geometry artifact key. */
[[nodiscard]] std::uint64_t terrain_runtime_journal_fingerprint(const terrain_runtime_journal& journal) noexcept;

} // namespace arc::scene
