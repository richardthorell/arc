#pragma once

#include <arc/assets/assets.h>
#include <arc/scene/terrain_asset.h>

#include <compare>
#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

namespace arc::scene
{

/** @brief Stable content-addressed identity for terrain derived data. */
struct terrain_content_key
{
    std::uint64_t high{};
    std::uint64_t low{};

    [[nodiscard]] constexpr bool valid() const noexcept
    {
        return high != 0u || low != 0u;
    }

    friend constexpr auto operator<=>(const terrain_content_key&, const terrain_content_key&) noexcept = default;
};

[[nodiscard]] std::string to_string(terrain_content_key key);

enum class terrain_artifact_kind : std::uint8_t
{
    render_geometry,
    fallback_geometry,
    attributes,
    collision,
    navigation,
    destruction,
    ray_queries
};

/** @brief Revision of an authoring-region dependency captured by a derived-data build. */
struct terrain_artifact_dependency_revision
{
    terrain_region_id region{};
    terrain_domain domains{terrain_domain::none};
    std::uint64_t revision{};

    friend constexpr auto operator<=>(const terrain_artifact_dependency_revision&,
                                      const terrain_artifact_dependency_revision&) noexcept = default;
};

/** @brief Renderer-independent inputs used to derive a deterministic artifact key. */
struct terrain_artifact_build_input
{
    terrain_region_id region{};
    std::uint64_t surface_fingerprint{};
    std::uint64_t authoring_revision{};
    std::uint64_t source_revision{};
    std::vector<terrain_artifact_dependency_revision> dependencies;
    std::string target_profile{"default"};
};

/** @brief Build an opaque content key without exposing virtual-geometry, collision, or backend implementation details.
 */
[[nodiscard]] terrain_content_key make_terrain_artifact_key(const terrain_artifact_build_input& input,
                                                            terrain_artifact_kind kind,
                                                            std::uint32_t compiler_version) noexcept;

/** @brief Opaque reference to one independently stored/cached terrain derived artifact. */
struct terrain_artifact_reference
{
    terrain_artifact_kind kind{terrain_artifact_kind::render_geometry};
    terrain_content_key key{};
    std::uint32_t compiler_version{};
    std::string storage_key;
};

/** @brief Cooked manifest for one authoring region. Runtime streaming/page hierarchy is intentionally separate. */
struct terrain_region_manifest
{
    terrain_region_id region{};
    terrain_world_bounds bounds{};
    std::uint64_t source_revision{};
    std::uint64_t compiled_revision{};
    std::vector<terrain_artifact_reference> artifacts;
};

/** @brief Top-level runtime manifest linking a TerrainAsset revision to independent derived products. */
struct terrain_cooked_manifest
{
    static constexpr std::uint32_t current_contract_version = 1;

    std::uint32_t contract_version{current_contract_version};
    assets::asset_guid terrain{};
    std::uint64_t authoring_revision{};
    std::vector<terrain_region_manifest> regions;
};

/** @brief Validate manifest-level identity/revision invariants without loading any artifact payload. */
[[nodiscard]] bool validate_terrain_cooked_manifest(const terrain_cooked_manifest& manifest) noexcept;

} // namespace arc::scene
