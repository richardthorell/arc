#pragma once

#include <arc/assets/assets.h>
#include <arc/core/result.h>
#include <arc/scene/terrain_asset.h>

#include <compare>
#include <cstdint>
#include <span>
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

/** @brief Independently readable byte range inside a terrain derived artifact. */
struct terrain_artifact_page_reference
{
    std::uint32_t index{};
    std::uint64_t offset{};
    std::uint32_t stored_size{};
    std::uint32_t decoded_size{};
    std::uint64_t content_hash{};
    bool root{};

    friend constexpr auto operator<=>(const terrain_artifact_page_reference&,
                                      const terrain_artifact_page_reference&) noexcept = default;
};

/** @brief Opaque reference to one independently stored/cached terrain derived artifact. */
struct terrain_artifact_reference
{
    terrain_artifact_kind kind{terrain_artifact_kind::render_geometry};
    terrain_content_key key{};
    std::uint32_t compiler_version{};
    std::string storage_key;
    std::uint32_t generation{};
    std::uint64_t payload_size{};
    std::uint64_t metadata_offset{};
    std::uint64_t metadata_size{};
    std::vector<terrain_artifact_page_reference> pages;
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
    static constexpr std::uint32_t current_contract_version = 2;

    std::uint32_t contract_version{current_contract_version};
    assets::asset_guid terrain{};
    std::uint64_t authoring_revision{};
    std::vector<terrain_region_manifest> regions;
};

struct terrain_cooked_manifest_error
{
    std::string message;
};

using terrain_cooked_manifest_bytes_result =
    core::result<std::vector<std::byte>, terrain_cooked_manifest_error>;
using terrain_cooked_manifest_result = core::result<terrain_cooked_manifest, terrain_cooked_manifest_error>;

/** @brief Validate manifest-level identity/revision/range invariants without loading any artifact payload. */
[[nodiscard]] bool validate_terrain_cooked_manifest(const terrain_cooked_manifest& manifest) noexcept;

/** @brief Serialize only terrain region/artifact metadata; detailed derived payloads remain external. */
[[nodiscard]] terrain_cooked_manifest_bytes_result encode_terrain_cooked_manifest(const terrain_cooked_manifest& manifest);

/** @brief Decode and validate a lightweight terrain cooked manifest. */
[[nodiscard]] terrain_cooked_manifest_result decode_terrain_cooked_manifest(std::span<const std::byte> bytes);

/** @brief Find a specific derived product without touching its external payload. */
[[nodiscard]] const terrain_artifact_reference* find_terrain_artifact(const terrain_cooked_manifest& manifest,
                                                                     terrain_region_id region,
                                                                     terrain_artifact_kind kind) noexcept;

} // namespace arc::scene
