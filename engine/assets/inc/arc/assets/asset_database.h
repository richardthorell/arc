#pragma once

/** @file
 * @brief Backend-neutral logical asset database contract.
 *
 * The Asset Database describes registered logical assets independently from
 * how their bytes are stored or loaded. asset_guid is the primary key;
 * paths/locators are diagnostic hints only.
 */

#include <arc/assets/assets.h>

#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace arc::assets
{

enum class asset_provider_kind : std::uint8_t
{
    project_source,
    builtin_source,
    virtual_builtin,
    cooked_package,
    runtime_overlay
};

/**
 * Stable provider identity vocabulary shared by every Asset Database backend.
 *
 * These IDs describe provider classes, not precedence. Cooked package instances
 * append a package/build-specific suffix to cooked_package_prefix. C5 owns the
 * eventual override/tombstone policy between providers.
 */
namespace asset_provider_ids
{
inline constexpr std::string_view project_source = "arc.project.source";
inline constexpr std::string_view builtin_source = "arc.builtin.source";
inline constexpr std::string_view virtual_builtin = "arc.builtin.virtual";
inline constexpr std::string_view cooked_package_prefix = "arc.package.";
inline constexpr std::string_view runtime_overlay = "arc.runtime.overlay";
} // namespace asset_provider_ids

[[nodiscard]] constexpr std::string_view asset_provider_kind_name(asset_provider_kind kind) noexcept
{
    switch (kind)
    {
        case asset_provider_kind::project_source:
            return "project-source";
        case asset_provider_kind::builtin_source:
            return "builtin-source";
        case asset_provider_kind::virtual_builtin:
            return "virtual-builtin";
        case asset_provider_kind::cooked_package:
            return "cooked-package";
        case asset_provider_kind::runtime_overlay:
            return "runtime-overlay";
    }
    return "unknown";
}

struct asset_database_provider
{
    std::string id;
    asset_provider_kind kind{asset_provider_kind::project_source};
    std::string debug_name;
    std::int32_t priority{};
    std::uint64_t generation{};
    bool active{};
    bool read_only{};

    friend bool operator==(const asset_database_provider&, const asset_database_provider&) = default;
};

struct asset_database_representation
{
    std::string name;
    std::string provider_id;
    asset_hash content_hash{};
    std::uint64_t size{};
    asset_residency residency{asset_residency::metadata_only};
    /** Human-readable storage/debug locator. Never logical asset identity. */
    std::string locator_hint;

    friend bool operator==(const asset_database_representation&, const asset_database_representation&) = default;
};

/**
 * Backend-neutral view of one logical asset.
 *
 * Source metadata is optional/editor-oriented. Runtime catalogs may omit it
 * while still exposing the same GUID/type/dependency/provider model.
 */
struct asset_database_record
{
    asset_guid guid{};
    asset_type_id type{};
    std::string title;
    std::string description;
    std::string source_hint;
    std::vector<asset_guid> dependencies;
    std::vector<asset_guid> reverse_dependencies;
    /** Bundle IDs containing this GUID in their resolved dependency closure. */
    std::vector<std::string> bundle_memberships;
    std::vector<asset_database_provider> providers;
    std::vector<asset_database_representation> representations;
    std::string active_provider;
    asset_state state{asset_state::unknown};
    asset_residency residency{asset_residency::metadata_only};
    std::uint64_t generation{};
    std::uint64_t revision{};
    std::uint32_t strong_references{};
    std::uint32_t pins{};
    bool has_last_good{};

    friend bool operator==(const asset_database_record&, const asset_database_record&) = default;
};

/** Logical database change stream; sequence is independent of record revision. */
enum class asset_database_change_kind : std::uint8_t
{
    added,
    removed,
    updated,
    provider_changed,
    catalog_reset
};

struct asset_database_change
{
    std::uint64_t sequence{};
    asset_database_change_kind kind{asset_database_change_kind::updated};
    asset_guid guid{};
    std::string provider_id;
};

struct asset_database_change_batch
{
    std::uint64_t next_sequence{};
    bool history_lost{};
    std::vector<asset_database_change> events;
};

/**
 * Read-only logical catalog/query boundary shared by authoring and runtime
 * content backends. Loading, pinning, eviction and generation publication
 * remain asset_manager responsibilities.
 */
class asset_database
{
public:
    virtual ~asset_database() = default;

    [[nodiscard]] virtual std::optional<asset_database_record> query(asset_guid guid,
                                                                     asset_type_id expected_type = {}) const = 0;
    [[nodiscard]] virtual std::vector<asset_guid> dependencies(asset_guid guid) const = 0;
    [[nodiscard]] virtual std::vector<asset_guid> reverse_dependencies(asset_guid guid) const = 0;
    /** Snapshot all logical records for cache rebuilds and indexed consumers. */
    [[nodiscard]] virtual std::vector<asset_database_record> records() const = 0;
    [[nodiscard]] virtual std::uint64_t revision() const = 0;
    /** Query events strictly after cursor; history_lost requires caller resnapshot. */
    [[nodiscard]] virtual asset_database_change_batch changes_since(std::uint64_t cursor) const = 0;

    [[nodiscard]] std::optional<asset_database_record> query(const asset_reference& reference) const
    {
        // The logical database never reconstructs identity from a path hint.
        if (!reference.guid.valid()) return std::nullopt;
        return query(reference.guid, reference.expected_type);
    }
};

/**
 * C2 authoring adapter over the existing asset_manager registry.
 *
 * This intentionally does not replace asset_manager. It exposes the manager's
 * logical records through the same contract future cooked/package backends use.
 */
class asset_manager_database final : public asset_database
{
public:
    explicit asset_manager_database(const asset_manager& manager) noexcept : manager_(&manager) {}

    [[nodiscard]] std::optional<asset_database_record> query(const asset_reference& reference) const
    {
        if (!reference.guid.valid()) return std::nullopt;
        return query(reference.guid, reference.expected_type);
    }

    [[nodiscard]] std::optional<asset_database_record> query(asset_guid guid,
                                                             asset_type_id expected_type = {}) const override;
    [[nodiscard]] std::vector<asset_guid> dependencies(asset_guid guid) const override;
    [[nodiscard]] std::vector<asset_guid> reverse_dependencies(asset_guid guid) const override;
    [[nodiscard]] std::vector<asset_database_record> records() const override;
    [[nodiscard]] std::uint64_t revision() const override;
    [[nodiscard]] asset_database_change_batch changes_since(std::uint64_t cursor) const override;

private:
    const asset_manager* manager_{};
};

} // namespace arc::assets
