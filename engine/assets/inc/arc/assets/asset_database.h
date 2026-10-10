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

/**
 * Read-only logical catalog/query boundary shared by authoring and runtime
 * content backends. Loading, pinning, eviction and generation publication
 * remain asset_manager responsibilities.
 */
class asset_database
{
public:
    virtual ~asset_database() = default;

    [[nodiscard]] virtual std::optional<asset_database_record>
    query(asset_guid guid, asset_type_id expected_type = {}) const = 0;
    [[nodiscard]] virtual std::vector<asset_guid> dependencies(asset_guid guid) const = 0;
    [[nodiscard]] virtual std::vector<asset_guid> reverse_dependencies(asset_guid guid) const = 0;
    [[nodiscard]] virtual std::uint64_t revision() const = 0;

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

    [[nodiscard]] std::optional<asset_database_record>
    query(asset_guid guid, asset_type_id expected_type = {}) const override;
    [[nodiscard]] std::vector<asset_guid> dependencies(asset_guid guid) const override;
    [[nodiscard]] std::vector<asset_guid> reverse_dependencies(asset_guid guid) const override;
    [[nodiscard]] std::uint64_t revision() const override;

private:
    const asset_manager* manager_{};
};

} // namespace arc::assets
