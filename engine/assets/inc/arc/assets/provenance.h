#pragma once

#include <string>
#include <string_view>

namespace arc::assets
{

/**
 * @brief Source provenance recorded alongside an imported asset.
 *
 * Provenance describes where source content came from and how to reproduce the
 * import. It is deliberately separate from asset_guid: refreshing or changing
 * source metadata must never change the stable ARC asset identity.
 */
struct asset_provenance
{
    std::string provider;
    std::string provider_asset_id;
    std::string source_revision;
    std::string source_hash;
    std::string original_url;
    std::string license;
    std::string variant;
    std::string import_recipe{"{}"};

    [[nodiscard]] bool empty() const noexcept
    {
        return provider.empty() && provider_asset_id.empty() && source_revision.empty() && source_hash.empty() &&
               original_url.empty() && license.empty() && variant.empty() && import_recipe == "{}";
    }

    friend bool operator==(const asset_provenance&, const asset_provenance&) = default;
};

/** @brief Return whether provenance has the minimum stable remote-source identity. */
[[nodiscard]] inline bool has_remote_source_identity(const asset_provenance& provenance) noexcept
{
    return !provenance.provider.empty() && !provenance.provider_asset_id.empty();
}

/**
 * @brief Validate cross-field provenance invariants before metadata is persisted.
 *
 * Empty provenance is valid for locally authored/imported assets. Remote source
 * identity is atomic: provider and provider asset ID must either both be set or
 * both be absent. The import recipe is stored as canonical JSON by the metadata
 * persistence layer; this contract only requires a non-empty representation.
 */
[[nodiscard]] inline bool valid_asset_provenance(const asset_provenance& provenance) noexcept
{
    const bool has_provider = !provenance.provider.empty();
    const bool has_provider_asset_id = !provenance.provider_asset_id.empty();
    if (has_provider != has_provider_asset_id) return false;
    return !provenance.import_recipe.empty();
}

} // namespace arc::assets
