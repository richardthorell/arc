#pragma once

#include <algorithm>
#include <string>
#include <string_view>
#include <vector>

namespace arc::flow
{

/** @brief Kind of asset referenced by authored Flow data. */
enum class dependency_kind
{
    asset,
    subgraph,
};

/**
 * @brief Stable authored Flow dependency.
 *
 * Dependencies are identified by ARC asset id rather than a physical path so renames and mount changes do not
 * invalidate graph references. source_node_id is presentation/debug metadata and is not part of dependency identity.
 */
struct dependency_reference
{
    std::string asset_id;
    dependency_kind kind{dependency_kind::asset};
    std::string source_node_id;
};

/** @brief Result of validating authored Flow references before compilation/cooking. */
struct dependency_validation
{
    std::vector<dependency_reference> dependencies;
    std::vector<dependency_reference> unresolved;

    [[nodiscard]] bool valid() const noexcept
    {
        return unresolved.empty();
    }
};

/**
 * @brief Normalize and validate Flow dependencies against an ARC asset-id resolver.
 *
 * The resolver is intentionally supplied by the asset layer so Flow does not depend on paths, mounts, or a concrete
 * asset database. Duplicate references collapse by (kind, asset_id), while unresolved references remain available to
 * dependency inspection and diagnostics instead of being silently dropped.
 */
template <typename Resolver>
[[nodiscard]] dependency_validation validate_dependencies(std::vector<dependency_reference> references,
                                                          Resolver&& resolves_asset_id)
{
    const auto has_empty_asset_id = [](const dependency_reference& reference) { return reference.asset_id.empty(); };
    references.erase(std::remove_if(references.begin(), references.end(), has_empty_asset_id), references.end());

    const auto compare_references = [](const dependency_reference& lhs, const dependency_reference& rhs)
    {
        if (lhs.kind != rhs.kind)
        {
            return lhs.kind < rhs.kind;
        }
        if (lhs.asset_id != rhs.asset_id)
        {
            return lhs.asset_id < rhs.asset_id;
        }
        return lhs.source_node_id < rhs.source_node_id;
    };
    std::stable_sort(references.begin(), references.end(), compare_references);

    dependency_validation result;
    for (const auto& reference : references)
    {
        if (!result.dependencies.empty())
        {
            const auto& previous = result.dependencies.back();
            if (previous.kind == reference.kind && previous.asset_id == reference.asset_id)
            {
                continue;
            }
        }

        result.dependencies.push_back(reference);
        if (!resolves_asset_id(std::string_view{reference.asset_id}))
        {
            result.unresolved.push_back(reference);
        }
    }
    return result;
}

} // namespace arc::flow
