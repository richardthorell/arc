#pragma once

#include <arc/assets/assets.h>

#include <algorithm>
#include <span>
#include <vector>

namespace arc::assets
{

struct asset_dependency_usage
{
    asset_guid owner{};
    asset_type_id owner_type{};
    std::filesystem::path owner_path;
    bool owner_read_only{};

    friend bool operator==(const asset_dependency_usage&, const asset_dependency_usage&) = default;
};

struct asset_delete_plan
{
    asset_guid guid{};
    bool found{};
    bool read_only{};
    std::vector<asset_dependency_usage> usages;

    [[nodiscard]] bool safe_to_delete() const noexcept
    {
        return found && !read_only && usages.empty();
    }
};

namespace detail
{
[[nodiscard]] inline const asset_snapshot* find_asset_snapshot(std::span<const asset_snapshot> assets,
                                                               asset_guid guid) noexcept
{
    const auto it =
        std::find_if(assets.begin(), assets.end(), [guid](const asset_snapshot& asset) { return asset.guid == guid; });
    return it == assets.end() ? nullptr : &*it;
}
} // namespace detail

/** @brief Return deterministic direct usages of an asset from a registry snapshot.
 *
 * The registry remains authoritative for dependency ownership. This helper is deliberately
 * read-only so editor Find Usages and destructive-operation previews can share the same
 * interpretation without mutating asset state.
 */
[[nodiscard]] inline std::vector<asset_dependency_usage> find_asset_usages(const asset_registry_snapshot& registry,
                                                                           asset_guid guid)
{
    std::vector<asset_dependency_usage> result;
    const auto* target = detail::find_asset_snapshot(registry.assets, guid);
    if (!target) return result;

    result.reserve(target->reverse_dependencies.size());
    for (const asset_guid owner_guid : target->reverse_dependencies)
    {
        const auto* owner = detail::find_asset_snapshot(registry.assets, owner_guid);
        if (!owner) continue;

        result.push_back(asset_dependency_usage{
            .owner = owner->guid,
            .owner_type = owner->type,
            .owner_path = owner->source_path,
            .owner_read_only = owner->read_only,
        });
    }

    std::sort(result.begin(), result.end(),
              [](const auto& lhs, const auto& rhs)
              {
                  if (lhs.owner_path != rhs.owner_path)
                      return lhs.owner_path.generic_string() < rhs.owner_path.generic_string();
                  return to_string(lhs.owner) < to_string(rhs.owner);
              });
    result.erase(std::unique(result.begin(), result.end(),
                             [](const auto& lhs, const auto& rhs) { return lhs.owner == rhs.owner; }),
                 result.end());
    return result;
}

/** @brief Build a non-mutating delete preview for editor/automation confirmation. */
[[nodiscard]] inline asset_delete_plan plan_asset_delete(const asset_registry_snapshot& registry, asset_guid guid)
{
    asset_delete_plan result{.guid = guid};
    const auto* target = detail::find_asset_snapshot(registry.assets, guid);
    if (!target) return result;

    result.found = true;
    result.read_only = target->read_only;
    result.usages = find_asset_usages(registry, guid);
    return result;
}

} // namespace arc::assets
