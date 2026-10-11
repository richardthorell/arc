#pragma once

/** @file
 * @brief C3.1 logical Content Bundle descriptors and deterministic membership.
 *
 * A bundle groups GUIDs; it never owns, renames, or duplicates an asset.
 * Distribution, authoring serialization, mounting and residency are later steps.
 */

#include <arc/assets/asset_database.h>

#include <algorithm>
#include <cstdint>
#include <map>
#include <set>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace arc::assets
{

using content_bundle_id = std::string;

namespace builtin_bundle_ids
{
inline constexpr std::string_view engine = "Engine";
inline constexpr std::string_view boot = "Boot";
inline constexpr std::string_view startup = "Startup/Common";
} // namespace builtin_bundle_ids

enum class bundle_preload_policy : std::uint8_t
{
    none,
    metadata,
    eager
};

enum class bundle_closure_policy : std::uint8_t
{
    roots_only,
    transitive_dependencies
};

struct content_bundle_descriptor
{
    static constexpr std::uint32_t current_version = 1;
    std::uint32_t version{current_version};
    content_bundle_id id;
    std::string display_name;
    std::vector<asset_guid> roots;
    std::vector<content_bundle_id> bundle_dependencies;
    bundle_preload_policy preload{bundle_preload_policy::none};
    bundle_closure_policy closure{bundle_closure_policy::transitive_dependencies};
    std::vector<std::string> tags;
};

enum class bundle_error : std::uint8_t
{
    none,
    invalid_descriptor,
    duplicate_bundle,
    missing_bundle,
    cyclic_bundle_dependency,
    missing_asset
};

struct bundle_closure_result
{
    bundle_error error{bundle_error::none};
    /** Sorted, unique asset GUIDs, including nested bundle membership. */
    std::vector<asset_guid> assets;
    /** Sorted, unique bundle IDs, including this bundle. */
    std::vector<content_bundle_id> bundles;

    [[nodiscard]] bool succeeded() const noexcept { return error == bundle_error::none; }
};

/** Descriptor registry; no asset state mutation, cooking, or I/O. */
class content_bundle_catalog
{
public:
    [[nodiscard]] bundle_error add(content_bundle_descriptor descriptor)
    {
        if (descriptor.version != content_bundle_descriptor::current_version || descriptor.id.empty())
            return bundle_error::invalid_descriptor;
        if (bundles_.contains(descriptor.id)) return bundle_error::duplicate_bundle;
        bundles_.emplace(descriptor.id, std::move(descriptor));
        return bundle_error::none;
    }

    [[nodiscard]] const content_bundle_descriptor* find(std::string_view id) const
    {
        const auto it = bundles_.find(std::string(id));
        return it == bundles_.end() ? nullptr : &it->second;
    }

    [[nodiscard]] bundle_closure_result closure(std::string_view id, const asset_database& database) const
    {
        bundle_closure_result result;
        std::set<content_bundle_id> visiting;
        std::set<content_bundle_id> visited;
        std::set<asset_guid> assets;

        const auto visit_asset = [&](auto&& self, asset_guid guid) -> bool
        {
            if (!guid.valid() || !database.query(guid)) return false;
            if (!assets.insert(guid).second) return true;
            const auto dependencies = database.dependencies(guid);
            for (const auto dependency : dependencies)
                if (!self(self, dependency)) return false;
            return true;
        };

        const auto visit_bundle = [&](auto&& self, const content_bundle_id& name) -> bool
        {
            if (visited.contains(name)) return true;
            if (!visiting.insert(name).second)
            {
                result.error = bundle_error::cyclic_bundle_dependency;
                return false;
            }
            const auto* bundle = find(name);
            if (!bundle)
            {
                result.error = bundle_error::missing_bundle;
                return false;
            }
            for (const auto& dependency : bundle->bundle_dependencies)
                if (!self(self, dependency)) return false;
            for (const auto guid : bundle->roots)
            {
                if (bundle->closure == bundle_closure_policy::transitive_dependencies)
                {
                    if (!visit_asset(visit_asset, guid))
                    {
                        result.error = bundle_error::missing_asset;
                        return false;
                    }
                }
                else if (!guid.valid() || !database.query(guid))
                {
                    result.error = bundle_error::missing_asset;
                    return false;
                }
                else
                    assets.insert(guid);
            }
            visiting.erase(name);
            visited.insert(name);
            return true;
        };

        if (!visit_bundle(visit_bundle, std::string(id))) return result;
        result.assets.assign(assets.begin(), assets.end());
        result.bundles.assign(visited.begin(), visited.end());
        return result;
    }

    /** Returns all descriptors whose transitive closure includes GUID. */
    [[nodiscard]] std::vector<content_bundle_id> memberships(asset_guid guid, const asset_database& database) const
    {
        std::vector<content_bundle_id> result;
        if (!guid.valid()) return result;
        for (const auto& [id, descriptor] : bundles_)
        {
            const auto resolved = closure(id, database);
            if (resolved.succeeded() &&
                std::binary_search(resolved.assets.begin(), resolved.assets.end(), guid))
                result.push_back(id);
        }
        return result;
    }

private:
    std::map<content_bundle_id, content_bundle_descriptor, std::less<>> bundles_;
};

} // namespace arc::assets
