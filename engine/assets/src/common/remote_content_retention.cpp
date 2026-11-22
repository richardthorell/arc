#include <arc/assets/remote_content_retention.h>

#include <algorithm>
#include <cctype>
#include <limits>
#include <span>
#include <string>
#include <unordered_map>
#include <utility>

namespace arc::assets
{
namespace
{
constexpr std::size_t maximum_retained_objects = 100000;
constexpr std::uint64_t maximum_retained_bytes = 16ull * 1024ull * 1024ull * 1024ull;

void append_identity(std::string& identity, std::string_view value)
{
    identity += std::to_string(value.size());
    identity += ':';
    identity += value;
}

bool same_target(const remote_content_target& first, const remote_content_target& second)
{
    return first.platform == second.platform && first.architecture == second.architecture &&
           first.renderer == second.renderer && first.cook_profile == second.cook_profile &&
           first.texture_family == second.texture_family;
}

bool validated_object(const remote_content_object& object)
{
    if (object.size == 0 || object.size > maximum_retained_bytes || object.path != "objects/sha256/" + object.sha256)
        return false;
    const auto digest = parse_asset_hash(object.sha256);
    return digest && to_string(*digest) == object.sha256;
}

bool validated_release_identity(const remote_content_release& release)
{
    if (release.project_guid.empty() || release.engine_version.empty() || release.base_build_id.empty() ||
        release.target.platform.empty() || release.target.architecture.empty() || release.target.renderer.empty() ||
        release.target.cook_profile.empty() || release.target.texture_family.empty() || release.generation == 0 ||
        release.chunks.size() >= maximum_retained_objects)
        return false;
    if (!validated_object(release.cook_manifest)) return false;
    return std::all_of(release.chunks.begin(), release.chunks.end(),
                       [](const auto& chunk) { return validated_object(chunk.object); });
}

} // namespace

std::string remote_content_retention_scope(const remote_content_release& release)
{
    std::string identity;
    append_identity(identity, release.project_guid);
    append_identity(identity, release.engine_version);
    append_identity(identity, release.base_build_id);
    append_identity(identity, release.target.platform);
    append_identity(identity, release.target.architecture);
    append_identity(identity, release.target.renderer);
    append_identity(identity, release.target.cook_profile);
    append_identity(identity, release.target.texture_family);
    return "c9." + to_string(hash_bytes(std::as_bytes(std::span(identity.data(), identity.size()))));
}

remote_content_retention_result
retain_remote_content_generations(derived_data_cache& cache, const remote_content_release& active,
                                  const std::optional<remote_content_release>& last_good)
{
    remote_content_retention_result result{.scope = remote_content_retention_scope(active)};
    if (!validated_release_identity(active))
    {
        result.error = "Invalid active remote generation identity or immutable object reference";
        return result;
    }
    if (last_good &&
        (!validated_release_identity(*last_good) || last_good->project_guid != active.project_guid ||
         last_good->engine_version != active.engine_version || last_good->base_build_id != active.base_build_id ||
         !same_target(last_good->target, active.target) || last_good->generation >= active.generation))
    {
        result.error = "Last-good remote generation is incompatible, invalid, or not older than active";
        return result;
    }

    // The active and previous generation may reference many shared chunks.
    // Retain each digest exactly once and reject conflicting lengths.
    std::unordered_map<std::string, std::uint64_t> objects;
    const auto append = [&](const remote_content_object& object)
    {
        const auto [it, inserted] = objects.try_emplace(object.sha256, object.size);
        if (!inserted && it->second != object.size) return false;
        return objects.size() <= maximum_retained_objects;
    };
    const auto include = [&](const remote_content_release& release)
    {
        if (!append(release.cook_manifest)) return false;
        for (const auto& chunk : release.chunks)
            if (!append(chunk.object)) return false;
        return true;
    };
    if (!include(active) || (last_good && !include(*last_good)))
    {
        result.error = "Active/last-good object closure contains inconsistent or too many references";
        return result;
    }
    std::vector<content_hash> hashes;
    hashes.reserve(objects.size());
    for (const auto& [name, size] : objects)
    {
        if (size > maximum_retained_bytes - result.declared_bytes)
        {
            result.error = "Remote generation retention exceeds declared byte limit";
            return result;
        }
        result.declared_bytes += size;
        hashes.push_back(*parse_asset_hash(name));
    }
    cache_error persistence_error;
    if (!cache.set_persistent_pins(result.scope, hashes, persistence_error))
    {
        result.error =
            persistence_error.message.empty() ? "Could not persist remote cache retention" : persistence_error.message;
        return result;
    }
    result.retained_objects = hashes.size();
    return result;
}

bool clear_remote_content_retention(derived_data_cache& cache, const remote_content_release& context,
                                    cache_error& error)
{
    if (!validated_release_identity(context))
    {
        error.message = "Cannot clear cache retention for invalid remote release identity";
        return false;
    }
    return cache.set_persistent_pins(remote_content_retention_scope(context), {}, error);
}

} // namespace arc::assets
