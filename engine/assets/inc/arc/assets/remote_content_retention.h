#pragma once

/** @file
 * @brief C9.2g1 persisted retention of active and last-good remote objects.
 *
 * Release acceptance and activation remain the caller's responsibility. This
 * only protects immutable CAS digests from quota eviction across restarts;
 * it never treats persisted retention records as trusted content metadata.
 */

#include <arc/assets/remote_content_cache.h>

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace arc::assets
{

struct [[nodiscard]] remote_content_retention_result
{
    std::string scope;
    std::size_t retained_objects{};
    std::uint64_t declared_bytes{};
    std::string error;

    [[nodiscard]] bool succeeded() const noexcept
    {
        return error.empty();
    }
};

/** Stable cache retention scope for a project + engine/runtime base build +
 * target/profile. Excludes the generation and cookBuildId so newer snapshots
 * can atomically replace earlier active + last-good retention sets.
 */
[[nodiscard]] std::string remote_content_retention_scope(const remote_content_release& release);

/** Atomically persist the union of active and last-good immutable object
 * digests (manifest and package chunks) into the existing DDC.
 *
 * Pass only C9.1-validated and successfully mounted/approved releases.
 * In lazy mode, this retains declared hashes but does not claim that all
 * objects have been fetched. On restart, retained hashes are restored before
 * pruning; callers must independently validate release identity and bytes
 * before activation. A newer release with an incompatible target is rejected.
 */
[[nodiscard]] remote_content_retention_result
retain_remote_content_generations(derived_data_cache& cache, const remote_content_release& active,
                                  const std::optional<remote_content_release>& last_good = std::nullopt);

/** Explicitly release retention when a project/target channel is retired.
 * Never call as part of transient disconnect/offline startup.
 */
[[nodiscard]] bool clear_remote_content_retention(derived_data_cache& cache, const remote_content_release& context,
                                                  cache_error& error);

} // namespace arc::assets
