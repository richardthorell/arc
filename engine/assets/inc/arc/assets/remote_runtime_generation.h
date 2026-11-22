#pragma once

/** @file
 * @brief C9.2g2 verified remote generation promotion and live overlay owner.
 *
 * Run activation and rollback only on a runtime-safe, serialized boundary,
 * after acquisition jobs complete. This class never performs network IO or
 * writes authored project files; C9.3 will own endpoint/source selection.
 */

#include <arc/assets/remote_content_generation.h>
#include <arc/assets/remote_content_retention.h>
#include <arc/assets/runtime_content.h>

#include <cstdint>
#include <filesystem>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <utility>

namespace arc::assets
{

struct remote_runtime_generation_state
{
    std::uint64_t active_generation{};
    std::uint64_t last_good_generation{};
    std::uint64_t highest_accepted_generation{};
    std::uint64_t overlay_sequence{};
    std::uint64_t rejected_activations{};
    std::uint64_t retired_mount_cleanup_failures{};
    bool overlay_attached{};
    std::string active_provider_id;
};

/** Safe-frame bridge between verified C9.2 cooked generations, existing
 * cooked_asset_catalog/VFS, DDC persistent active + last-good retention, and
 * the existing runtime_update_receiver for *artifact-level* live overlays.
 *
 * A remote cooked *package* is not an update-receiver blob: it is mounted as
 * its own verified base provider beneath an optional higher-priority overlay.
 * Only the receiver ingests individual artifact revisions/tombstones.
 */
class remote_runtime_generation_session
{
public:
    remote_runtime_generation_session(cooked_asset_catalog& catalog, derived_data_cache& cache,
                                      io::async_file_service& files, remote_content_expectations expected,
                                      std::int32_t initial_priority = 25);
    ~remote_runtime_generation_session();

    remote_runtime_generation_session(const remote_runtime_generation_session&) = delete;
    remote_runtime_generation_session& operator=(const remote_runtime_generation_session&) = delete;

    /** Verify every immutable package chunk/artifact first, then promote a
     * single matching generation above the prior remote base. On failure,
     * current catalog and active generation are untouched. Retention update
     * is persisted before promotion, and is restored if mount fails.
     *
     * The caller must not prune the same cache concurrently with activation.
     */
    [[nodiscard]] asset_status activate_verified(const remote_content_release& candidate);

    /** Explicitly revert the latest base promotion by unmounting its provider
     * and revealing the verified prior base; reject if any live overlay is
     * active so overlay bytes cannot silently shadow the older generation.
     */
    [[nodiscard]] asset_status rollback();

    /** Attach the existing CAS artifact update receiver above remote/package
     * bases. Revalidates any persisted overlay against the target and runtime
     * build before exposing it. Does not import arbitrary HTTP package blobs.
     */
    [[nodiscard]] asset_status attach_overlay(std::filesystem::path root);

    /** The update receiver still owns complete SHA-verified artifact ingestion,
     * tombstones, monotonic sequence checks and atomic manifest publication.
     */
    [[nodiscard]] asset_status ingest_overlay(const runtime_update_manifest& update,
                                              std::span<const std::pair<content_hash, io::file_buffer>> blobs);

    [[nodiscard]] remote_runtime_generation_state state() const;

private:
    [[nodiscard]] bool compatible(const remote_content_release& release) const noexcept;

    cooked_asset_catalog* catalog_{};
    derived_data_cache* cache_{};
    io::async_file_service* files_{};
    remote_content_expectations expected_;
    std::int32_t next_priority_{};
    std::uint64_t highest_accepted_{};
    std::uint64_t rejected_activations_{};
    std::uint64_t cleanup_failures_{};
    std::optional<remote_content_release> active_;
    std::optional<remote_content_release> last_good_;
    std::string active_provider_id_;
    std::string last_good_provider_id_;
    std::shared_ptr<cas_overlay_provider> overlay_;
    std::unique_ptr<runtime_update_receiver> receiver_;
};

} // namespace arc::assets
