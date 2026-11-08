#pragma once

/** @file
 * @brief Mount D0-referenced cooked content into a metadata-first runtime asset catalog.
 *
 * The distribution has already passed D1.2 verification. This stage does not
 * load scenes, allocate device resources or acquire bundle residency.
 */

#include <arc/assets/runtime_content.h>
#include <arc/project/runtime_boot.h>

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace arc::project
{

/** Services remain owned by the platform host for the entire catalog session. */
struct runtime_catalog_request
{
    const runtime_boot_context& boot;
    io::virtual_file_system& files;
    io::async_file_service* native_files{}; ///< Required only for an on-disk distribution.
    io::virtual_path native_mount;          ///< Optional; defaults to package://runtime.
    io::virtual_path artifact_mount;        ///< Optional; defaults to artifact://game.
};

class runtime_catalog_session;

using runtime_catalog_result = core::result<std::unique_ptr<runtime_catalog_session>, project_error>;

/**
 * Owns only mounts created during this bootstrap. Never unmounts an externally
 * supplied Android/package VFS source. Destroy before the host's VFS/files.
 */
class runtime_catalog_session final
{
public:
    ~runtime_catalog_session();

    runtime_catalog_session(const runtime_catalog_session&) = delete;
    runtime_catalog_session& operator=(const runtime_catalog_session&) = delete;
    runtime_catalog_session(runtime_catalog_session&&) = delete;
    runtime_catalog_session& operator=(runtime_catalog_session&&) = delete;

    [[nodiscard]] assets::cooked_asset_catalog& catalog() noexcept;
    [[nodiscard]] const assets::cooked_asset_catalog& catalog() const noexcept;

    /** D0 launch-order GUIDs verified to exist as cooked scene metadata. */
    [[nodiscard]] const std::vector<assets::asset_guid>& startup_scenes() const noexcept;
    /** Requested C3 bundle IDs verified in the mounted cook manifests. */
    [[nodiscard]] const std::vector<assets::content_bundle_id>& startup_bundles() const noexcept;

private:
    friend runtime_catalog_result bootstrap_runtime_catalog(const runtime_catalog_request& request);
    explicit runtime_catalog_session(io::virtual_file_system& files) noexcept;

    io::virtual_file_system* files_{};
    io::mount_id native_mount_{};
    std::unique_ptr<assets::cooked_asset_catalog> catalog_;
    std::vector<assets::asset_guid> startup_scenes_;
    std::vector<assets::content_bundle_id> startup_bundles_;
};

/**
 * Validate D0-referenced cook targets and bundle/scene metadata, then mount
 * packages into the existing cooked_asset_catalog in deterministic order.
 * Chunk handles are resolved, but artifact payload bytes remain nonresident.
 *
 * Any failure destroys the temporary session and rolls back all mounts. On
 * success the host owns the returned session and controls its lifetime.
 */
[[nodiscard]] runtime_catalog_result bootstrap_runtime_catalog(const runtime_catalog_request& request);

} // namespace arc::project
