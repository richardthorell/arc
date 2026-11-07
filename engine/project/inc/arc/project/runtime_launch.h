#pragma once

/** @file
 * @brief Validated, platform-neutral inputs for packaged game and server startup.
 *
 * D1.1 defines the contract only. Manifest discovery, content mounting, module
 * activation and the running session are implemented by subsequent D1 slices.
 */

#include <arc/assets/content_bundle.h>
#include <arc/framework/capabilities.h>
#include <arc/io/virtual_file_system.h>
#include <arc/project/distribution_manifest.h>

#include <filesystem>
#include <string>
#include <vector>

namespace arc::project
{

/** The requested execution role must match the signed/staged D0 manifest. */
enum class runtime_launch_role : std::uint8_t
{
    client,
    server
};

/**
 * The platform host supplies one source of distribution bytes. The shared
 * startup path must not inspect the working directory, source tree or SDK.
 */
enum class runtime_distribution_location_kind : std::uint8_t
{
    executable, ///< Absolute host executable path; D1.2 discovers its distribution root.
    directory,  ///< Explicit absolute distribution root (e.g. relocated smoke test).
    mounted_vfs ///< Pre-mounted provider, including Android package assets.
};

struct runtime_distribution_location
{
    runtime_distribution_location_kind kind{runtime_distribution_location_kind::executable};
    std::filesystem::path native_path; ///< Executable path or directory for native modes, never serialized.
    io::virtual_file_system* files{};  ///< Borrowed VFS, required for mounted_vfs.
    io::virtual_path virtual_root;     ///< Already-mounted virtual root for mounted_vfs.
};

/** Host-owned resources stay outside the serialized distribution manifest. */
struct runtime_launch_host
{
    runtime_distribution_location distribution;
    std::filesystem::path writable_storage_root; ///< Absolute OS-provided user-data root; not a source/cache path.
    framework::platform_capabilities capabilities;
    std::string device_profile; ///< Optional device/profile selector; empty delegates selection to runtime.
};

/**
 * D0 is the sole authority for project, engine, target, role, runtime config,
 * ordered startup scenes, and content-manifest paths. D1 adds only host state
 * and requested bundle activation roots (C3/C4 bundle IDs, not asset paths).
 */
struct runtime_launch_descriptor
{
    distribution_manifest manifest;
    runtime_launch_role role{runtime_launch_role::client};
    runtime_launch_host host;
    std::vector<assets::content_bundle_id> startup_bundles;
};

/** Pure validation: no filesystem reads, VFS mounts, SDK lookup or game startup. */
[[nodiscard]] project_status validate_runtime_launch(const runtime_launch_descriptor& descriptor);

/**
 * Immutable-by-convention validated launch inputs passed to later D1 stages.
 * Ownership of the VFS remains with the host for the entire runtime session.
 */
struct runtime_launch_context
{
    distribution_manifest manifest;
    runtime_launch_role role{runtime_launch_role::client};
    runtime_launch_host host;
    std::vector<assets::content_bundle_id> startup_bundles;
};

using runtime_launch_result = core::result<runtime_launch_context, project_error>;

/** Validate once, transfer the launch inputs, and preserve scene/bundle ordering. */
[[nodiscard]] runtime_launch_result prepare_runtime_launch(runtime_launch_descriptor descriptor);

} // namespace arc::project
