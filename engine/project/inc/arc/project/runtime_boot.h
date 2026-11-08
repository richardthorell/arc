#pragma once

/** @file
 * @brief Manifest discovery, compatibility checks and staged-file verification before packaged runtime startup.
 */

#include <arc/project/runtime_launch.h>

#include <filesystem>
#include <string>
#include <string_view>
#include <vector>

namespace arc::project
{

inline constexpr std::string_view runtime_distribution_manifest_name = "arc-distribution.json";

/** Bound the amount of work performed during ordinary client/server startup. */
enum class runtime_boot_integrity : std::uint8_t
{
    startup, ///< Check native inventory presence/size and SHA-256 of runtime config and cook manifests.
    full     ///< Hash every staged file. Useful for release verification and integration smoke tests.
};

/** Identity compiled into the host, not read from the untrusted distribution manifest. */
struct runtime_boot_expectations
{
    std::string engine_version; ///< Required exact engine version (native module ABI must match).
    std::string sdk_version;    ///< Required exact ARC SDK version.
    std::string architecture;   ///< Host architecture, e.g. x86_64 or arm64.
    std::string renderer;       ///< Host backend, e.g. vulkan or none for server.
    std::string configuration;  ///< Host build configuration, e.g. Shipping.
    std::string project_guid;   ///< Optional project identity bound to the host module.
    std::string target_id;      ///< Optional exact D0 target ID.
    std::string cook_profile;   ///< Optional exact cooked-content profile.
};

/** Input assembled by the native host before any content provider is opened. */
struct runtime_boot_request
{
    runtime_launch_role role{runtime_launch_role::client};
    runtime_launch_host host;
    runtime_boot_expectations expected;
    std::vector<assets::content_bundle_id> startup_bundles;
    runtime_boot_integrity integrity{runtime_boot_integrity::startup};
};

/**
 * A validated D1.1 launch plus the resolved native distribution root, if any.
 * A mounted-VFS launch keeps its virtual root in launch.host.distribution.
 * The original executable path is preserved in the D1.1 host contract.
 */
struct runtime_boot_context
{
    runtime_launch_context launch;
    std::filesystem::path native_distribution_root; ///< Canonical native root, empty for mounted providers.
};

using runtime_boot_result = core::result<runtime_boot_context, project_error>;

/**
 * Find and parse the D0 manifest, enforce binary/manifest target compatibility
 * and check the integrity of required staged inputs before runtime startup.
 *
 * Native executable discovery walks a bounded number of ancestors of the
 * supplied absolute executable path. It never consults the process CWD, source
 * checkout or installed SDK. A directory source is an explicit root. Mounted
 * VFS sources resolve under their supplied virtual root without native paths.
 *
 * No providers are mounted/unmounted; no game modules/worlds are started.
 * The caller must keep a borrowed VFS alive for the resulting launch context.
 */
[[nodiscard]] runtime_boot_result discover_runtime_boot(runtime_boot_request request);

} // namespace arc::project
