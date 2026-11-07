#pragma once

#include <arc/project/project.h>

#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

namespace arc::project
{

inline constexpr std::string_view distribution_format = "arc.distribution";
inline constexpr std::uint32_t distribution_format_version = 1;

/** Manifest paths are portable, package-root-relative POSIX-style paths. */
struct distribution_file
{
    std::string path;
    std::string kind; ///< executable, library, plugin, configuration, content, metadata, or symbols.
    std::uint64_t size{};
    std::string sha256; ///< Lower-case SHA-256 of the final staged bytes.
};

struct distribution_target
{
    std::string id;            ///< .arcproject cook profile / target identity.
    std::string platform;      ///< windows, linux, macos, android, or ios.
    std::string architecture;  ///< x86_64 or arm64.
    std::string renderer;      ///< vulkan, direct3d12, metal, or none.
    std::string configuration; ///< Debug, RelWithDebInfo, or Shipping.
    std::string cook_profile;  ///< Selected existing arc-cook profile.
};

struct distribution_entrypoint
{
    std::string kind;  ///< executable or android-activity.
    std::string value; ///< Relative file path for executable; component name for activity.
};

/**
 * One target-neutral description of a staged runnable distribution.
 * Cook manifests remain the authority for assets, variants, and bundle closure.
 * No host/build/SDK paths may appear in this document.
 */
struct distribution_manifest
{
    std::uint32_t format_version{distribution_format_version};
    std::string project_guid;
    std::string engine_version;
    std::string sdk_version;
    distribution_target target;
    std::string role;      ///< client or server.
    std::string finalizer; ///< Finalizer ID, not an OS path.
    distribution_entrypoint entrypoint;
    std::vector<std::string> startup_scenes; ///< Ordered scene GUIDs.
    std::string runtime_config;              ///< Relative staged config path.
    std::vector<std::string> cook_manifests; ///< Relative staged arc-cook manifests.
    std::vector<distribution_file> files;    ///< Inventory sorted on serialization.
};

using distribution_manifest_result = core::result<distribution_manifest, project_error>;
using distribution_json_result = core::result<std::string, project_error>;

/** Validate all metadata and portable staged-path invariants without touching disk. */
[[nodiscard]] project_status validate_distribution_manifest(const distribution_manifest& manifest);

/** Canonical JSON, stable across file insertion order, without timestamps/host paths. */
[[nodiscard]] distribution_json_result serialize_distribution_manifest(const distribution_manifest& manifest);

/** Reject unsupported versions; v1 deliberately has no implicit migration. */
[[nodiscard]] distribution_manifest_result parse_distribution_manifest(std::string_view json);

} // namespace arc::project
