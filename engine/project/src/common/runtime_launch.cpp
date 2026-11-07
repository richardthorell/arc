#include <arc/project/runtime_launch.h>

#include <set>
#include <string_view>
#include <utility>

namespace arc::project
{
namespace
{

project_status invalid(project_error_code code, std::string field, std::string message)
{
    return project_status::failure({.code = code, .field = std::move(field), .message = std::move(message)});
}

std::string_view platform_name(framework::platform_family family)
{
    switch (family)
    {
        case framework::platform_family::windows:
            return "windows";
        case framework::platform_family::linux_os:
            return "linux";
        case framework::platform_family::android:
            return "android";
        case framework::platform_family::macos:
            return "macos";
        case framework::platform_family::ios:
            return "ios";
        case framework::platform_family::unknown:
            break;
    }
    return {};
}

bool absolute_normalized(const std::filesystem::path& path)
{
    return !path.empty() && path.is_absolute() && path == path.lexically_normal();
}

} // namespace

project_status validate_runtime_launch(const runtime_launch_descriptor& descriptor)
{
    // Keep target, role, version, scene GUID and staged-path policy in D0.
    const auto valid_manifest = validate_distribution_manifest(descriptor.manifest);
    if (!valid_manifest) return valid_manifest;

    if (descriptor.role != runtime_launch_role::client && descriptor.role != runtime_launch_role::server)
        return invalid(project_error_code::invalid_descriptor, "role", "Unknown packaged runtime launch role");

    const auto expected_role = descriptor.role == runtime_launch_role::server ? "server" : "client";
    if (descriptor.manifest.role != expected_role)
        return invalid(project_error_code::invalid_descriptor, "role",
                       "Platform host requested '" + std::string(expected_role) +
                           "' but the distribution manifest declares '" + descriptor.manifest.role + "'");

    // D0 rejects a client with no startupScenes; do not duplicate that policy
    // with a different error code or accidentally allow an implicit empty world.

    const auto actual_platform = platform_name(descriptor.host.capabilities.family);
    if (actual_platform.empty() || actual_platform != descriptor.manifest.target.platform)
        return invalid(project_error_code::unsupported_platform, "host.capabilities.family",
                       "Distribution target platform does not match the active platform host");

    const auto& location = descriptor.host.distribution;
    switch (location.kind)
    {
        case runtime_distribution_location_kind::executable:
        case runtime_distribution_location_kind::directory:
            if (!absolute_normalized(location.native_path) || location.files || !location.virtual_root.empty())
                return invalid(project_error_code::unsafe_path, "host.distribution",
                               "Native distribution location requires one absolute normalized host path "
                               "and must not also supply a VFS mount");
            break;

        case runtime_distribution_location_kind::mounted_vfs:
            if (!location.native_path.empty() || !location.files || location.virtual_root.empty())
                return invalid(project_error_code::invalid_descriptor, "host.distribution",
                               "Mounted distribution requires a VFS and virtual root, not an OS path");
            break;

        default:
            return invalid(project_error_code::invalid_descriptor, "host.distribution",
                           "Unknown distribution location kind");
    }

    if (!absolute_normalized(descriptor.host.writable_storage_root))
        return invalid(project_error_code::unsafe_path, "host.writableStorageRoot",
                       "Host must supply an absolute normalized writable user-data path");

    // Device profile IDs are opaque C6 selectors, never filesystem paths.
    const auto& device_profile = descriptor.host.device_profile;
    if (device_profile.find_first_of("\r\n\t\\") != std::string::npos)
        return invalid(project_error_code::invalid_descriptor, "host.deviceProfile",
                       "Device profile identifier must not contain control characters or backslashes");

    std::set<assets::content_bundle_id> seen;
    for (const auto& bundle : descriptor.startup_bundles)
    {
        if (bundle.empty() || !seen.insert(bundle).second)
            return invalid(project_error_code::invalid_descriptor, "startupBundles",
                           "Startup Content Bundle IDs must be nonempty and unique");
    }

    return project_status::success();
}

runtime_launch_result prepare_runtime_launch(runtime_launch_descriptor descriptor)
{
    auto checked = validate_runtime_launch(descriptor);
    if (!checked) return runtime_launch_result::failure(std::move(checked).error());

    return runtime_launch_result::success({.manifest = std::move(descriptor.manifest),
                                           .role = descriptor.role,
                                           .host = std::move(descriptor.host),
                                           .startup_bundles = std::move(descriptor.startup_bundles)});
}

} // namespace arc::project
