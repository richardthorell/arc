#include <arc/project/package_workflow.h>

#include <algorithm>
#include <string_view>

namespace arc::project
{
namespace
{
bool has_enabled_module(const project_descriptor& descriptor, module_kind kind)
{
    return std::any_of(descriptor.modules.begin(), descriptor.modules.end(),
                       [kind](const auto& module) { return module.enabled && module.kind == kind; });
}

project_error workflow_error(project_error_code code, std::string field, std::string message)
{
    return {.code = code, .field = std::move(field), .message = std::move(message)};
}
} // namespace

package_workflow_result plan_package_workflow(const project_descriptor& descriptor, const project_context& context,
                                              const package_workflow_request& request)
{
    if (descriptor.cook_profiles.empty())
        return package_workflow_result::failure(workflow_error(project_error_code::invalid_descriptor, "cookProfiles",
                                                               "Project does not declare a cook profile"));

    const auto profile_id = request.profile.empty() ? descriptor.cook_profiles.front().id : request.profile;
    const auto profile = std::find_if(descriptor.cook_profiles.begin(), descriptor.cook_profiles.end(),
                                      [&](const auto& candidate) { return candidate.id == profile_id; });
    if (profile == descriptor.cook_profiles.end())
        return package_workflow_result::failure(
            workflow_error(project_error_code::invalid_descriptor, "cookProfiles",
                           "Cook profile '" + profile_id + "' is not declared by the project"));

    // A cook profile is not an authority to package a disabled or unsupported target.
    const auto& platform = profile->platform;
    const auto& renderer = profile->renderer;
    const auto& architecture = profile->architecture;
    if ((platform != "windows" && platform != "linux" && platform != "macos" && platform != "android" &&
         platform != "ios") ||
        (architecture != "x86_64" && architecture != "arm64") ||
        (renderer != "none" && renderer != "vulkan" && renderer != "direct3d12" && renderer != "metal"))
        return package_workflow_result::failure(
            workflow_error(project_error_code::unsupported_platform, "cookProfiles",
                           "Cook profile contains unsupported platform, architecture or renderer"));

    if ((renderer == "direct3d12" && platform != "windows") ||
        (renderer == "metal" && platform != "macos" && platform != "ios") ||
        (platform == "android" && renderer != "vulkan") ||
        ((platform == "android" || platform == "ios") && architecture != "arm64"))
        return package_workflow_result::failure(
            workflow_error(project_error_code::unsupported_platform, "cookProfiles",
                           "Cook profile renderer or architecture is incompatible with target platform"));

    const auto expected_target = platform + "-" + (architecture == "x86_64" ? "x64" : architecture) + "-" +
                                 (renderer == "none" ? "headless" : renderer);
    if (!request.target_id.empty() && request.target_id != expected_target)
        return package_workflow_result::failure(
            workflow_error(project_error_code::invalid_descriptor, "target",
                           "Selected target does not match the selected cook profile"));
    const auto target =
        std::find_if(descriptor.target_platforms.begin(), descriptor.target_platforms.end(),
                     [&](const auto& candidate) { return candidate.id == expected_target && candidate.enabled; });
    if (target == descriptor.target_platforms.end())
        return package_workflow_result::failure(
            workflow_error(project_error_code::unsupported_platform, "targetPlatforms",
                           "Cook profile '" + profile_id + "' requires enabled target '" + expected_target + "'"));

    const auto default_finalizer = platform == "windows"   ? "windows-directory"
                                   : platform == "android" ? "android-gradle"
                                   : platform == "linux"   ? "linux-directory"
                                   : platform == "macos"   ? "macos-bundle"
                                                           : "ios-bundle";
    const auto finalizer = request.finalizer.empty() ? default_finalizer : request.finalizer;
    if (finalizer != default_finalizer)
        return package_workflow_result::failure(
            workflow_error(project_error_code::unsupported_platform, "finalizer",
                           "Finalizer '" + finalizer + "' is incompatible with target platform '" + platform + "'"));

    if (std::find(descriptor.build_configurations.begin(), descriptor.build_configurations.end(),
                  profile->configuration) == descriptor.build_configurations.end())
        return package_workflow_result::failure(
            workflow_error(project_error_code::invalid_descriptor, "buildConfigurations",
                           "Cook profile '" + profile_id + "' requires undeclared build configuration '" +
                               profile->configuration + "'"));

    const auto module_role = request.role == package_workflow_role::server ? module_kind::server : module_kind::runtime;
    if (request.action != package_workflow_action::cook && !has_enabled_module(descriptor, module_role))
    {
        const auto role_name = request.role == package_workflow_role::server ? "server" : "runtime";
        return package_workflow_result::failure(
            workflow_error(project_error_code::missing_module, "modules",
                           "Project does not declare an enabled " + std::string(role_name) + " module for packaging"));
    }

    if (request.role == package_workflow_role::runtime && profile->renderer == "none")
        return package_workflow_result::failure(workflow_error(project_error_code::invalid_descriptor, "cookProfiles",
                                                               "Client workflow requires a graphical cook profile"));

    if (request.role == package_workflow_role::server && profile->renderer != "none")
        return package_workflow_result::failure(
            workflow_error(project_error_code::invalid_descriptor, "cookProfiles",
                           "Dedicated-server workflow requires a cook profile with renderer 'none'"));

    auto package_root = descriptor.package.output;
    if (package_root.is_relative()) package_root = context.root / package_root;
    package_root = package_root.lexically_normal();

    const auto role_directory = request.role == package_workflow_role::server ? "Server" : "Runtime";
    return package_workflow_result::success(
        {.profile = profile->id,
         .project_guid = descriptor.guid,
         .engine_version = descriptor.engine_version,
         .configuration = profile->configuration,
         .target_id = target->id,
         .platform = platform,
         .architecture = architecture,
         .renderer = renderer,
         .finalizer = finalizer,
         .action = request.action,
         .role = request.role,
         .cook_output = (context.build_root / "Cooked" / profile->id).lexically_normal(),
         .package_output = (package_root / profile->id / role_directory).lexically_normal()});
}

const char* workflow_stage_name(project_workflow_stage stage) noexcept
{
    switch (stage)
    {
        case project_workflow_stage::validate:
            return "validate";
        case project_workflow_stage::snapshot:
            return "snapshot";
        case project_workflow_stage::configure:
            return "configure";
        case project_workflow_stage::compile:
            return "compile";
        case project_workflow_stage::cook:
            return "cook";
        case project_workflow_stage::package:
            return "package";
        case project_workflow_stage::deploy:
            return "deploy";
        case project_workflow_stage::launch:
            return "launch";
        case project_workflow_stage::connect:
            return "connect";
    }
    return "unknown";
}

project_workflow_result plan_project_workflow(const project_descriptor& descriptor, const project_context& context,
                                              const project_workflow_request& request)
{
    using action = project_workflow_action;
    using stage = project_workflow_stage;
    const bool runs = request.action == action::play || request.action == action::run_built;
    if (request.profile.empty() || request.target_id.empty())
        return project_workflow_result::failure(
            workflow_error(project_error_code::invalid_descriptor, "profile",
                           "Workflow requires explicit profile and target identities"));
    if (runs == (request.destination == project_workflow_destination::none))
        return project_workflow_result::failure(
            workflow_error(project_error_code::invalid_descriptor, "destination",
                           "Only Play and Run Built require a destination; build actions must not select one"));
    if (request.action == action::play && request.role != package_workflow_role::runtime)
        return project_workflow_result::failure(workflow_error(project_error_code::invalid_descriptor, "role",
                                                               "Current-scene Play requires the runtime role"));
    const bool safe_session =
        !request.session_id.empty() && request.session_id.size() <= 64 &&
        std::all_of(request.session_id.begin(), request.session_id.end(), [](char c)
                    { return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9') || c == '-'; });
    if ((request.action == action::play && !safe_session) ||
        (request.action != action::play && !request.session_id.empty()))
        return project_workflow_result::failure(
            workflow_error(project_error_code::invalid_descriptor, "sessionId",
                           "Play requires a bounded alphanumeric session ID; other actions must not provide one"));

    auto output = plan_package_workflow(
        descriptor, context,
        {.profile = request.profile,
         .action = request.action == action::cook ? package_workflow_action::cook : package_workflow_action::package,
         .role = request.role,
         .target_id = request.target_id});
    if (!output) return project_workflow_result::failure(output.error());
    if (request.destination == project_workflow_destination::device && output.value().platform != "android")
        return project_workflow_result::failure(workflow_error(project_error_code::unsupported_platform, "destination",
                                                               "Device workflows currently support Android only"));
    if (request.destination == project_workflow_destination::local &&
        (output.value().platform == "android" || output.value().platform == "ios"))
        return project_workflow_result::failure(workflow_error(project_error_code::unsupported_platform, "destination",
                                                               "Mobile targets require a device destination"));
    if (request.action == action::play && output.value().configuration == "Shipping")
        return project_workflow_result::failure(
            workflow_error(project_error_code::invalid_descriptor, "configuration",
                           "Controlled Play requires a non-Shipping development profile"));

    project_workflow_plan result{.output = std::move(output.value()),
                                 .action = request.action,
                                 .destination = request.destination,
                                 .session_id = request.session_id,
                                 .stages = {stage::validate}};
    if (request.action == action::play)
    {
        const auto session_root = context.build_root / "PlaySessions" / ("session-" + request.session_id);
        result.output.cook_output = (session_root / "Cooked").lexically_normal();
        result.output.package_output = (session_root / "Package").lexically_normal();
    }
    if (request.action == action::play) result.stages.push_back(stage::snapshot);
    if (request.action != action::cook && request.action != action::run_built)
    {
        result.stages.push_back(stage::configure);
        if (request.action != action::configure) result.stages.push_back(stage::compile);
    }
    if (request.action == action::cook || request.action == action::build || request.action == action::play)
        result.stages.push_back(stage::cook);
    if (request.action == action::build || request.action == action::play) result.stages.push_back(stage::package);
    if (runs)
    {
        if (request.destination == project_workflow_destination::device) result.stages.push_back(stage::deploy);
        result.stages.push_back(stage::launch);
        if (request.action == action::play) result.stages.push_back(stage::connect);
    }
    return project_workflow_result::success(std::move(result));
}

} // namespace arc::project
