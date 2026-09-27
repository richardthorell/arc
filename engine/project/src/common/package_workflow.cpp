#include <arc/project/package_workflow.h>

#include <algorithm>

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
         .configuration = profile->configuration,
         .action = request.action,
         .role = request.role,
         .cook_output = (context.build_root / "Cooked" / profile->id).lexically_normal(),
         .package_output = (package_root / profile->id / role_directory).lexically_normal()});
}

} // namespace arc::project
