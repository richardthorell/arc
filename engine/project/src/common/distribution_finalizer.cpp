#include <arc/project/distribution_finalizer.h>

#include <utility>

namespace arc::project
{
namespace
{
project_status incompatible(std::string field, std::string message)
{
    return project_status::failure(
        {.code = project_error_code::invalid_descriptor, .field = std::move(field), .message = std::move(message)});
}
} // namespace

project_status validate_distribution_finalization(const distribution_finalize_request& request,
                                                  const distribution_finalizer& finalizer)
{
    if (const auto result = validate_distribution_manifest(request.manifest); !result) return result;

    const auto& workflow = request.workflow;
    const auto& manifest = request.manifest;
    if (workflow.project_guid != manifest.project_guid || workflow.engine_version != manifest.engine_version)
        return incompatible("projectGuid/engineVersion",
                            "Distribution manifest project or engine differs from workflow");

    if (workflow.profile != manifest.target.cook_profile || workflow.target_id != manifest.target.id ||
        workflow.platform != manifest.target.platform || workflow.architecture != manifest.target.architecture ||
        workflow.renderer != manifest.target.renderer || workflow.configuration != manifest.target.configuration)
        return incompatible("target", "Distribution manifest target disagrees with resolved project workflow");

    if (manifest.role != (workflow.role == package_workflow_role::server ? "server" : "client"))
        return incompatible("role", "Distribution manifest role disagrees with resolved project workflow");
    if (manifest.finalizer != workflow.finalizer || finalizer.id() != workflow.finalizer)
        return incompatible("finalizer", "Selected distribution finalizer does not match workflow and manifest");

    if (request.staging_root.empty() || request.output_root.empty() || !request.staging_root.is_absolute() ||
        !request.output_root.is_absolute())
        return incompatible("paths", "Finalizer requires absolute staging and output roots");
    if (request.staging_root.lexically_normal() == request.output_root.lexically_normal())
        return incompatible("paths", "Staging and output roots must be distinct");

    return finalizer.validate(request);
}

project_status run_distribution_finalizer(const distribution_finalize_request& request,
                                          distribution_finalizer& finalizer)
{
    if (const auto validation = validate_distribution_finalization(request, finalizer); !validation) return validation;
    return finalizer.finalize(request);
}

} // namespace arc::project
