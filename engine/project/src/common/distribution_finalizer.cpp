#include <arc/project/distribution_finalizer.h>

#include <utility>
#include <algorithm>
#include <cctype>
#include <set>
#include <string>

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

    // The assembler owns transitive dependency discovery. The dispatcher checks
    // its declared closure against the actual, hashed staged file inventory.
    std::set<std::string> seen;
    for (const auto& dependency : request.dependencies)
    {
        if (dependency.kind != "library" && dependency.kind != "plugin")
            return incompatible("dependencies.kind", "Native dependency must be classified as library or plugin");
        // Reject destination path aliases by comparing only canonical inventory paths.
        const auto found = std::find_if(manifest.files.begin(), manifest.files.end(),
                                        [&](const auto& file) { return file.path == dependency.destination; });
        if (found == manifest.files.end())
        {
            if (dependency.required)
                return incompatible("dependencies", "Missing required native " + dependency.kind +
                                                        " in staged inventory: " + dependency.destination);
            continue;
        }
        if (found->kind != dependency.kind)
            return incompatible("dependencies.kind",
                                "Native dependency has wrong staged kind: " + dependency.destination);
        std::string key = dependency.destination;
        std::transform(key.begin(), key.end(), key.begin(),
                       [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
        if (!seen.insert(key).second)
            return incompatible("dependencies", "Duplicate native dependency: " + dependency.destination);
    }

    return finalizer.validate(request);
}

project_status run_distribution_finalizer(const distribution_finalize_request& request,
                                          distribution_finalizer& finalizer)
{
    if (const auto validation = validate_distribution_finalization(request, finalizer); !validation) return validation;
    return finalizer.finalize(request);
}

} // namespace arc::project
