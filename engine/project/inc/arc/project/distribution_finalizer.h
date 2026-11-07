#pragma once

#include <arc/project/distribution_manifest.h>
#include <arc/project/package_workflow.h>

#include <filesystem>
#include <string_view>
#include <vector>

namespace arc::project
{

/**
 * Dependency closure discovered by the common assembler from build/link metadata
 * and project plugin declarations. The assembler must inspect transitive native
 * dependencies and classify host-provided/system libraries separately.
 *
 * Paths here are final distribution-relative destinations, not build source paths.
 * Empty dependencies are valid for statically linked distributions.
 */
struct distribution_native_dependency
{
    std::string destination; ///< Staged relative filename, e.g. bin/physics.dll.
    std::string kind;        ///< library or plugin; plugins remain distinguishable.
    bool required{true};     ///< Optional dependencies may be absent.
};

/**
 * Finalizer invocation receives a fully staged directory and a validated
 * distribution manifest. The assembler (D2) is responsible for discovering,
 * copying and hashing files before calling this boundary.
 */
struct distribution_finalize_request
{
    const package_workflow_plan& workflow;
    const distribution_manifest& manifest;
    std::filesystem::path staging_root;                       ///< Host path, never serialized.
    std::filesystem::path output_root;                        ///< Host path, never serialized.
    std::vector<distribution_native_dependency> dependencies; ///< Resolved native dependency closure.
};

/** Native packagers implement this interface, not the shared assembler. */
class distribution_finalizer
{
public:
    virtual ~distribution_finalizer() = default;
    [[nodiscard]] virtual std::string_view id() const noexcept = 0;
    [[nodiscard]] virtual project_status validate(const distribution_finalize_request& request) const = 0;
    [[nodiscard]] virtual project_status finalize(const distribution_finalize_request& request) = 0;
};

/**
 * Fail before invoking a platform-specific adapter when the staged manifest
 * disagrees with the workflow. No filesystem mutation occurs in this function.
 */
[[nodiscard]] project_status validate_distribution_finalization(const distribution_finalize_request& request,
                                                                const distribution_finalizer& finalizer);

/** Validate, then dispatch to the selected native finalizer. */
[[nodiscard]] project_status run_distribution_finalizer(const distribution_finalize_request& request,
                                                        distribution_finalizer& finalizer);

} // namespace arc::project
