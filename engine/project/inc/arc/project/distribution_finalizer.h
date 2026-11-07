#pragma once

#include <arc/project/distribution_manifest.h>
#include <arc/project/package_workflow.h>

#include <filesystem>
#include <string_view>

namespace arc::project
{

/**
 * Finalizer invocation receives a fully staged directory and a validated
 * distribution manifest. The assembler (D2) is responsible for discovering,
 * copying and hashing files before calling this boundary.
 */
struct distribution_finalize_request
{
    const package_workflow_plan& workflow;
    const distribution_manifest& manifest;
    std::filesystem::path staging_root; ///< Host path, never serialized.
    std::filesystem::path output_root;  ///< Host path, never serialized.
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
