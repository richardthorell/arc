#pragma once

#include <arc/project/distribution_finalizer.h>

#include <filesystem>
#include <string>
#include <vector>

namespace arc::project
{

/** A fully resolved input to stage; source paths never enter the D0 manifest. */
struct distribution_assembly_file
{
    std::filesystem::path source;
    std::string destination; ///< Portable distribution-relative path.
    std::string kind;        ///< D0 file kind (executable/library/plugin/configuration/content/metadata).
};

/**
 * The caller selects compiled binaries and the current cook-package output.
 * The assembler owns staging, hashing, D0 inventory creation and finalization.
 * Native dependency discovery (build/link metadata) remains a separate producer
 * of distribution_assembly_file and distribution_native_dependency entries.
 */
struct distribution_assembly_request
{
    const package_workflow_plan& workflow;
    distribution_manifest manifest; ///< Metadata only: files must be empty.
    std::filesystem::path output_root;
    std::vector<distribution_assembly_file> files;
    std::vector<distribution_native_dependency> dependencies;
};

using distribution_assembly_result = core::result<distribution_manifest, project_error>;

/**
 * Stage exact build/cook inputs into a private sibling directory, hash them
 * into D0, dispatch to the selected finalizer, then remove the private stage.
 * Neither a failed attempt nor an exception may publish a partial distribution.
 */
[[nodiscard]] distribution_assembly_result assemble_distribution(const distribution_assembly_request& request,
                                                                  distribution_finalizer& finalizer);

} // namespace arc::project
