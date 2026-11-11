#pragma once

#include <arc/project/distribution_assembler.h>

#include <filesystem>
#include <string>
#include <vector>

namespace arc::project
{

struct windows_dependency_closure
{
    std::vector<distribution_assembly_file> files;
    std::vector<distribution_native_dependency> dependencies;
};

using windows_dependency_result = core::result<windows_dependency_closure, project_error>;

/**
 * Inspect PE import tables for the executable, explicit plugins and their
 * transitive non-system DLLs. The search roots must be supplied by the caller
 * and are never serialized. Missing imports fail before distribution assembly.
 * Delay-loaded imports and non-imported dynamic loads are not inferred.
 */
[[nodiscard]] windows_dependency_result resolve_windows_dependencies(
    const std::filesystem::path& executable, const std::vector<std::filesystem::path>& plugins,
    const std::vector<std::filesystem::path>& search_directories);

} // namespace arc::project
