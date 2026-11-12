#pragma once

#include <arc/project/distribution_manifest.h>

#include <filesystem>
#include <vector>

namespace arc::project
{

/**
 * Publish an optional separate symbol directory for an existing Shipping
 * distribution. PDBs are explicitly selected and matched by basename to
 * inventoried executable/library/plugin binaries. No symbols enter runtime.
 */
[[nodiscard]] project_status write_distribution_symbols(const std::filesystem::path& distribution_root,
                                                        const std::filesystem::path& symbols_root,
                                                        const std::vector<std::filesystem::path>& symbol_files);

} // namespace arc::project
