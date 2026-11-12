#pragma once

#include <arc/project/distribution_manifest.h>

#include <filesystem>

namespace arc::project
{

/**
 * Export the already-finalized and hash-verified distribution as a reproducible
 * ZIP32 archive (stored entries; no compression or host timestamps).
 *
 * Entry names are package-root relative and sorted; the canonical D0 manifest
 * is included. The output is published only after a complete successful write.
 */
[[nodiscard]] project_status write_distribution_zip(const std::filesystem::path& distribution_root,
                                                    const std::filesystem::path& zip_path);

} // namespace arc::project
