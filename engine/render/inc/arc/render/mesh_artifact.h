#pragma once

/** @file
 * @brief Validate and decode the conventional .arcmesh schema-v2 artifact.
 *
 * This format is produced by the existing ARC mesh cooker. No source asset,
 * filesystem path, render device, or Android host is required by the decoder.
 */

#include <arc/core/result.h>
#include <arc/render/mesh.h>

#include <cstddef>
#include <span>
#include <string>
#include <string_view>

namespace arc::render
{
struct decoded_conventional_mesh
{
    mesh_data mesh;
    float lod_ratio{1.0f};
    float geometric_error{};
};

/** Read LOD0 for an explicitly named mesh, or the sole mesh if name is empty.
 *
 * The entire artifact is validated, including other mesh/LOD records, before
 * the selected mesh is returned. Nonempty subresource names are exact matches.
 * Errors are descriptive and no partially decoded mesh is published.
 *
 * The current cooker uses native-layout mesh_vertex and uint32 index arrays.
 * This reader intentionally matches that existing schema; a future portable
 * schema must have an explicit version and a separate decoder.
 */
[[nodiscard]] core::result<decoded_conventional_mesh, std::string>
decode_conventional_mesh_artifact(std::span<const std::byte> bytes, std::string_view subresource = {});
} // namespace arc::render
