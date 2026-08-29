#pragma once

#include <arc/core/result.h>
#include <arc/render/virtual_mesh.h>

#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace arc::render
{

inline constexpr std::uint32_t virtual_geometry_artifact_schema_version = 3;
inline constexpr std::uint32_t virtual_geometry_artifact_page_alignment = 4096;

/** @brief Failure categories produced while reading or writing a cooked virtual-geometry artifact. */
enum class virtual_geometry_artifact_error_code : std::uint8_t
{
    invalid_data,
    unsupported_version,
    out_of_bounds,
    integrity_failure,
    size_overflow
};

/** @brief Structured failure returned by virtual-geometry artifact operations. */
struct virtual_geometry_artifact_error
{
    virtual_geometry_artifact_error_code code{virtual_geometry_artifact_error_code::invalid_data};
    std::string message;
};

/** @brief One source mesh supplied to the deterministic `.arcvg` bundle encoder. */
struct virtual_geometry_artifact_source
{
    std::string_view name;
    std::uint64_t material_index{};
    const virtual_mesh_data* geometry{};
};

/** @brief Absolute byte range and integrity metadata for one independently readable page. */
struct virtual_geometry_artifact_page_range
{
    std::uint64_t offset{};
    std::uint32_t stored_size{};
    std::uint32_t decoded_size{};
    std::uint64_t content_hash{};
    bool root{};
};

/** @brief Range-readable metadata for one mesh record in a cooked `.arcvg` bundle. */
struct virtual_geometry_artifact_mesh_index
{
    std::string name;
    std::uint64_t material_index{};
    std::uint64_t metadata_offset{};
    std::uint64_t metadata_size{};
    std::vector<virtual_geometry_artifact_page_range> pages;
};

/** @brief Validated index for a versioned virtual-geometry artifact. */
struct virtual_geometry_artifact_index
{
    std::uint32_t schema_version{};
    std::uint64_t conventional_artifact_hash{};
    std::uint64_t artifact_size{};
    std::vector<virtual_geometry_artifact_mesh_index> meshes;
};

using virtual_geometry_artifact_bytes_result = core::result<std::vector<std::byte>, virtual_geometry_artifact_error>;
using virtual_geometry_artifact_index_result =
    core::result<virtual_geometry_artifact_index, virtual_geometry_artifact_error>;

/**
 * @brief Encode an indexed, page-aligned `.arcvg` schema-v3 artifact.
 * @param meshes Source mesh records; geometry pointers must remain valid for the duration of the call.
 * @param conventional_artifact_hash Hash linking the artifact to its conventional LOD companion.
 * @return Deterministic little-endian artifact bytes or a structured validation failure.
 */
[[nodiscard]] virtual_geometry_artifact_bytes_result
encode_virtual_geometry_artifact(std::span<const virtual_geometry_artifact_source> meshes,
                                 std::uint64_t conventional_artifact_hash = 0);

/**
 * @brief Validate the header, metadata table, and independently readable page ranges of a `.arcvg` artifact.
 * @param bytes Complete artifact bytes.
 * @return The range-readable artifact index or a structured validation failure.
 */
[[nodiscard]] virtual_geometry_artifact_index_result
inspect_virtual_geometry_artifact(std::span<const std::byte> bytes);

/**
 * @brief Read and integrity-check one independently aligned page blob.
 * @param bytes Complete artifact bytes or a mapped package view containing the requested page range.
 * @param index Previously validated artifact index.
 * @param mesh_index Mesh/subasset table index.
 * @param page_index Page index within the mesh entry.
 * @return Encoded page bytes ready for worker decompression.
 */
[[nodiscard]] virtual_geometry_artifact_bytes_result
read_virtual_geometry_artifact_page(std::span<const std::byte> bytes, const virtual_geometry_artifact_index& index,
                                    std::uint32_t mesh_index, std::uint32_t page_index);

} // namespace arc::render
