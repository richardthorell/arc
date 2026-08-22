#pragma once

/**
 * @file arc/render_tools/material_asset.h
 * @brief Versioned authored-material migration and cooked material package schema.
 */

#include <arc/core/result.h>
#include <arc/render/material_pass.h>
#include <arc/render/shader.h>

#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace arc::render::tools
{

/** Current authored material document version used by the editor and cooker. */
inline constexpr std::uint32_t material_authoring_version = 4;
/** Current cooked ARC material package schema version. */
inline constexpr std::uint32_t material_package_version = 3;
/** Stable signature of the legacy version-2 cooked material payload. */
inline constexpr std::string_view material_package_v2_signature = "ARC_MATERIAL_2";
/** Stable signature of the version-3 pass-aware cooked material payload. */
inline constexpr std::string_view material_package_signature = "ARC_MATERIAL_3";

/** Failure category produced while reading an authored or cooked material document. */
enum class material_asset_error_code : std::uint8_t
{
    malformed_json,
    unsupported_version,
    invalid_document,
    corrupt_package
};

/** Structured material schema error. */
struct material_asset_error
{
    material_asset_error_code code{material_asset_error_code::invalid_document};
    std::string message;
};

/**
 * @brief Canonical authored material document consumed by cooking tools.
 *
 * ARC accepts historical authored versions and normalizes them to
 * @ref material_authoring_version without dropping unknown fields. The graph,
 * shader path, and pass-routing properties are extracted for tools while
 * canonical_json remains the authoritative migrated document persisted into
 * cooked packages.
 */
struct material_authoring_document
{
    std::uint32_t source_version{1};
    std::uint32_t version{material_authoring_version};
    bool migrated{};
    std::string canonical_json;
    std::string graph_json;
    std::string shader_path;
    material_domain domain{material_domain::surface};
    material_shading_model shading_model{material_shading_model::standard};
    material_alpha_mode alpha_mode{material_alpha_mode::opaque};
    bool double_sided{};
};

using material_authoring_result = core::result<material_authoring_document, material_asset_error>;

/** Parse versions 1-4 and migrate them in memory to the canonical authored material schema. */
[[nodiscard]] material_authoring_result parse_material_authoring_json(std::string_view source);

/** Data stored by the legacy ARC_MATERIAL_2 cooked package envelope. */
struct material_package_v2
{
    shader_package_id shader_package{};
    shader_permutation_id permutation{};
    std::vector<shader_parameter_descriptor> parameters;
    std::string canonical_document_json;
};

/** Data stored by the pass-aware ARC_MATERIAL_3 cooked package envelope. */
struct material_package_v3
{
    material_compiled_program compiled;
    std::vector<shader_parameter_descriptor> parameters;
    std::string canonical_document_json;
};

using material_package_v3_result = core::result<material_package_v3, material_asset_error>;

/** Serialize the legacy version-2 material package without changing its byte layout. */
[[nodiscard]] std::vector<std::byte> serialize_material_package_v2(const material_package_v2& package);

/** Serialize deterministic pass-aware ARC_MATERIAL_3 bytes. */
[[nodiscard]] std::vector<std::byte> serialize_material_package_v3(const material_package_v3& package);

/** Decode and validate deterministic ARC_MATERIAL_3 bytes. */
[[nodiscard]] material_package_v3_result deserialize_material_package_v3(std::span<const std::byte> bytes);

} // namespace arc::render::tools
