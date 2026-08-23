#pragma once

/**
 * @file arc/render_tools/material_asset.h
 * @brief Current authored-material and cooked material package schema.
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
/** Stable signature of the pass-aware cooked material payload. */
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
 * ARC material authoring is intentionally strict: the document must use the
 * current schema and must provide exactly one compiled implementation, either a
 * material graph or a handwritten Material Shader path.
 */
struct material_authoring_document
{
    std::uint32_t version{material_authoring_version};
    std::string canonical_json;
    std::string graph_json;
    std::string shader_path;
    material_domain domain{material_domain::surface};
    material_shading_model shading_model{material_shading_model::standard};
    material_alpha_mode alpha_mode{material_alpha_mode::opaque};
    bool double_sided{};
};

using material_authoring_result = core::result<material_authoring_document, material_asset_error>;

/** Parse and validate a current-version material with exactly one compiled implementation. */
[[nodiscard]] material_authoring_result parse_material_authoring_json(std::string_view source);

/** Data stored by the pass-aware ARC_MATERIAL_3 cooked package envelope. */
struct material_package_v3
{
    material_compiled_program compiled;
    std::vector<shader_parameter_descriptor> parameters;
    std::string canonical_document_json;
};

using material_package_v3_result = core::result<material_package_v3, material_asset_error>;

/** Serialize deterministic pass-aware ARC_MATERIAL_3 bytes. */
[[nodiscard]] std::vector<std::byte> serialize_material_package_v3(const material_package_v3& package);

/** Decode and validate deterministic ARC_MATERIAL_3 bytes. */
[[nodiscard]] material_package_v3_result deserialize_material_package_v3(std::span<const std::byte> bytes);

} // namespace arc::render::tools
