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
inline constexpr std::uint32_t material_package_version = 4;
/** Stable signature of the pass-aware cooked material payload. */
inline constexpr std::string_view material_package_signature = "ARC_MATERIAL_4";

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
    bool cast_shadows{true};
};

using material_authoring_result = core::result<material_authoring_document, material_asset_error>;

/** Parse and validate a current-version material with exactly one compiled implementation. */
[[nodiscard]] material_authoring_result parse_material_authoring_json(std::string_view source);

/** Data stored by the pass-aware ARC_MATERIAL_4 cooked package envelope. */
struct material_package_v4
{
    material_compiled_program compiled;
    std::vector<shader_parameter_descriptor> parameters;
    std::string canonical_document_json;
};

using material_package_v4_result = core::result<material_package_v4, material_asset_error>;

using material_package_v3 = material_package_v4;
using material_package_v3_result = material_package_v4_result;

/** Current authored Material Instance document version. */
inline constexpr std::uint32_t material_instance_authoring_version = 1;
/** Current cooked Material Instance package schema version. */
inline constexpr std::uint32_t material_instance_package_version = 1;
/** Stable signature of the cooked Material Instance payload. */
inline constexpr std::string_view material_instance_package_signature = "ARC_MATERIAL_INSTANCE_1";

/** GUID-backed authored asset reference used by Material Instance documents. */
struct material_instance_asset_reference
{
    std::string guid;
    std::string path_hint;
};

/** One stable parameter override stored by a Material Instance asset. */
struct material_instance_parameter_override_document
{
    shader_parameter_id parameter_id{};
    std::string value_json;
};

/** Per-selected-function input override authored on a Function Slot selection. */
struct material_instance_function_input_override_document
{
    std::string pin_id;
    std::string value_json;
};

/** Compile-time Function Slot selection stored by a Material Instance asset. */
struct material_instance_function_override_document
{
    std::string slot_id;
    material_instance_asset_reference function;
    std::vector<material_instance_function_input_override_document> input_overrides;
};

/** Canonical authored Material Instance document. */
struct material_instance_authoring_document
{
    std::uint32_t version{material_instance_authoring_version};
    std::string name;
    material_instance_asset_reference parent;
    std::vector<material_instance_parameter_override_document> parameter_overrides;
    std::vector<material_instance_function_override_document> function_overrides;
    std::string canonical_json;
};

using material_instance_authoring_result = core::result<material_instance_authoring_document, material_asset_error>;

/** Parse and validate a current-version first-class Material Instance document. */
[[nodiscard]] material_instance_authoring_result parse_material_instance_authoring_json(std::string_view source);

/**
 * Cooked Material Instance envelope.
 *
 * The nested Material package is the fully specialized parent program. Runtime/editor
 * loaders can therefore consume the same Material ABI payload as a normal Material,
 * while authored instance overrides and parent identity remain available separately.
 */
struct material_instance_package_v1
{
    std::string parent_guid;
    std::uint64_t function_specialization_key{};
    material_package_v4 material;
    std::string canonical_instance_json;
};

using material_instance_package_v1_result = core::result<material_instance_package_v1, material_asset_error>;

[[nodiscard]] std::vector<std::byte> serialize_material_instance_package_v1(const material_instance_package_v1& package);
[[nodiscard]] material_instance_package_v1_result
deserialize_material_instance_package_v1(std::span<const std::byte> bytes);

/** Serialize deterministic pass-aware ARC_MATERIAL_4 bytes. */
[[nodiscard]] std::vector<std::byte> serialize_material_package_v4(const material_package_v4& package);

[[nodiscard]] inline std::vector<std::byte> serialize_material_package_v3(const material_package_v3& package)
{
    return serialize_material_package_v4(package);
}

/** Decode and validate deterministic ARC_MATERIAL_4 bytes. */
[[nodiscard]] material_package_v4_result deserialize_material_package_v4(std::span<const std::byte> bytes);

[[nodiscard]] inline material_package_v3_result deserialize_material_package_v3(std::span<const std::byte> bytes)
{
    return deserialize_material_package_v4(bytes);
}

} // namespace arc::render::tools
