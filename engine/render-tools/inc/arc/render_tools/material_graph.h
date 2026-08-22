#pragma once

/**
 * @file arc/render_tools/material_graph.h
 * @brief Backend-neutral native material graph IR, reusable functions, descriptor compiler, and shader code generation.
 */

#include <arc/render/material_abi.h>
#include <arc/render/shader.h>

#include <array>
#include <compare>
#include <cstdint>
#include <span>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace arc::render::tools
{

/** @brief Version of the native material graph IR emitted by ARC tooling. */
inline constexpr std::uint32_t material_ir_version = 1;

/** @brief Version of the deterministic Material IR to Slang generator. */
inline constexpr std::uint32_t material_shader_codegen_version = 2;

/** @brief Version of first-class reusable Material/Shader Function documents. */
inline constexpr std::uint32_t material_function_version = 1;

/** @brief Operation represented by one normalized material IR node. */
enum class material_ir_node_kind : std::uint8_t
{
    constant,
    vector2,
    vector3,
    vector4,
    tex_coord,
    time,
    texture_sample,
    normal_map,
    saturate,
    clamp,
    lerp,
    add,
    subtract,
    multiply,
    divide,
    function_call,
    output
};

/** @brief Numeric literal carried by a material IR node or function-pin default. */
struct material_ir_literal
{
    std::array<float, 4> values{};
    std::uint8_t components{};

    friend constexpr auto operator<=>(const material_ir_literal&, const material_ir_literal&) noexcept = default;
};

/** @brief One typed public input or output in a reusable Material/Shader Function signature. */
struct material_function_pin
{
    std::string id;
    std::string name;
    shader_parameter_type type{shader_parameter_type::float32};
    bool has_default{};
    material_ir_literal default_value;

    friend auto operator<=>(const material_function_pin&, const material_function_pin&) = default;
};

/** @brief Validated public metadata for one reusable Material/Shader Function asset. */
struct material_function_descriptor
{
    std::string name;
    std::string description;
    std::vector<material_function_pin> inputs;
    std::vector<material_function_pin> outputs;
    bool shader_backed{};

    friend auto operator<=>(const material_function_descriptor&, const material_function_descriptor&) = default;
};

/** @brief Authored reusable function source supplied to the native Material Graph compiler. */
struct material_function_source
{
    std::string path;
    std::string source;
};

/** @brief One normalized authored material graph node. */
struct material_ir_node
{
    std::string id;
    material_ir_node_kind kind{material_ir_node_kind::constant};
    material_ir_literal literal;
    float strength{1.0f};
    float minimum{};
    float maximum{1.0f};
    bool exposed_parameter{};
    shader_parameter_id parameter_id{};
    std::string parameter_name;

    // Populated only for shader-backed Material Function calls. Graph-backed
    // functions are deterministically inlined before native IR is emitted.
    std::string function_path;
    std::string function_entry_point;
    std::string function_source;
    std::vector<material_function_pin> function_inputs;
    std::vector<material_function_pin> function_outputs;

    friend auto operator<=>(const material_ir_node&, const material_ir_node&) = default;
};

/** @brief One normalized directed connection between material graph pins. */
struct material_ir_connection
{
    std::string source_node;
    std::string source_pin;
    std::string target_node;
    std::string target_pin;

    friend auto operator<=>(const material_ir_connection&, const material_ir_connection&) = default;
};

/** @brief Backend-neutral material graph representation used by shader code-generation stages. */
struct material_ir
{
    std::uint32_t version{material_ir_version};
    std::vector<material_ir_node> nodes;
    std::vector<material_ir_connection> connections;
    std::string output_node_id;
};

/** @brief Stable surface output semantic exposed by material graph v1. */
enum class material_surface_output : std::uint8_t
{
    base_color,
    metallic,
    roughness,
    normal,
    ambient_occlusion,
    emissive,
    opacity,
    alpha_cutoff
};

/** @brief Connection selected for one material surface output. */
struct material_surface_output_binding
{
    material_surface_output output{material_surface_output::base_color};
    bool connected{};
    std::string source_node;
    std::string source_pin;
};

/** @brief Deterministic texture slot assigned to one reachable texture-sample node. */
struct material_texture_binding
{
    std::string node_id;
    std::uint32_t slot{};
    shader_parameter_id parameter_id{};
    std::string parameter_name;
};

/** @brief Runtime/input capabilities required by a compiled material graph. */
struct material_feature_requirements
{
    bool uses_time{};
    bool uses_uv0{};
    bool uses_texture_sampling{};
    bool uses_normal_mapping{};
};

/** @brief Backend-neutral descriptor emitted alongside native material IR. */
struct material_graph_descriptor
{
    std::uint32_t material_abi{arc::render::material_abi_version};
    std::vector<shader_parameter_descriptor> parameters;
    std::vector<material_texture_binding> textures;
    std::vector<material_surface_output_binding> outputs;
    material_feature_requirements requirements;
};

/** @brief Native material graph compilation output prior to shader code generation. */
struct material_graph_compilation
{
    material_ir ir;
    material_graph_descriptor descriptor;
    std::vector<shader_diagnostic> diagnostics;
};

/** @brief Deterministic generated Slang source implementing the Material ABI v1 evaluator. */
struct material_shader_source
{
    std::string source;
    std::unordered_map<std::uint32_t, std::string> generated_line_nodes;
    std::vector<shader_parameter_descriptor> parameters;
    std::vector<shader_diagnostic> diagnostics;
};

using material_graph_compile_result = core::result<material_graph_compilation, shader_compile_error>;
using material_shader_codegen_result = core::result<material_shader_source, shader_compile_error>;
using material_function_validation_result = core::result<material_function_descriptor, shader_compile_error>;

/**
 * @brief Validate and normalize authored material graph JSON into native IR and descriptor data.
 *
 * This overload accepts graphs that do not reference reusable function assets.
 */
[[nodiscard]] material_graph_compile_result compile_material_graph_json(std::string_view graph_json);

/**
 * @brief Compile a material graph with a deterministic library of reusable Material/Shader Functions.
 *
 * Graph-backed functions are recursively inlined before Material IR is emitted. Shader-backed
 * functions remain typed `function_call` IR nodes and are namespaced by code generation. Missing
 * functions, recursive dependencies, invalid boundary pins, and incomplete required inputs fail
 * native compilation rather than being deferred to the renderer.
 */
[[nodiscard]] material_graph_compile_result
compile_material_graph_json(std::string_view graph_json, std::span<const material_function_source> functions);

/** @brief Validate one first-class Material/Shader Function document and return its public descriptor. */
[[nodiscard]] material_function_validation_result validate_material_function_json(std::string_view function_json,
                                                                                   std::string_view source_path = {});

/** @brief Return true when the authored JSON document identifies itself as a reusable material function. */
[[nodiscard]] bool is_material_function_json(std::string_view source) noexcept;

/**
 * @brief Deterministically generate Slang from validated native Material IR.
 *
 * The generated source implements `arc_evaluate_material(ArcSurfaceInput)` against Material ABI v1.
 * Shader-backed reusable functions are emitted once in stable namespaces before the evaluator.
 */
[[nodiscard]] material_shader_codegen_result generate_material_slang(const material_graph_compilation& compilation);

} // namespace arc::render::tools
