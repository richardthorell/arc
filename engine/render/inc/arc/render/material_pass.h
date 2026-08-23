#pragma once

/**
 * @file arc/render/material_pass.h
 * @brief Backend-neutral material pass routing, permutations, and compiled program bindings.
 */

#include <arc/render/material.h>
#include <arc/render/material_abi.h>

#include <cstddef>
#include <cstdint>
#include <optional>
#include <vector>

namespace arc::render
{

/** @brief Version of the backend-neutral material/pass composition contract. */
inline constexpr std::uint32_t material_pass_contract_version = 1;

/**
 * @brief Complete backend-neutral identity of one material render-pass permutation.
 *
 * Backends may attach API-specific pipeline state to this key, but the key itself deliberately
 * contains no Vulkan, D3D12, or Metal concepts.
 */
struct material_pass_permutation_key
{
    std::uint32_t contract_version{material_pass_contract_version};
    std::uint32_t material_abi{material_abi_version};
    material_pass pass{material_pass::forward};
    material_render_path render_path{material_render_path::deferred};
    material_shading_model shading_model{material_shading_model::standard};
    shader_permutation_key material;
    bool evaluates_material{};
    bool writes_motion{};

    friend bool operator==(const material_pass_permutation_key&,
                           const material_pass_permutation_key&) noexcept = default;
};

/** @brief One compiled shader permutation that implements a material render pass. */
struct material_pass_binding
{
    material_pass pass{material_pass::forward};
    shader_permutation_id permutation{};
    shader_entry_point_id entry_point{};
    shader_content_hash build_hash{};

    friend bool operator==(const material_pass_binding&, const material_pass_binding&) noexcept = default;
};

/**
 * @brief Runtime binding table for one compiled material implementation.
 *
 * Graph-generated and handwritten Material Shaders produce the same contract so the renderer does not depend on the
 * implementation source.
 */
struct material_compiled_program
{
    std::uint32_t contract_version{material_pass_contract_version};
    std::uint32_t material_abi{material_abi_version};
    shader_package_id package{};
    std::vector<material_pass_binding> passes;
};

/** @brief Return whether a material is eligible to participate in the requested render pass. */
[[nodiscard]] bool material_supports_pass(const material_descriptor& material, material_pass pass) noexcept;

/** @brief Return whether this pass needs to execute the Material ABI evaluator. */
[[nodiscard]] bool material_pass_evaluates_surface(material_pass pass, material_alpha_mode alpha_mode) noexcept;

/** @brief Build the complete backend-neutral permutation key for one material/pass combination. */
[[nodiscard]] material_pass_permutation_key make_material_pass_permutation_key(const material_descriptor& material,
                                                                               material_pass pass,
                                                                               std::uint8_t debug_view = 0,
                                                                               bool wireframe = false) noexcept;

/** @brief Return a stable cross-process hash for a material-pass permutation key. */
[[nodiscard]] std::uint64_t hash_material_pass_permutation_key(const material_pass_permutation_key& key) noexcept;

/** @brief Return the stable shader permutation ID associated with a material-pass key. */
[[nodiscard]] shader_permutation_id
make_material_pass_permutation_id(const material_pass_permutation_key& key) noexcept;

/** @brief Find the compiled binding for one pass, if the material implementation provides it. */
[[nodiscard]] const material_pass_binding* find_material_pass_binding(const material_compiled_program& program,
                                                                      material_pass pass) noexcept;

/** @brief Return whether a cooked compiled material program is valid for the requested pass. */
[[nodiscard]] inline bool material_program_supports_pass(const material_compiled_program& program,
                                                         material_pass pass) noexcept
{
    return program.contract_version == material_pass_contract_version && program.material_abi == material_abi_version &&
           program.package.valid() && find_material_pass_binding(program, pass) != nullptr;
}

} // namespace arc::render
