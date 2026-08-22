#include <arc/render/material_pass.h>

#include <algorithm>
#include <cstdint>
#include <type_traits>

namespace arc::render
{
namespace
{

constexpr std::uint64_t fnv_offset_basis = 14695981039346656037ull;
constexpr std::uint64_t fnv_prime = 1099511628211ull;

void hash_byte(std::uint64_t& hash, std::uint8_t value) noexcept
{
    hash ^= value;
    hash *= fnv_prime;
}

template <class T> void hash_integral(std::uint64_t& hash, T value) noexcept
{
    using unsigned_type = std::make_unsigned_t<T>;
    auto bits = static_cast<std::uint64_t>(static_cast<unsigned_type>(value));
    for (std::size_t index = 0; index < sizeof(unsigned_type); ++index)
    {
        hash_byte(hash, static_cast<std::uint8_t>(bits & 0xffu));
        bits >>= 8u;
    }
}

void hash_material_features(std::uint64_t& hash, const shader_permutation_key& key) noexcept
{
    hash_integral(hash, static_cast<std::uint8_t>(key.alpha_mode));
    hash_integral(hash, key.debug_view);
    hash_integral(hash, static_cast<std::uint8_t>(key.has_base_color_texture));
    hash_integral(hash, static_cast<std::uint8_t>(key.has_metallic_roughness_texture));
    hash_integral(hash, static_cast<std::uint8_t>(key.has_normal_texture));
    hash_integral(hash, static_cast<std::uint8_t>(key.has_occlusion_texture));
    hash_integral(hash, static_cast<std::uint8_t>(key.has_emissive_texture));
    hash_integral(hash, static_cast<std::uint8_t>(key.has_clear_coat_texture));
    hash_integral(hash, static_cast<std::uint8_t>(key.has_clear_coat_roughness_texture));
    hash_integral(hash, static_cast<std::uint8_t>(key.has_clear_coat_normal_texture));
    hash_integral(hash, static_cast<std::uint8_t>(key.has_anisotropy_texture));
    hash_integral(hash, static_cast<std::uint8_t>(key.has_subsurface_texture));
    hash_integral(hash, static_cast<std::uint8_t>(key.has_thickness_texture));
    hash_integral(hash, static_cast<std::uint8_t>(key.has_transmission_texture));
    hash_integral(hash, static_cast<std::uint8_t>(key.double_sided));
    hash_integral(hash, static_cast<std::uint8_t>(key.wireframe));
    hash_integral(hash, static_cast<std::uint8_t>(key.clear_coat));
    hash_integral(hash, static_cast<std::uint8_t>(key.sheen));
    hash_integral(hash, static_cast<std::uint8_t>(key.transmission));
    hash_integral(hash, static_cast<std::uint8_t>(key.subsurface));
    hash_integral(hash, static_cast<std::uint8_t>(key.anisotropy));
    hash_integral(hash, static_cast<std::uint8_t>(key.parallax));
}

} // namespace

bool material_supports_pass(const material_descriptor& material, material_pass pass) noexcept
{
    if (material.domain != material_domain::surface) return false;

    switch (pass)
    {
        case material_pass::depth:
        case material_pass::shadow:
        case material_pass::motion:
            return material.alpha_mode != material_alpha_mode::blend;
        case material_pass::gbuffer:
            return material.alpha_mode != material_alpha_mode::blend &&
                   resolve_material_render_path(material) == material_render_path::deferred;
        case material_pass::forward:
            return true;
        case material_pass::object_id:
        case material_pass::selection:
            return true;
        case material_pass::ray_hit:
            return false;
    }
    return false;
}

bool material_pass_evaluates_surface(material_pass pass, material_alpha_mode alpha_mode) noexcept
{
    switch (pass)
    {
        case material_pass::gbuffer:
        case material_pass::forward:
            return true;
        case material_pass::depth:
        case material_pass::shadow:
        case material_pass::motion:
            return alpha_mode == material_alpha_mode::masked;
        case material_pass::object_id:
        case material_pass::selection:
        case material_pass::ray_hit:
            return false;
    }
    return false;
}

material_pass_permutation_key make_material_pass_permutation_key(const material_descriptor& material,
                                                                 material_pass pass, std::uint8_t debug_view,
                                                                 bool wireframe) noexcept
{
    return {.pass = pass,
            .render_path = resolve_material_render_path(material),
            .shading_model = material.shading_model,
            .material = make_shader_permutation_key(material, debug_view, wireframe),
            .evaluates_material = material_pass_evaluates_surface(pass, material.alpha_mode),
            .writes_motion = pass == material_pass::motion || pass == material_pass::gbuffer};
}

std::uint64_t hash_material_pass_permutation_key(const material_pass_permutation_key& key) noexcept
{
    std::uint64_t hash = fnv_offset_basis;
    hash_integral(hash, key.contract_version);
    hash_integral(hash, key.material_abi);
    hash_integral(hash, static_cast<std::uint8_t>(key.pass));
    hash_integral(hash, static_cast<std::uint8_t>(key.render_path));
    hash_integral(hash, static_cast<std::uint8_t>(key.shading_model));
    hash_material_features(hash, key.material);
    hash_integral(hash, static_cast<std::uint8_t>(key.evaluates_material));
    hash_integral(hash, static_cast<std::uint8_t>(key.writes_motion));
    return hash;
}

shader_permutation_id make_material_pass_permutation_id(const material_pass_permutation_key& key) noexcept
{
    const auto hash = hash_material_pass_permutation_key(key);
    return {hash == 0 ? std::uint64_t{1} : hash};
}

const material_pass_binding* find_material_pass_binding(const material_compiled_program& program,
                                                        material_pass pass) noexcept
{
    const auto found = std::ranges::find(program.passes, pass, &material_pass_binding::pass);
    return found == program.passes.end() ? nullptr : &*found;
}

material_pipeline_resolution resolve_material_pipeline(const material_descriptor& material, material_pass pass,
                                                       const material_compiled_program* compiled) noexcept
{
    const bool compiled_contract_valid = compiled != nullptr &&
                                         compiled->contract_version == material_pass_contract_version &&
                                         compiled->material_abi == material_abi_version && compiled->package.valid() &&
                                         find_material_pass_binding(*compiled, pass) != nullptr;

    switch (material.pipeline)
    {
        case material_pipeline::legacy:
            return {.use_legacy = true};
        case material_pipeline::compiled:
            return {.use_legacy = !compiled_contract_valid, .use_compiled = compiled_contract_valid};
        case material_pipeline::compare:
            return {.use_legacy = true, .use_compiled = compiled_contract_valid, .compare = compiled_contract_valid};
    }
    return {.use_legacy = true};
}

} // namespace arc::render
