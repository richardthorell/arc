#include <arc/render/material.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <functional>
#include <unordered_set>

namespace arc::render
{
namespace
{

bool parameter_type_matches(shader_parameter_type expected, const material_parameter_value& value) noexcept
{
    switch (expected)
    {
        case shader_parameter_type::boolean:
            return std::holds_alternative<bool>(value);
        case shader_parameter_type::int32:
            return std::holds_alternative<std::int32_t>(value);
        case shader_parameter_type::uint32:
            return std::holds_alternative<std::uint32_t>(value);
        case shader_parameter_type::float32:
            return std::holds_alternative<float>(value);
        case shader_parameter_type::float2:
            return std::holds_alternative<math::vector2f>(value);
        case shader_parameter_type::float3:
            return std::holds_alternative<math::vector3f>(value);
        case shader_parameter_type::float4:
            return std::holds_alternative<math::vector4f>(value);
        case shader_parameter_type::matrix4x4:
            return std::holds_alternative<math::matrix4x4f>(value);
        case shader_parameter_type::texture_2d:
        case shader_parameter_type::texture_cube:
        case shader_parameter_type::sampler:
            return std::holds_alternative<resource_handle>(value);
    }
    return false;
}

template <typename T>
bool write_runtime_parameter(std::vector<std::byte>& block, const shader_parameter_descriptor& parameter,
                             const T* values, std::size_t value_count) noexcept
{
    const auto byte_count = sizeof(T) * value_count;
    const auto offset = static_cast<std::size_t>(parameter.offset);
    if (offset > block.size() || byte_count > static_cast<std::size_t>(parameter.size) ||
        byte_count > block.size() - offset)
        return false;

    std::memcpy(block.data() + offset, values, byte_count);
    return true;
}

bool apply_runtime_parameter_override(material_runtime_program& program,
                                      const material_parameter_override& override_value) noexcept
{
    const auto parameter = std::ranges::find(program.parameters, override_value.id, &shader_parameter_descriptor::id);
    if (parameter == program.parameters.end()) return false;

    switch (parameter->type)
    {
        case shader_parameter_type::boolean:
        {
            const std::uint32_t value = std::get<bool>(override_value.value) ? 1u : 0u;
            return write_runtime_parameter(program.parameter_defaults, *parameter, &value, 1u);
        }
        case shader_parameter_type::int32:
        {
            const auto value = std::get<std::int32_t>(override_value.value);
            return write_runtime_parameter(program.parameter_defaults, *parameter, &value, 1u);
        }
        case shader_parameter_type::uint32:
        {
            const auto value = std::get<std::uint32_t>(override_value.value);
            return write_runtime_parameter(program.parameter_defaults, *parameter, &value, 1u);
        }
        case shader_parameter_type::float32:
        {
            const auto value = std::get<float>(override_value.value);
            return write_runtime_parameter(program.parameter_defaults, *parameter, &value, 1u);
        }
        case shader_parameter_type::float2:
        {
            const auto& value = std::get<math::vector2f>(override_value.value);
            const std::array values{value[0], value[1]};
            return write_runtime_parameter(program.parameter_defaults, *parameter, values.data(), values.size());
        }
        case shader_parameter_type::float3:
        {
            const auto& value = std::get<math::vector3f>(override_value.value);
            const std::array values{value[0], value[1], value[2]};
            return write_runtime_parameter(program.parameter_defaults, *parameter, values.data(), values.size());
        }
        case shader_parameter_type::float4:
        {
            const auto& value = std::get<math::vector4f>(override_value.value);
            const std::array values{value[0], value[1], value[2], value[3]};
            return write_runtime_parameter(program.parameter_defaults, *parameter, values.data(), values.size());
        }
        case shader_parameter_type::matrix4x4:
        {
            const auto& value = std::get<math::matrix4x4f>(override_value.value);
            return write_runtime_parameter(program.parameter_defaults, *parameter, value.data(), 16u);
        }
        case shader_parameter_type::texture_2d:
        case shader_parameter_type::texture_cube:
        case shader_parameter_type::sampler:
            return false;
    }
    return false;
}

void apply_runtime_parameter_overrides(material_descriptor& material,
                                       std::span<const material_parameter_override> overrides)
{
    if (!material.runtime_program || overrides.empty()) return;

    auto runtime_program = std::make_shared<material_runtime_program>(*material.runtime_program);
    if (runtime_program->parameter_defaults.size() < runtime_program->parameter_block_size)
        runtime_program->parameter_defaults.resize(runtime_program->parameter_block_size);

    bool changed = false;
    for (const auto& override_value : overrides)
        changed |= apply_runtime_parameter_override(*runtime_program, override_value);

    if (changed) material.runtime_program = std::move(runtime_program);
}

} // namespace

material_render_path resolve_material_render_path(const material_descriptor& material) noexcept
{
    if (!material.deferred_compatible || material.shading_model != material_shading_model::standard ||
        material.alpha_mode == material_alpha_mode::blend || material.clear_coat_factor > 0.0f ||
        material.sheen_factor > 0.0f || material.transmission_factor > 0.0f || material.subsurface_factor > 0.0f ||
        material.anisotropy_factor != 0.0f || material.parallax_height_scale != 0.0f ||
        material.displacement_mode != material_displacement_mode::none)
        return material_render_path::clustered_forward;
    return material_render_path::deferred;
}

material_instance_result resolve_material_instance(const material_definition_descriptor& definition,
                                                   const material_instance_descriptor& instance)
{
    if (!instance.parent.valid())
        return material_instance_result::failure(
            {.code = material_instance_error_code::invalid_parent, .message = "material instance parent is invalid"});

    std::unordered_set<std::uint64_t> seen;
    material_descriptor result = definition.material;
    if (!instance.name.empty()) result.name = instance.name;
    for (const auto& override_value : instance.overrides)
    {
        if (!override_value.id.valid())
            return material_instance_result::failure(
                {.code = material_instance_error_code::unknown_parameter,
                 .parameter = override_value.id,
                 .message = "material instance override has an invalid stable ID"});
        if (!seen.insert(override_value.id.value).second)
            return material_instance_result::failure(
                {.code = material_instance_error_code::duplicate_override,
                 .parameter = override_value.id,
                 .message = "material instance contains a duplicate parameter override"});

        const auto layout =
            std::ranges::find(definition.parameter_layout, override_value.id, &shader_parameter_descriptor::id);
        if (layout == definition.parameter_layout.end())
            return material_instance_result::failure(
                {.code = material_instance_error_code::unknown_parameter,
                 .parameter = override_value.id,
                 .message = "material instance references a parameter absent from its parent"});
        if (!parameter_type_matches(layout->type, override_value.value))
            return material_instance_result::failure(
                {.code = material_instance_error_code::incompatible_type,
                 .parameter = override_value.id,
                 .message = "material instance override type does not match its parent layout"});

        const auto existing = std::ranges::find(result.parameters, override_value.id, &material_parameter_override::id);
        if (existing == result.parameters.end())
            result.parameters.push_back(override_value);
        else
            *existing = override_value;
    }
    apply_runtime_parameter_overrides(result, instance.overrides);
    result.render_path = resolve_material_render_path(result);
    return material_instance_result::success(std::move(result));
}

shader_permutation_key make_shader_permutation_key(const material_descriptor& material, std::uint8_t debug_view,
                                                   bool wireframe) noexcept
{
    return {.alpha_mode = material.alpha_mode,
            .debug_view = debug_view,
            .has_base_color_texture = material.base_color_texture.valid(),
            .has_metallic_roughness_texture = material.metallic_roughness_texture.valid(),
            .has_normal_texture = material.normal_texture.valid(),
            .has_occlusion_texture = material.occlusion_texture.valid(),
            .has_emissive_texture = material.emissive_texture.valid(),
            .has_clear_coat_texture = material.clear_coat_texture.valid(),
            .has_clear_coat_roughness_texture = material.clear_coat_roughness_texture.valid(),
            .has_clear_coat_normal_texture = material.clear_coat_normal_texture.valid(),
            .has_anisotropy_texture = material.anisotropy_texture.valid(),
            .has_subsurface_texture = material.subsurface_texture.valid(),
            .has_thickness_texture = material.thickness_texture.valid(),
            .has_transmission_texture = material.transmission_texture.valid(),
            .double_sided = material.double_sided,
            .wireframe = wireframe,
            .clear_coat = material.clear_coat_factor > 0.0f,
            .sheen = material.sheen_factor > 0.0f,
            .transmission = material.transmission_factor > 0.0f,
            .subsurface = material.subsurface_factor > 0.0f,
            .anisotropy = material.anisotropy_factor != 0.0f,
            .parallax = material.parallax_height_scale != 0.0f ||
                        material.displacement_mode != material_displacement_mode::none};
}

std::size_t hash_shader_permutation_key(const shader_permutation_key& key) noexcept
{
    auto combine = [](std::size_t seed, std::size_t value)
    { return seed ^ (value + 0x9e3779b97f4a7c15ull + (seed << 6u) + (seed >> 2u)); };

    std::size_t seed = 0;
    seed = combine(seed, std::hash<std::uint8_t>{}(static_cast<std::uint8_t>(key.alpha_mode)));
    seed = combine(seed, std::hash<std::uint8_t>{}(key.debug_view));
    seed = combine(seed, std::hash<bool>{}(key.has_base_color_texture));
    seed = combine(seed, std::hash<bool>{}(key.has_metallic_roughness_texture));
    seed = combine(seed, std::hash<bool>{}(key.has_normal_texture));
    seed = combine(seed, std::hash<bool>{}(key.has_occlusion_texture));
    seed = combine(seed, std::hash<bool>{}(key.has_emissive_texture));
    seed = combine(seed, std::hash<bool>{}(key.has_clear_coat_texture));
    seed = combine(seed, std::hash<bool>{}(key.has_clear_coat_roughness_texture));
    seed = combine(seed, std::hash<bool>{}(key.has_clear_coat_normal_texture));
    seed = combine(seed, std::hash<bool>{}(key.has_anisotropy_texture));
    seed = combine(seed, std::hash<bool>{}(key.has_subsurface_texture));
    seed = combine(seed, std::hash<bool>{}(key.has_thickness_texture));
    seed = combine(seed, std::hash<bool>{}(key.has_transmission_texture));
    seed = combine(seed, std::hash<bool>{}(key.double_sided));
    seed = combine(seed, std::hash<bool>{}(key.wireframe));
    seed = combine(seed, std::hash<bool>{}(key.clear_coat));
    seed = combine(seed, std::hash<bool>{}(key.sheen));
    seed = combine(seed, std::hash<bool>{}(key.transmission));
    seed = combine(seed, std::hash<bool>{}(key.subsurface));
    seed = combine(seed, std::hash<bool>{}(key.anisotropy));
    seed = combine(seed, std::hash<bool>{}(key.parallax));
    return seed;
}

float srgb_to_linear(float value) noexcept
{
    value = std::max(value, 0.0f);
    return value <= 0.04045f ? value / 12.92f : std::pow((value + 0.055f) / 1.055f, 2.4f);
}

float linear_to_srgb(float value) noexcept
{
    value = std::max(value, 0.0f);
    return value <= 0.0031308f ? value * 12.92f : 1.055f * std::pow(value, 1.0f / 2.4f) - 0.055f;
}

math::vector3f srgb_to_linear(const math::vector3f& value) noexcept
{
    return {srgb_to_linear(value[0]), srgb_to_linear(value[1]), srgb_to_linear(value[2])};
}

math::vector3f linear_to_srgb(const math::vector3f& value) noexcept
{
    return {linear_to_srgb(value[0]), linear_to_srgb(value[1]), linear_to_srgb(value[2])};
}

float ggx_distribution(float n_dot_h, float roughness) noexcept
{
    const float alpha = std::max(roughness * roughness, 0.001f);
    const float alpha_squared = alpha * alpha;
    const float denominator = n_dot_h * n_dot_h * (alpha_squared - 1.0f) + 1.0f;
    return alpha_squared / std::max(3.14159265358979323846f * denominator * denominator, 1.0e-6f);
}

float smith_ggx_correlated(float n_dot_v, float n_dot_l, float roughness) noexcept
{
    n_dot_v = std::clamp(n_dot_v, 0.0f, 1.0f);
    n_dot_l = std::clamp(n_dot_l, 0.0f, 1.0f);
    const float alpha = std::max(roughness * roughness, 0.001f);
    const float alpha_squared = alpha * alpha;
    const float lambda_v =
        n_dot_l * std::sqrt(std::max(n_dot_v * n_dot_v * (1.0f - alpha_squared) + alpha_squared, 0.0f));
    const float lambda_l =
        n_dot_v * std::sqrt(std::max(n_dot_l * n_dot_l * (1.0f - alpha_squared) + alpha_squared, 0.0f));
    return 0.5f / std::max(lambda_v + lambda_l, 1.0e-6f);
}

math::vector3f fresnel_schlick(float cos_theta, const math::vector3f& f0) noexcept
{
    const float factor = std::pow(1.0f - std::clamp(cos_theta, 0.0f, 1.0f), 5.0f);
    return {f0[0] + (1.0f - f0[0]) * factor, f0[1] + (1.0f - f0[1]) * factor, f0[2] + (1.0f - f0[2]) * factor};
}

math::vector3f beer_lambert_attenuation(const math::vector3f& attenuation_color, float attenuation_distance,
                                        float thickness) noexcept
{
    if (!std::isfinite(attenuation_distance) || attenuation_distance <= 0.0f) return math::vector3f::one;

    const float path = std::max(thickness, 0.0f) / attenuation_distance;
    return {std::pow(std::clamp(attenuation_color[0], 1.0e-6f, 1.0f), path),
            std::pow(std::clamp(attenuation_color[1], 1.0e-6f, 1.0f), path),
            std::pow(std::clamp(attenuation_color[2], 1.0e-6f, 1.0f), path)};
}

} // namespace arc::render