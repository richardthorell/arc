#include <arc/editor/material_preview.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <numbers>
#include <utility>

namespace arc::editor
{
namespace
{

using color3 = std::array<float, 3>;

constexpr float preview_sphere_radius = 0.82f;
constexpr float minimum_roughness = 0.045f;
constexpr float dielectric_reflectance = 0.04f;
constexpr float preview_exposure = 1.15f;

color3 add(color3 a, color3 b) noexcept { return { a[0] + b[0], a[1] + b[1], a[2] + b[2] }; }
color3 mul(color3 a, color3 b) noexcept { return { a[0] * b[0], a[1] * b[1], a[2] * b[2] }; }
color3 mul(color3 value, float scale) noexcept { return { value[0] * scale, value[1] * scale, value[2] * scale }; }
float dot(color3 a, color3 b) noexcept { return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]; }
color3 cross(color3 a, color3 b) noexcept
{
    return { a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0] };
}
color3 normalize(color3 value) noexcept
{
    const float length = std::sqrt(std::max(dot(value, value), 1.0e-12f));
    return mul(value, 1.0f / length);
}
color3 mix(color3 a, color3 b, float amount) noexcept
{
    return add(mul(a, 1.0f - amount), mul(b, amount));
}

float linear_from_srgb(float value) noexcept
{
    value = std::clamp(value, 0.0f, 1.0f);
    return value <= 0.04045f ? value / 12.92f : std::pow((value + 0.055f) / 1.055f, 2.4f);
}

float srgb_from_linear(float value) noexcept
{
    value = std::max(value, 0.0f);
    return value <= 0.0031308f ? value * 12.92f : 1.055f * std::pow(value, 1.0f / 2.4f) - 0.055f;
}

float aces_tonemap(float value) noexcept
{
    value *= preview_exposure;
    return std::clamp((value * (2.51f * value + 0.03f)) / (value * (2.43f * value + 0.59f) + 0.14f), 0.0f, 1.0f);
}

struct sampled_texture
{
    render::texture_data data;
    bool valid{};
};

sampled_texture load_preview_texture(const std::filesystem::path& root, const std::string& path)
{
    if (path.empty())
        return {};
    const auto result = render::load_texture_asset(resolve_material_texture_path(root, path));
    if (!result.succeeded() || !result.texture.has_pixels())
        return {};
    const bool supported = result.texture.format == render::texture_format::rgba8_unorm ||
        result.texture.format == render::texture_format::rgba8_srgb ||
        result.texture.format == render::texture_format::rgba32f;
    return { result.texture, supported };
}

std::array<float, 4> sample(const sampled_texture& texture, float u, float v, bool color_data)
{
    if (!texture.valid)
        return { 1.0f, 1.0f, 1.0f, 1.0f };
    const auto& data = texture.data;
    u -= std::floor(u);
    v = std::clamp(v, 0.0f, 1.0f);
    const auto x = std::min(data.width - 1u, static_cast<std::uint32_t>(u * data.width));
    const auto y = std::min(data.height - 1u, static_cast<std::uint32_t>((1.0f - v) * data.height));
    const auto pixel = static_cast<std::size_t>(y * data.width + x);
    std::array<float, 4> result{};
    if (data.format == render::texture_format::rgba32f)
        std::memcpy(result.data(), data.pixels.data() + pixel * sizeof(float) * 4u, sizeof(result));
    else
    {
        const auto offset = pixel * 4u;
        for (std::size_t channel = 0; channel < 4u; ++channel)
            result[channel] = static_cast<float>(std::to_integer<std::uint8_t>(data.pixels[offset + channel])) / 255.0f;
    }
    if (color_data && data.format == render::texture_format::rgba8_srgb)
    {
        result[0] = linear_from_srgb(result[0]);
        result[1] = linear_from_srgb(result[1]);
        result[2] = linear_from_srgb(result[2]);
    }
    return result;
}

float distribution_ggx(float n_dot_h, float roughness) noexcept
{
    const float alpha = roughness * roughness;
    const float alpha2 = alpha * alpha;
    const float denominator = n_dot_h * n_dot_h * (alpha2 - 1.0f) + 1.0f;
    return alpha2 / std::max(std::numbers::pi_v<float> * denominator * denominator, 1.0e-5f);
}

float geometry_schlick(float n_dot_direction, float roughness) noexcept
{
    const float k = ((roughness + 1.0f) * (roughness + 1.0f)) * 0.125f;
    return n_dot_direction / std::max(n_dot_direction * (1.0f - k) + k, 1.0e-5f);
}

color3 fresnel_schlick(float cos_theta, color3 f0) noexcept
{
    const float factor = std::pow(std::clamp(1.0f - cos_theta, 0.0f, 1.0f), 5.0f);
    return add(f0, mul({ 1.0f - f0[0], 1.0f - f0[1], 1.0f - f0[2] }, factor));
}

color3 evaluate_light(
    color3 normal,
    color3 view,
    color3 light,
    color3 radiance,
    color3 base_color,
    float metallic,
    float roughness)
{
    const float n_dot_l = std::max(dot(normal, light), 0.0f);
    const float n_dot_v = std::max(dot(normal, view), 0.0f);
    if (n_dot_l <= 0.0f || n_dot_v <= 0.0f)
        return {};
    const auto half_vector = normalize(add(view, light));
    const float n_dot_h = std::max(dot(normal, half_vector), 0.0f);
    const float v_dot_h = std::max(dot(view, half_vector), 0.0f);
    const auto f0 = mix({ dielectric_reflectance, dielectric_reflectance, dielectric_reflectance }, base_color, metallic);
    const auto fresnel = fresnel_schlick(v_dot_h, f0);
    const float distribution = distribution_ggx(n_dot_h, roughness);
    const float geometry = geometry_schlick(n_dot_l, roughness) * geometry_schlick(n_dot_v, roughness);
    const auto specular = mul(fresnel, distribution * geometry / std::max(4.0f * n_dot_l * n_dot_v, 1.0e-5f));
    const auto diffuse_weight = mul({ 1.0f - fresnel[0], 1.0f - fresnel[1], 1.0f - fresnel[2] }, 1.0f - metallic);
    const auto diffuse = mul(mul(diffuse_weight, base_color), 1.0f / std::numbers::pi_v<float>);
    return mul(mul(add(diffuse, specular), radiance), n_dot_l);
}

color3 preview_background(float x, float y) noexcept
{
    const float radial = std::clamp(1.0f - std::sqrt(x * x + y * y) * 0.55f, 0.0f, 1.0f);
    const float vertical = std::clamp((y + 1.0f) * 0.5f, 0.0f, 1.0f);
    return mix({ 0.018f, 0.024f, 0.029f }, { 0.085f, 0.105f, 0.118f }, radial * (0.45f + vertical * 0.35f));
}

} // namespace

material_preview_result render_material_preview(
    const material_asset& asset,
    const std::filesystem::path& asset_root,
    std::uint32_t size)
{
    size = std::clamp(size, 32u, 256u);
    const auto base_map = load_preview_texture(asset_root, asset.textures.base_color);
    const auto metallic_roughness_map = load_preview_texture(asset_root, asset.textures.metallic_roughness);
    const auto normal_map = load_preview_texture(asset_root, asset.textures.normal);
    const auto ao_map = load_preview_texture(asset_root, asset.textures.ao);
    const auto emissive_map = load_preview_texture(asset_root, asset.textures.emissive);

    render::texture_data output;
    output.name = asset.name + " Preview";
    output.width = size;
    output.height = size;
    output.format = render::texture_format::rgba8_srgb;
    output.pixels.resize(static_cast<std::size_t>(size) * size * 4u);

    const auto& material = asset.material;
    const color3 base_factor{ material.base_color[0], material.base_color[1], material.base_color[2] };
    const color3 emissive_factor{ material.emissive_factor[0], material.emissive_factor[1], material.emissive_factor[2] };
    const color3 view{ 0.0f, 0.0f, 1.0f };
    const auto key_light = normalize(color3{ -0.48f, 0.62f, 0.76f });
    const auto fill_light = normalize(color3{ 0.72f, 0.18f, 0.54f });
    const auto rim_light = normalize(color3{ 0.32f, 0.70f, -0.64f });

    for (std::uint32_t pixel_y = 0; pixel_y < size; ++pixel_y)
    {
        for (std::uint32_t pixel_x = 0; pixel_x < size; ++pixel_x)
        {
            const float x = (2.0f * (static_cast<float>(pixel_x) + 0.5f) / size - 1.0f) / preview_sphere_radius;
            const float y = (1.0f - 2.0f * (static_cast<float>(pixel_y) + 0.5f) / size) / preview_sphere_radius;
            const float radius_squared = x * x + y * y;
            const auto background = preview_background(x, y);
            auto color = background;
            if (radius_squared <= 1.0f)
            {
                auto normal = normalize(color3{ x, y, std::sqrt(std::max(0.0f, 1.0f - radius_squared)) });
                const float u = 0.5f + std::atan2(normal[0], normal[2]) / (2.0f * std::numbers::pi_v<float>);
                const float v = 0.5f + std::asin(std::clamp(normal[1], -1.0f, 1.0f)) / std::numbers::pi_v<float>;
                const auto base_sample = sample(base_map, u, v, true);
                const auto mr_sample = sample(metallic_roughness_map, u, v, false);
                const auto ao_sample = sample(ao_map, u, v, false);
                const auto emissive_sample = sample(emissive_map, u, v, true);
                const auto normal_sample = sample(normal_map, u, v, false);
                if (normal_map.valid)
                {
                    auto tangent = normalize(color3{ normal[2], 0.0f, -normal[0] });
                    const auto bitangent = normalize(cross(normal, tangent));
                    const color3 tangent_normal = normalize({
                        (normal_sample[0] * 2.0f - 1.0f) * material.normal_scale,
                        (normal_sample[1] * 2.0f - 1.0f) * material.normal_scale,
                        std::max(normal_sample[2] * 2.0f - 1.0f, 0.01f)
                    });
                    normal = normalize(add(add(mul(tangent, tangent_normal[0]), mul(bitangent, tangent_normal[1])), mul(normal, tangent_normal[2])));
                }
                const color3 base_color = mul(base_factor, { base_sample[0], base_sample[1], base_sample[2] });
                const float metallic = std::clamp(material.metallic * (metallic_roughness_map.valid ? mr_sample[2] : 1.0f), 0.0f, 1.0f);
                const float roughness = std::clamp(material.roughness * (metallic_roughness_map.valid ? mr_sample[1] : 1.0f), minimum_roughness, 1.0f);
                const float ao = std::clamp(1.0f + ((ao_map.valid ? ao_sample[0] : 1.0f) - 1.0f) * material.occlusion_strength, 0.0f, 1.0f);
                const float base_energy = 1.0f - material.clear_coat_factor * dielectric_reflectance;
                color = mul(mul(base_color, 0.055f * base_energy), ao);
                color = add(color, mul(evaluate_light(normal, view, key_light, { 4.3f, 3.9f, 3.45f }, base_color, metallic, roughness), base_energy));
                color = add(color, mul(evaluate_light(normal, view, fill_light, { 0.75f, 0.95f, 1.25f }, base_color, metallic, roughness), base_energy));
                color = add(color, mul(evaluate_light(normal, view, rim_light, { 0.42f, 0.58f, 0.82f }, base_color, metallic, roughness), base_energy));
                if (material.clear_coat_factor > 0.0f)
                {
                    const auto coat = evaluate_light(
                        normal, view, key_light, { 4.3f, 3.9f, 3.45f },
                        { dielectric_reflectance, dielectric_reflectance, dielectric_reflectance },
                        1.0f, std::clamp(material.clear_coat_roughness, minimum_roughness, 1.0f));
                    color = add(color, mul(coat, material.clear_coat_factor));
                }
                if (material.shading_model == render::material_shading_model::skin &&
                    material.subsurface_factor > 0.0f)
                {
                    const float wrap = std::clamp((dot(normal, key_light) + 0.35f) / 1.35f, 0.0f, 1.0f);
                    const color3 subsurface{
                        material.subsurface_color[0],
                        material.subsurface_color[1],
                        material.subsurface_color[2]
                    };
                    color = add(color, mul(mul(base_color, subsurface), wrap * material.subsurface_factor * 0.28f));
                }
                if (material.transmission_factor > 0.0f)
                {
                    const auto attenuation = render::beer_lambert_attenuation(
                        material.attenuation_color,
                        material.attenuation_distance,
                        std::max(material.thickness_factor, 0.0f));
                    const color3 transmitted = mul(background, { attenuation[0], attenuation[1], attenuation[2] });
                    color = mix(color, transmitted, std::clamp(material.transmission_factor, 0.0f, 1.0f));
                }
                const float emissive_scale = material.emissive_luminance_nits > 0.0f
                    ? material.emissive_luminance_nits / 100.0f
                    : material.emissive_strength;
                color = add(color, mul(mul(emissive_factor, { emissive_sample[0], emissive_sample[1], emissive_sample[2] }), emissive_scale));
                const float opacity = std::clamp(material.base_color[3] * base_sample[3], 0.0f, 1.0f);
                if (material.alpha_mode == render::material_alpha_mode::masked && opacity < material.alpha_cutoff)
                    color = background;
                else if (material.alpha_mode == render::material_alpha_mode::blend)
                    color = mix(background, color, opacity);
            }

            const auto offset = static_cast<std::size_t>(pixel_y * size + pixel_x) * 4u;
            for (std::size_t channel = 0; channel < 3u; ++channel)
                output.pixels[offset + channel] = static_cast<std::byte>(std::clamp(std::lround(srgb_from_linear(aces_tonemap(color[channel])) * 255.0f), 0l, 255l));
            output.pixels[offset + 3u] = std::byte{ 255 };
        }
    }
    return { std::move(output), "rendered PBR material sphere preview" };
}

} // namespace arc::editor
