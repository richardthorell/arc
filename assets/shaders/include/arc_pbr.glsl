#ifndef ARC_PBR_GLSL
#define ARC_PBR_GLSL

#include "arc_math.glsl"

const float ARC_DIELECTRIC_F0 = 0.04;

struct arc_surface_data
{
    vec3 base_color;
    vec3 normal;
    vec3 clear_coat_normal;
    vec3 tangent;
    vec3 emissive;
    float metallic;
    float perceptual_roughness;
    float occlusion;
    float clear_coat;
    float clear_coat_roughness;
    float anisotropy;
};

struct arc_brdf_result
{
    vec3 diffuse;
    vec3 specular;
};

float arc_saturate(float value)
{
    return clamp(value, 0.0, 1.0);
}

vec3 arc_fresnel_schlick(float cos_theta, vec3 f0)
{
    float factor = pow(1.0 - arc_saturate(cos_theta), 5.0);
    return f0 + (vec3(1.0) - f0) * factor;
}

vec3 arc_fresnel_schlick_roughness(float cos_theta, vec3 f0, float roughness)
{
    return f0 + (max(vec3(1.0 - roughness), f0) - f0) *
        pow(1.0 - arc_saturate(cos_theta), 5.0);
}

float arc_distribution_ggx(float n_dot_h, float alpha)
{
    float alpha_squared = alpha * alpha;
    float denominator = n_dot_h * n_dot_h * (alpha_squared - 1.0) + 1.0;
    return alpha_squared / max(ARC_PI * denominator * denominator, 1.0e-7);
}

float arc_smith_ggx_correlated(float n_dot_v, float n_dot_l, float alpha)
{
    float alpha_squared = alpha * alpha;
    float lambda_v = n_dot_l * sqrt(max((-n_dot_v * alpha_squared + n_dot_v) * n_dot_v + alpha_squared, 0.0));
    float lambda_l = n_dot_v * sqrt(max((-n_dot_l * alpha_squared + n_dot_l) * n_dot_l + alpha_squared, 0.0));
    return 0.5 / max(lambda_v + lambda_l, 1.0e-6);
}

float arc_distribution_ggx_anisotropic(vec3 half_vector, vec3 normal, vec3 tangent, vec3 bitangent, float alpha_x, float alpha_y)
{
    float t_dot_h = dot(tangent, half_vector);
    float b_dot_h = dot(bitangent, half_vector);
    float n_dot_h = dot(normal, half_vector);
    float denominator = t_dot_h * t_dot_h / (alpha_x * alpha_x) +
        b_dot_h * b_dot_h / (alpha_y * alpha_y) + n_dot_h * n_dot_h;
    return 1.0 / max(ARC_PI * alpha_x * alpha_y * denominator * denominator, 1.0e-7);
}

vec3 arc_multiscatter_compensation(vec3 f0, float roughness)
{
    float energy_loss = 1.0 - (0.28 * roughness);
    return vec3(1.0) + f0 * (1.0 / max(energy_loss, 0.5) - 1.0);
}

arc_brdf_result arc_evaluate_brdf(arc_surface_data surface, vec3 view_direction, vec3 light_direction)
{
    arc_brdf_result result;
    result.diffuse = vec3(0.0);
    result.specular = vec3(0.0);

    vec3 half_vector = normalize(view_direction + light_direction);
    float n_dot_v = max(dot(surface.normal, view_direction), 1.0e-5);
    float n_dot_l = max(dot(surface.normal, light_direction), 0.0);
    float n_dot_h = max(dot(surface.normal, half_vector), 0.0);
    float v_dot_h = max(dot(view_direction, half_vector), 0.0);
    if (n_dot_l <= 0.0)
        return result;

    float roughness = clamp(surface.perceptual_roughness, 0.04, 1.0);
    float alpha = roughness * roughness;
    vec3 f0 = mix(vec3(ARC_DIELECTRIC_F0), surface.base_color, surface.metallic);
    vec3 fresnel = arc_fresnel_schlick(v_dot_h, f0);
    float distribution;
    if (abs(surface.anisotropy) > 1.0e-4 && dot(surface.tangent, surface.tangent) > 0.5)
    {
        vec3 tangent = normalize(surface.tangent - surface.normal * dot(surface.normal, surface.tangent));
        vec3 bitangent = normalize(cross(surface.normal, tangent));
        float aspect = sqrt(max(1.0 - 0.9 * abs(surface.anisotropy), 0.1));
        float alpha_x = max(alpha / aspect, 0.002);
        float alpha_y = max(alpha * aspect, 0.002);
        if (surface.anisotropy < 0.0)
        {
            float swap_value = alpha_x;
            alpha_x = alpha_y;
            alpha_y = swap_value;
        }
        distribution = arc_distribution_ggx_anisotropic(
            half_vector, surface.normal, tangent, bitangent, alpha_x, alpha_y);
    }
    else
    {
        distribution = arc_distribution_ggx(n_dot_h, alpha);
    }
    float visibility = arc_smith_ggx_correlated(n_dot_v, n_dot_l, alpha);
    result.specular = distribution * visibility * fresnel *
        arc_multiscatter_compensation(f0, roughness);

    vec3 diffuse_weight = (vec3(1.0) - fresnel) * (1.0 - surface.metallic);
    result.diffuse = diffuse_weight * surface.base_color / ARC_PI;

    if (surface.clear_coat > 0.0)
    {
        vec3 coat_normal = dot(surface.clear_coat_normal, surface.clear_coat_normal) > 0.5
            ? normalize(surface.clear_coat_normal)
            : surface.normal;
        float coat_n_dot_v = max(dot(coat_normal, view_direction), 1.0e-5);
        float coat_n_dot_l = max(dot(coat_normal, light_direction), 0.0);
        float coat_n_dot_h = max(dot(coat_normal, half_vector), 0.0);
        float coat_alpha = max(surface.clear_coat_roughness * surface.clear_coat_roughness, 0.002);
        float coat_fresnel = ARC_DIELECTRIC_F0 +
            (1.0 - ARC_DIELECTRIC_F0) * pow(1.0 - v_dot_h, 5.0);
        float coat = coat_n_dot_l > 0.0
            ? arc_distribution_ggx(coat_n_dot_h, coat_alpha) *
                arc_smith_ggx_correlated(coat_n_dot_v, coat_n_dot_l, coat_alpha) *
                coat_fresnel * surface.clear_coat
            : 0.0;
        float base_energy = 1.0 - surface.clear_coat * coat_fresnel;
        result.diffuse *= base_energy;
        result.specular = result.specular * base_energy + vec3(coat);
    }
    return result;
}

vec3 arc_evaluate_split_sum_ibl(
    arc_surface_data surface,
    vec3 view_direction,
    vec3 diffuse_irradiance,
    vec3 prefiltered_radiance,
    vec2 integrated_brdf)
{
    float n_dot_v = max(dot(surface.normal, view_direction), 0.0);
    vec3 f0 = mix(vec3(ARC_DIELECTRIC_F0), surface.base_color, surface.metallic);
    vec3 fresnel = arc_fresnel_schlick_roughness(
        n_dot_v, f0, surface.perceptual_roughness);
    vec3 diffuse = diffuse_irradiance * surface.base_color *
        (vec3(1.0) - fresnel) * (1.0 - surface.metallic);
    vec3 specular = prefiltered_radiance *
        (fresnel * integrated_brdf.x + integrated_brdf.y) *
        arc_multiscatter_compensation(f0, surface.perceptual_roughness);
    return (diffuse + specular) * surface.occlusion;
}

vec3 arc_beer_lambert(vec3 attenuation_color, float attenuation_distance, float distance_inside)
{
    if (attenuation_distance <= 0.0)
        return vec3(1.0);
    vec3 coefficient = -log(clamp(attenuation_color, vec3(1.0e-4), vec3(1.0))) /
        attenuation_distance;
    return exp(-coefficient * max(distance_inside, 0.0));
}

#endif
