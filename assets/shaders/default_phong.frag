#version 450
#extension GL_GOOGLE_include_directive : require

#include "include/arc_pbr.glsl"
#include "include/arc_material_parameters.glsl"
#define ARC_LIGHT_BUFFER_BINDING 15
#include "include/arc_lighting.glsl"

layout(location = 0) in vec3 in_normal;
layout(location = 1) in vec3 in_world_position;
layout(location = 2) in vec4 in_color;
layout(location = 3) in vec2 in_texcoord;
layout(location = 4) in float in_view_depth;
layout(location = 5) in vec4 in_tangent;

layout(location = 0) out vec4 out_color;

layout(set = 0, binding = 0) uniform sampler2D base_texture;
layout(set = 0, binding = 1) uniform sampler2D metallic_roughness_texture;
layout(set = 0, binding = 2) uniform sampler2D normal_texture;
layout(set = 0, binding = 3) uniform sampler2D occlusion_texture;
layout(set = 0, binding = 4) uniform sampler2D emissive_texture;
layout(set = 0, binding = 5) uniform sampler2DArrayShadow shadow_map;
layout(set = 0, binding = 7) uniform sampler2D clear_coat_texture;
layout(set = 0, binding = 8) uniform sampler2D clear_coat_roughness_texture;
layout(set = 0, binding = 9) uniform sampler2D clear_coat_normal_texture;
layout(set = 0, binding = 10) uniform sampler2D anisotropy_texture;
layout(set = 0, binding = 11) uniform sampler2D subsurface_texture;
layout(set = 0, binding = 12) uniform sampler2D thickness_texture;
layout(set = 0, binding = 13) uniform sampler2D transmission_texture;

layout(push_constant) uniform mesh_constants
{
    mat4 model_view_projection;
    mat4 model;
    vec4 base_color;
    vec4 light_direction_intensity;
    vec4 light_color;
    vec4 camera_position;
    vec4 visualization;
    vec4 fog_color_density;
    vec4 fog_params;
    vec4 material_params;
} constants;

layout(set = 0, binding = 6) uniform shadow_data
{
    mat4 light_view_projection[4];
    vec4 cascade_splits;
    vec4 params;
    vec4 cascade_texel_size;
} shadows;

bool has_texture(float flag)
{
    return mod(floor(constants.light_color.w / flag), 2.0) >= 1.0;
}

bool has_advanced_texture(float flag)
{
    return mod(floor(material_parameters.attenuation_color.w / flag), 2.0) >= 1.0;
}

float sample_shadow(vec3 world_position)
{
    if (shadows.params.x <= 0.0)
        return 1.0;

    int cascade = 0;
    vec2 uv = vec2(0.0);
    float depth = 1.0;
    bool covered = false;
    for (int candidate = 0; candidate < 4; ++candidate)
    {
        vec4 shadow_position = shadows.light_view_projection[candidate] * vec4(world_position, 1.0);
        vec3 projected = shadow_position.xyz / shadow_position.w;
        vec2 candidate_uv = projected.xy * 0.5 + vec2(0.5);
        if (candidate_uv.x >= 0.0 && candidate_uv.y >= 0.0 &&
            candidate_uv.x <= 1.0 && candidate_uv.y <= 1.0 &&
            projected.z >= 0.0 && projected.z <= 1.0)
        {
            cascade = candidate;
            uv = candidate_uv;
            depth = projected.z;
            covered = true;
            break;
        }
    }
    if (!covered)
        return 1.0;

    int filter_mode = int(shadows.params.w + 0.5);
    float radius = filter_mode == 2 ? 2.0 : 1.0;
    if (filter_mode == 0)
        radius = 0.0;
    else if (filter_mode == 3)
        radius = mix(1.5, 4.0, clamp(depth, 0.0, 1.0));
    float bias = shadows.params.y + shadows.params.z * clamp(1.0 - dot(normalize(in_normal), normalize(-constants.light_direction_intensity.xyz)), 0.0, 1.0);
    vec2 texel = vec2(1.0 / float(textureSize(shadow_map, 0).x));
    float visibility = 0.0;
    float samples = 0.0;
    for (float y = -radius; y <= radius; y += 1.0)
    {
        for (float x = -radius; x <= radius; x += 1.0)
        {
            visibility += texture(shadow_map, vec4(uv + vec2(x, y) * texel, float(cascade), depth - bias));
            samples += 1.0;
        }
    }
    visibility = samples > 0.0 ? visibility / samples : texture(shadow_map, vec4(uv, float(cascade), depth - bias));
    return mix(1.0 - shadows.params.x, 1.0, visibility);
}

vec3 apply_height_fog(vec3 color)
{
    float density = constants.fog_color_density.w;
    if (density <= 0.0)
        return color;

    float distance_from_camera = length(constants.camera_position.xyz - in_world_position);
    float start_distance = max(constants.fog_params.x, 0.0);
    float height_falloff = max(constants.fog_params.y, 0.0);
    float max_opacity = clamp(constants.fog_params.z, 0.0, 1.0);
    float sun_scattering = max(constants.fog_params.w, 0.0);

    float distance_term = max(distance_from_camera - start_distance, 0.0) * density;
    float height_term = exp(-max(in_world_position.y, 0.0) * height_falloff);
    float fog_amount = clamp(1.0 - exp(-distance_term * height_term), 0.0, max_opacity);

    vec3 fog_color = constants.fog_color_density.rgb;
    vec3 light_dir = normalize(-constants.light_direction_intensity.xyz);
    vec3 view_dir = normalize(constants.camera_position.xyz - in_world_position);
    float sun_term = pow(max(dot(view_dir, light_dir), 0.0), 8.0) * sun_scattering;
    fog_color += constants.light_color.rgb * sun_term;
    return mix(color, fog_color, fog_amount);
}

vec3 material_normal()
{
    vec3 n = normalize(in_normal);
    if (!has_texture(4.0))
        return n;

    vec3 t = normalize(in_tangent.xyz);
    t = normalize(t - n * dot(n, t));
    vec3 b = normalize(cross(n, t) * in_tangent.w);
    vec3 mapped = texture(normal_texture, in_texcoord).xyz * 2.0 - vec3(1.0);
    mapped.xy *= constants.material_params.x;
    return normalize(mat3(t, b, n) * mapped);
}

vec3 material_clear_coat_normal(vec3 fallback)
{
    if (!has_advanced_texture(4.0))
        return fallback;
    vec3 t = normalize(in_tangent.xyz);
    t = normalize(t - fallback * dot(fallback, t));
    vec3 b = normalize(cross(fallback, t) * in_tangent.w);
    vec3 mapped = texture(clear_coat_normal_texture, in_texcoord).xyz * 2.0 - vec3(1.0);
    return normalize(mat3(t, b, fallback) * mapped);
}

void main()
{
    vec4 sampled_base = has_texture(1.0) ? texture(base_texture, in_texcoord) : vec4(1.0);
    vec4 material_color = sampled_base * in_color;
    int alpha_mode = int(constants.material_params.w + 0.5);
    if (alpha_mode == 1 && material_color.a < constants.visualization.w)
        discard;

    vec3 normal = material_normal();
    vec3 light_dir = normalize(-constants.light_direction_intensity.xyz);
    vec3 view_dir = normalize(constants.camera_position.xyz - in_world_position);
    float key_n_dot_l = max(dot(normal, light_dir), 0.0);

    vec4 mr = has_texture(2.0) ? texture(metallic_roughness_texture, in_texcoord) : vec4(1.0);
    float roughness = clamp(constants.visualization.z * mr.g, 0.04, 1.0);
    float metallic = clamp(constants.visualization.y * mr.b, 0.0, 1.0);
    float ao = has_texture(8.0)
        ? mix(1.0, texture(occlusion_texture, in_texcoord).r, constants.material_params.y)
        : 1.0;
    vec3 emissive = has_texture(16.0)
        ? texture(emissive_texture, in_texcoord).rgb *
            material_parameters.emissive_factor.rgb * material_parameters.emissive_factor.w
        : material_parameters.emissive_factor.rgb * material_parameters.emissive_factor.w;

    arc_surface_data surface;
    surface.base_color = material_color.rgb;
    surface.normal = normal;
    surface.clear_coat_normal = material_clear_coat_normal(normal);
    surface.tangent = in_tangent.xyz;
    surface.emissive = emissive;
    surface.metallic = metallic;
    surface.perceptual_roughness = roughness;
    surface.occlusion = ao;
    surface.clear_coat = material_parameters.material_lobes.x *
        (has_advanced_texture(1.0) ? texture(clear_coat_texture, in_texcoord).r : 1.0);
    surface.clear_coat_roughness = material_parameters.material_lobes.y *
        (has_advanced_texture(2.0) ? texture(clear_coat_roughness_texture, in_texcoord).g : 1.0);
    surface.anisotropy = material_parameters.material_lobes.z *
        (has_advanced_texture(8.0) ? texture(anisotropy_texture, in_texcoord).b : 1.0);
    float shadow = sample_shadow(in_world_position);
    vec3 radiance = constants.light_color.rgb * constants.light_direction_intensity.w;
    vec3 direct = arc_evaluate_scene_lights(surface, view_dir, in_world_position, shadow);
    vec3 ambient = arc_evaluate_split_sum_ibl(
        surface,
        view_dir,
        vec3(0.18),
        vec3(0.18) * mix(0.35, 1.0, 1.0 - roughness),
        vec2(1.0 - 0.5 * roughness, 0.04));
    int shading_model = int(material_parameters.volume_params.x + 0.5);
    float subsurface_factor = material_parameters.subsurface_color_factor.w *
        (has_advanced_texture(16.0) ? texture(subsurface_texture, in_texcoord).r : 1.0);
    if (shading_model == 1 && subsurface_factor > 0.0)
    {
        float wrapped = clamp((dot(normal, light_dir) + 0.45) / 1.45, 0.0, 1.0);
        float back_scatter = pow(clamp(dot(-normal, light_dir), 0.0, 1.0), 2.0);
        direct += material_parameters.subsurface_color_factor.rgb *
            subsurface_factor * radiance *
            (wrapped * 0.22 + back_scatter * 0.18) * shadow;
    }
    vec3 lit_color = ambient + direct + emissive;
    float transmission_factor = material_parameters.material_lobes.w *
        (has_advanced_texture(64.0) ? texture(transmission_texture, in_texcoord).r : 1.0);
    float thickness = material_parameters.volume_params.z *
        (has_advanced_texture(32.0) ? texture(thickness_texture, in_texcoord).r : 1.0);
    if (shading_model == 2 && transmission_factor > 0.0)
    {
        float ior = max(material_parameters.volume_params.y, 1.0001);
        float fresnel = pow((ior - 1.0) / (ior + 1.0), 2.0);
        vec3 transmitted_environment = mix(
            constants.fog_color_density.rgb,
            constants.light_color.rgb * 0.18,
            clamp(refract(-view_dir, normal, 1.0 / ior).y * 0.5 + 0.5, 0.0, 1.0));
        transmitted_environment *= arc_beer_lambert(
            material_parameters.attenuation_color.rgb,
            material_parameters.volume_params.w,
            thickness);
        float transmission_weight = transmission_factor * (1.0 - fresnel);
        lit_color = mix(lit_color, transmitted_environment + emissive, transmission_weight);
    }

    int mode = int(constants.visualization.x + 0.5);
    vec3 color = lit_color;
    if (mode == 1)
        color = material_color.rgb;
    else if (mode == 2)
        color = vec3(material_color.a);
    else if (mode == 3)
        color = normal * 0.5 + vec3(0.5);
    else if (mode == 4)
        color = mix(vec3(0.04), material_color.rgb, metallic);
    else if (mode == 5)
        color = vec3(1.0 - roughness);
    else if (mode == 6)
        color = vec3(metallic);
    else if (mode == 7)
        color = vec3(ao);
    else if (mode == 8)
        color = emissive;
    else if (mode == 9)
        color = vec3(key_n_dot_l * shadow);
    else if (mode == 10)
        color = vec3(in_texcoord, 0.0);
    else
        color = apply_height_fog(color);

    out_color = vec4(color, material_color.a);
}
