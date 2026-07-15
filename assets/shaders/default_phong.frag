#version 450
#extension GL_GOOGLE_include_directive : require

#include "include/arc_math.glsl"

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

float saturate(float value)
{
    return clamp(value, 0.0, 1.0);
}

vec3 aces_filmic(vec3 color)
{
    color *= 1.08;
    return clamp((color * (2.51 * color + 0.03)) / (color * (2.43 * color + 0.59) + 0.14), 0.0, 1.0);
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

float distribution_ggx(float n_dot_h, float roughness)
{
    float a = roughness * roughness;
    float a2 = a * a;
    float denom = n_dot_h * n_dot_h * (a2 - 1.0) + 1.0;
    return a2 / max(ARC_PI * denom * denom, 0.00001);
}

float geometry_schlick_ggx(float n_dot_v, float roughness)
{
    float r = roughness + 1.0;
    float k = (r * r) / 8.0;
    return n_dot_v / max(n_dot_v * (1.0 - k) + k, 0.00001);
}

vec3 fresnel_schlick(float cos_theta, vec3 f0)
{
    return f0 + (1.0 - f0) * pow(1.0 - cos_theta, 5.0);
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
    vec3 half_dir = normalize(light_dir + view_dir);

    vec4 mr = has_texture(2.0) ? texture(metallic_roughness_texture, in_texcoord) : vec4(1.0);
    float roughness = clamp(constants.visualization.z * mr.g, 0.04, 1.0);
    float metallic = clamp(constants.visualization.y * mr.b, 0.0, 1.0);
    float ao = has_texture(8.0)
        ? mix(1.0, texture(occlusion_texture, in_texcoord).r, constants.material_params.y)
        : 1.0;
    vec3 emissive = has_texture(16.0)
        ? texture(emissive_texture, in_texcoord).rgb * constants.material_params.z
        : vec3(0.0);

    float n_dot_l = saturate(dot(normal, light_dir));
    float n_dot_v = saturate(dot(normal, view_dir));
    float n_dot_h = saturate(dot(normal, half_dir));
    float h_dot_v = saturate(dot(half_dir, view_dir));

    vec3 f0 = mix(vec3(0.04), material_color.rgb, metallic);
    vec3 f = fresnel_schlick(h_dot_v, f0);
    float d = distribution_ggx(n_dot_h, roughness);
    float g = geometry_schlick_ggx(n_dot_l, roughness) * geometry_schlick_ggx(n_dot_v, roughness);
    vec3 specular = (d * g * f) / max(4.0 * n_dot_v * n_dot_l, 0.0001);
    vec3 diffuse = (1.0 - f) * (1.0 - metallic) * material_color.rgb / ARC_PI;
    vec3 radiance = constants.light_color.rgb * constants.light_direction_intensity.w;
    float shadow = sample_shadow(in_world_position);
    vec3 direct = (diffuse + specular) * radiance * n_dot_l * shadow;
    vec3 ambient = material_color.rgb * ao * 0.18;
    vec3 lit_color = ambient + direct + emissive;

    int mode = int(constants.visualization.x + 0.5);
    vec3 color = lit_color;
    if (mode == 1)
        color = material_color.rgb;
    else if (mode == 2)
        color = vec3(material_color.a);
    else if (mode == 3)
        color = normal * 0.5 + vec3(0.5);
    else if (mode == 4)
        color = f0;
    else if (mode == 5)
        color = vec3(1.0 - roughness);
    else if (mode == 6)
        color = vec3(metallic);
    else if (mode == 7)
        color = vec3(ao);
    else if (mode == 8)
        color = emissive;
    else if (mode == 9)
        color = vec3(n_dot_l * shadow);
    else if (mode == 10)
        color = vec3(in_texcoord, 0.0);
    else
        color = apply_height_fog(color);

    if (mode == 0)
    {
        color *= max(constants.camera_position.w, 0.001);
        color = aces_filmic(color);
    }
    out_color = vec4(color, material_color.a);
}
