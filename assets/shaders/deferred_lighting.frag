#version 450

layout(location = 0) in vec2 in_uv;
layout(location = 0) out vec4 out_color;

layout(set = 0, binding = 0) uniform sampler2D gbuffer_albedo;
layout(set = 0, binding = 1) uniform sampler2D gbuffer_normal;
layout(set = 0, binding = 2) uniform sampler2D gbuffer_material;
layout(set = 0, binding = 3) uniform usampler2D gbuffer_object_id;

layout(push_constant) uniform deferred_constants
{
    vec4 light_direction_intensity;
    vec4 light_color_exposure;
    vec4 ambient_visualization;
} constants;

const float PI = 3.14159265359;

float saturate(float value)
{
    return clamp(value, 0.0, 1.0);
}

vec3 aces_filmic(vec3 color)
{
    color *= 1.08;
    return clamp((color * (2.51 * color + 0.03)) / (color * (2.43 * color + 0.59) + 0.14), 0.0, 1.0);
}

vec3 fresnel_schlick(float cos_theta, vec3 f0)
{
    return f0 + (1.0 - f0) * pow(1.0 - cos_theta, 5.0);
}

void main()
{
    vec4 albedo = texture(gbuffer_albedo, in_uv);
    vec4 normal_ao = texture(gbuffer_normal, in_uv);
    vec4 material = texture(gbuffer_material, in_uv);
    uint object_id = texture(gbuffer_object_id, in_uv).r;
    if (object_id == 0u && albedo.a <= 0.0)
        discard;

    vec3 normal = normalize(normal_ao.xyz * 2.0 - vec3(1.0));
    float ao = normal_ao.w;
    float metallic = saturate(material.x);
    float roughness = clamp(material.y, 0.04, 1.0);
    float emissive_strength = max(material.z, 0.0);
    int mode = int(constants.ambient_visualization.w + 0.5);

    vec3 light_dir = normalize(-constants.light_direction_intensity.xyz);
    vec3 view_dir = vec3(0.0, 0.0, 1.0);
    vec3 half_dir = normalize(light_dir + view_dir);
    float n_dot_l = saturate(dot(normal, light_dir));
    float h_dot_v = saturate(dot(half_dir, view_dir));
    vec3 f0 = mix(vec3(0.04), albedo.rgb, metallic);
    vec3 f = fresnel_schlick(h_dot_v, f0);
    vec3 diffuse = (1.0 - f) * (1.0 - metallic) * albedo.rgb / PI;
    vec3 specular = f * (1.0 - roughness);
    vec3 radiance = constants.light_color_exposure.rgb * constants.light_direction_intensity.w;
    vec3 ambient = albedo.rgb * constants.ambient_visualization.rgb * ao;
    vec3 lit = ambient + (diffuse + specular) * radiance * n_dot_l + albedo.rgb * emissive_strength;

    vec3 color = lit;
    if (mode == 1)
        color = albedo.rgb;
    else if (mode == 2)
        color = vec3(albedo.a);
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
        color = albedo.rgb * emissive_strength;
    else if (mode == 9)
        color = vec3(n_dot_l);
    else if (mode == 10)
        color = vec3(in_uv, 0.0);
    else
        color = aces_filmic(color * max(constants.light_color_exposure.w, 0.001));

    out_color = vec4(color, 1.0);
}
