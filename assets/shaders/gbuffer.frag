#version 450

layout(location = 0) in vec3 in_normal;
layout(location = 1) in vec3 in_world_position;
layout(location = 2) in vec4 in_color;
layout(location = 3) in vec2 in_texcoord;
layout(location = 4) in vec4 in_tangent;
layout(location = 5) in vec4 in_clip_position;
layout(location = 6) in vec4 in_previous_clip_position;

layout(location = 0) out vec4 out_albedo;
layout(location = 1) out vec4 out_normal;
layout(location = 2) out vec4 out_material;
layout(location = 3) out vec2 out_motion;
layout(location = 4) out uint out_object_id;

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
} shadows;

bool has_texture(float flag)
{
    return mod(floor(constants.light_color.w / flag), 2.0) >= 1.0;
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

void main()
{
    vec4 sampled_base = has_texture(1.0) ? texture(base_texture, in_texcoord) : vec4(1.0);
    vec4 material_color = sampled_base * in_color;
    int alpha_mode = int(constants.material_params.w + 0.5);
    if (alpha_mode == 1 && material_color.a < constants.visualization.w)
        discard;

    vec4 mr = has_texture(2.0) ? texture(metallic_roughness_texture, in_texcoord) : vec4(1.0);
    float roughness = clamp(constants.visualization.z * mr.g, 0.04, 1.0);
    float metallic = clamp(constants.visualization.y * mr.b, 0.0, 1.0);
    float ao = has_texture(8.0)
        ? mix(1.0, texture(occlusion_texture, in_texcoord).r, constants.material_params.y)
        : 1.0;
    vec3 emissive = has_texture(16.0)
        ? texture(emissive_texture, in_texcoord).rgb * constants.material_params.z
        : vec3(0.0);

    vec2 current_ndc = in_clip_position.xy / max(in_clip_position.w, 0.00001);
    vec2 previous_ndc = in_previous_clip_position.xy / max(in_previous_clip_position.w, 0.00001);

    out_albedo = vec4(material_color.rgb, material_color.a);
    out_normal = vec4(material_normal() * 0.5 + vec3(0.5), ao);
    out_material = vec4(metallic, roughness, length(emissive), constants.visualization.x);
    out_motion = (current_ndc - previous_ndc) * 0.5;
    out_object_id = uint(constants.fog_params.w);
}
