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
    vec4 cascade_texel_size;
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

vec2 sample_shadow(vec3 world_position, vec3 normal)
{
    if (shadows.params.x <= 0.0)
        return vec2(1.0, 0.0);

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
        return vec2(1.0, 3.0);

    vec3 light_dir = normalize(-constants.light_direction_intensity.xyz);
    float normal_bias = shadows.params.z * clamp(1.0 - dot(normal, light_dir), 0.0, 1.0);
    float compare_depth = depth - shadows.params.y - normal_bias;
    int filter_mode = int(shadows.params.w + 0.5);
    if (filter_mode == 0)
        return vec2(texture(shadow_map, vec4(uv, float(cascade), compare_depth)), float(cascade));

    float radius = filter_mode == 2 ? 2.0 : 1.0;
    if (filter_mode == 3)
        radius = mix(1.5, 4.0, clamp(depth, 0.0, 1.0));

    vec2 texel = vec2(1.0 / float(textureSize(shadow_map, 0).x));
    float visibility = 0.0;
    float samples = 0.0;
    for (float y = -radius; y <= radius; y += 1.0)
    {
        for (float x = -radius; x <= radius; x += 1.0)
        {
            visibility += texture(shadow_map, vec4(uv + vec2(x, y) * texel, float(cascade), compare_depth));
            samples += 1.0;
        }
    }
    visibility = samples > 0.0 ? visibility / samples : 1.0;
    return vec2(mix(1.0 - shadows.params.x, 1.0, visibility), float(cascade));
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

    vec3 normal = material_normal();
    vec2 shadow = sample_shadow(in_world_position, normal);

    out_albedo = vec4(material_color.rgb, material_color.a);
    out_normal = vec4(normal * 0.5 + vec3(0.5), ao);
    out_material = vec4(metallic, roughness, length(emissive), shadow.x);
    out_motion = vec2(shadow.y / 3.0, shadow.x);
    out_object_id = uint(constants.fog_params.w);
}
