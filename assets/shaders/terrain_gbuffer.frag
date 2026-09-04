#version 450
#extension GL_GOOGLE_include_directive : require

#include "include/arc_texture_sampling.glsl"

layout(location = 0) in vec3 in_normal;
layout(location = 1) in vec3 in_world_position;
layout(location = 2) in vec4 in_weights;
layout(location = 3) in vec2 in_texcoord;
layout(location = 4) in vec4 in_tangent;
layout(location = 5) in vec4 in_clip_position;
layout(location = 6) in vec4 in_previous_clip_position;

layout(location = 0) out vec4 out_albedo;
layout(location = 1) out vec4 out_normal;
layout(location = 2) out vec4 out_material;
layout(location = 3) out vec4 out_emissive;
layout(location = 4) out vec2 out_motion;
layout(location = 5) out uint out_object_id;

layout(set = 0, binding = 0) uniform sampler2D grass_texture;
layout(set = 0, binding = 1) uniform sampler2D dirt_texture;
layout(set = 0, binding = 2) uniform sampler2D rock_texture;
layout(set = 0, binding = 3) uniform sampler2D sand_texture;
layout(set = 0, binding = 7) uniform sampler2D grass_normal_texture;
layout(set = 0, binding = 8) uniform sampler2D dirt_normal_texture;
layout(set = 0, binding = 9) uniform sampler2D rock_normal_texture;
layout(set = 0, binding = 10) uniform sampler2D sand_normal_texture;
layout(set = 0, binding = 11) uniform sampler2D grass_surface_texture;
layout(set = 0, binding = 12) uniform sampler2D dirt_surface_texture;
layout(set = 0, binding = 13) uniform sampler2D rock_surface_texture;
layout(set = 0, binding = 14) uniform sampler2D sand_surface_texture;

layout(push_constant) uniform mesh_constants {
    mat4 model_view_projection; mat4 model; vec4 base_color; vec4 light_direction_intensity;
    vec4 light_color; vec4 camera_position; vec4 visualization; vec4 fog_color_density;
    vec4 fog_params; vec4 material_params;
} constants;

bool has_layer(float flag) { return mod(floor(constants.light_color.w / flag), 2.0) >= 1.0; }
bool has_normal(float flag) { return mod(floor(constants.visualization.y / flag), 2.0) >= 1.0; }
bool has_surface(float flag) { return mod(floor(constants.visualization.z / flag), 2.0) >= 1.0; }
float hash21(vec2 value) { return fract(sin(dot(value, vec2(127.1, 311.7))) * 43758.5453); }
vec3 antitile(sampler2D source, vec2 world_uv, float scale, uint texture_metadata_index)
{
    vec2 uv = world_uv / max(scale, 0.01);
    vec2 macro_cell = floor(world_uv / 18.0);
    vec2 offset = vec2(hash21(macro_cell), hash21(macro_cell + 19.7));
    vec3 primary = arc_sample_texture_2d(source, uv, texture_metadata_index).rgb;
    vec3 secondary = arc_sample_texture_2d(source, uv * 0.873 + offset * 7.0, texture_metadata_index).rgb;
    float variation = hash21(floor(world_uv * 0.12)) * 0.12 - 0.06;
    return mix(primary, secondary, 0.38) * (1.0 + variation);
}
vec3 layer_sample(int layer)
{
    vec2 world_uv = in_world_position.xz;
    if (layer == 0) return has_layer(1.0) ? antitile(grass_texture, world_uv, constants.material_params.x, 0u) : vec3(0.19, 0.30, 0.10);
    if (layer == 1) return has_layer(2.0) ? antitile(dirt_texture, world_uv, constants.material_params.y, 1u) : vec3(0.27, 0.19, 0.11);
    if (layer == 3) return has_layer(8.0) ? antitile(sand_texture, world_uv, constants.material_params.w, 3u) : vec3(0.52, 0.43, 0.27);
    if (!has_layer(4.0)) return vec3(0.23, 0.24, 0.22);
    vec3 n = abs(normalize(in_normal));
    n = pow(n, vec3(5.0)); n /= max(n.x + n.y + n.z, 0.0001);
    float scale = max(constants.material_params.z, 0.01);
    vec3 x = antitile(rock_texture, in_world_position.zy, scale, 2u);
    vec3 y = antitile(rock_texture, in_world_position.xz, scale, 2u);
    vec3 z = antitile(rock_texture, in_world_position.xy, scale, 2u);
    return x * n.x + y * n.y + z * n.z;
}
vec3 surface_sample(int layer)
{
    vec2 uv = in_world_position.xz;
    if (layer == 0 && has_surface(1.0)) return arc_sample_texture_2d(grass_surface_texture, uv / max(constants.material_params.x, 0.01), 8u).rgb;
    if (layer == 1 && has_surface(2.0)) return arc_sample_texture_2d(dirt_surface_texture, uv / max(constants.material_params.y, 0.01), 9u).rgb;
    if (layer == 2 && has_surface(4.0)) return arc_sample_texture_2d(rock_surface_texture, uv / max(constants.material_params.z, 0.01), 10u).rgb;
    if (layer == 3 && has_surface(8.0)) return arc_sample_texture_2d(sand_surface_texture, uv / max(constants.material_params.w, 0.01), 11u).rgb;
    return vec3(1.0, layer == 0 ? 0.82 : layer == 1 ? 0.88 : layer == 2 ? 0.72 : 0.91, 0.5);
}
vec3 layer_normal(int layer, vec3 geometric_normal)
{
    float flag = exp2(float(layer));
    if (!has_normal(flag)) return geometric_normal;
    vec2 scale = vec2(layer == 0 ? constants.material_params.x : layer == 1 ? constants.material_params.y :
        layer == 2 ? constants.material_params.z : constants.material_params.w);
    vec2 uv = in_world_position.xz / max(scale, vec2(0.01));
    vec3 mapped = layer == 0 ? arc_sample_texture_2d(grass_normal_texture, uv, 4u).xyz :
        layer == 1 ? arc_sample_texture_2d(dirt_normal_texture, uv, 5u).xyz :
        layer == 2 ? arc_sample_texture_2d(rock_normal_texture, uv, 6u).xyz :
        arc_sample_texture_2d(sand_normal_texture, uv, 7u).xyz;
    mapped = mapped * 2.0 - 1.0;
    vec3 tangent = normalize(vec3(1.0, 0.0, 0.0) - geometric_normal * geometric_normal.x);
    vec3 bitangent = normalize(cross(tangent, geometric_normal));
    return normalize(mat3(tangent, bitangent, geometric_normal) * mapped);
}
void main()
{
    vec4 weights = max(in_weights, vec4(0.0)); weights /= max(dot(weights, vec4(1.0)), 0.0001);
    vec3 surfaces[4] = vec3[4](surface_sample(0), surface_sample(1), surface_sample(2), surface_sample(3));
    vec4 height_weights = weights + vec4(surfaces[0].b, surfaces[1].b, surfaces[2].b, surfaces[3].b) * 0.16;
    float strongest = max(max(height_weights.x, height_weights.y), max(height_weights.z, height_weights.w));
    weights = max(height_weights - vec4(strongest - 0.12), vec4(0.0));
    weights /= max(dot(weights, vec4(1.0)), 0.0001);
    vec3 albedo = layer_sample(0) * weights.x + layer_sample(1) * weights.y +
        layer_sample(2) * weights.z + layer_sample(3) * weights.w;
    vec3 geometric_normal = normalize(in_normal);
    vec3 normal = normalize(layer_normal(0, geometric_normal) * weights.x +
        layer_normal(1, geometric_normal) * weights.y + layer_normal(2, geometric_normal) * weights.z +
        layer_normal(3, geometric_normal) * weights.w);
    float roughness = dot(weights, vec4(surfaces[0].g, surfaces[1].g, surfaces[2].g, surfaces[3].g));
    float ao = dot(weights, vec4(surfaces[0].r, surfaces[1].r, surfaces[2].r, surfaces[3].r));
    vec2 current_ndc = in_clip_position.xy / max(in_clip_position.w, 0.00001);
    vec2 previous_ndc = in_previous_clip_position.xy / max(in_previous_clip_position.w, 0.00001);
    out_albedo = vec4(albedo * constants.base_color.rgb, 1.0);
    out_normal = vec4(normal * 0.5 + 0.5, clamp(ao, 0.0, 1.0));
    out_material = vec4(0.0, roughness, 1.0, 0.0);
    out_emissive = vec4(0.0);
    out_motion = (current_ndc - previous_ndc) * 0.5;
    out_object_id = uint(constants.fog_params.w);
}
