#version 450
#extension GL_GOOGLE_include_directive : require

#include "include/arc_pbr.glsl"
#define ARC_LIGHT_BUFFER_BINDING 15
#include "include/arc_lighting.glsl"

layout(location = 0) in vec3 in_normal;
layout(location = 1) in vec3 in_world_position;
layout(location = 2) in vec4 in_weights;
layout(location = 3) in vec2 in_texcoord;
layout(location = 4) in float in_view_depth;
layout(location = 5) in vec4 in_tangent;
layout(location = 0) out vec4 out_color;
layout(set = 0, binding = 0) uniform sampler2D grass_texture;
layout(set = 0, binding = 1) uniform sampler2D dirt_texture;
layout(set = 0, binding = 2) uniform sampler2D rock_texture;
layout(set = 0, binding = 3) uniform sampler2D sand_texture;
layout(push_constant) uniform mesh_constants {
    mat4 model_view_projection; mat4 model; vec4 base_color; vec4 light_direction_intensity;
    vec4 light_color; vec4 camera_position; vec4 visualization; vec4 fog_color_density;
    vec4 fog_params; vec4 material_params; vec4 emissive_factor; vec4 material_lobes;
    vec4 volume_params; vec4 subsurface_color_factor; vec4 attenuation_color;
} constants;
bool has_layer(float flag) { return mod(floor(constants.light_color.w / flag), 2.0) >= 1.0; }
vec3 sample_layer(sampler2D source, vec2 uv, float scale, vec3 fallback, bool ready)
{
    if (!ready) return fallback;
    vec2 mapped = uv / max(scale, 0.01);
    return mix(texture(source, mapped).rgb, texture(source, mapped * 0.873 + vec2(3.7, 8.1)).rgb, 0.38);
}
void main()
{
    vec4 weights = max(in_weights, vec4(0.0)); weights /= max(dot(weights, vec4(1.0)), 0.0001);
    vec3 color = sample_layer(grass_texture, in_world_position.xz, constants.material_params.x, vec3(0.19,0.30,0.10), has_layer(1.0)) * weights.x;
    color += sample_layer(dirt_texture, in_world_position.xz, constants.material_params.y, vec3(0.27,0.19,0.11), has_layer(2.0)) * weights.y;
    color += sample_layer(rock_texture, in_world_position.xz, constants.material_params.z, vec3(0.23,0.24,0.22), has_layer(4.0)) * weights.z;
    color += sample_layer(sand_texture, in_world_position.xz, constants.material_params.w, vec3(0.52,0.43,0.27), has_layer(8.0)) * weights.w;
    vec3 normal = normalize(in_normal);
    vec3 light_dir = normalize(-constants.light_direction_intensity.xyz);
    vec3 view_dir = normalize(constants.camera_position.xyz - in_world_position);
    arc_surface_data surface;
    surface.base_color = color;
    surface.normal = normal;
    surface.clear_coat_normal = normal;
    surface.tangent = in_tangent.xyz;
    surface.emissive = vec3(0.0);
    surface.metallic = 0.0;
    surface.perceptual_roughness = 0.82;
    surface.occlusion = 1.0;
    surface.clear_coat = 0.0;
    surface.clear_coat_roughness = 0.0;
    surface.anisotropy = 0.0;
    vec3 ambient = arc_evaluate_split_sum_ibl(
        surface,
        view_dir,
        vec3(0.16, 0.19, 0.22),
        vec3(0.16, 0.19, 0.22) * mix(0.35, 1.0, 1.0 - surface.perceptual_roughness),
        vec2(1.0 - 0.5 * surface.perceptual_roughness, 0.04));
    vec3 lit = ambient + arc_evaluate_scene_lights(surface, view_dir, in_world_position, 1.0);
    float fog = constants.fog_color_density.w > 0.0 ? clamp(1.0 - exp(-max(in_view_depth - constants.fog_params.x, 0.0) * constants.fog_color_density.w), 0.0, constants.fog_params.z) : 0.0;
    out_color = vec4(mix(lit, constants.fog_color_density.rgb, fog), 1.0);
}
