#version 450
#extension GL_GOOGLE_include_directive : require

#include "include/arc_material_surface.glsl"
#include "include/arc_material_parameters.glsl"
#define ARC_LIGHT_BUFFER_BINDING 15
#include "include/arc_lighting.glsl"
#include "include/arc_shadows.glsl"

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
    int cascade = -1;
    return arc_directional_shadow_visibility(
        world_position,
        normalize(in_normal),
        constants.camera_position.xyz,
        normalize(-constants.light_direction_intensity.xyz),
        cascade);
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

vec3 material_normal(arc_material_inputs input)
{
    vec3 n = normalize(input.normal_ws);
    if (!has_texture(4.0))
        return n;

    vec3 t = normalize(input.tangent_ws.xyz);
    t = normalize(t - n * dot(n, t));
    vec3 b = normalize(cross(n, t) * input.tangent_ws.w);
    vec3 mapped = texture(normal_texture, input.uv0).xyz * 2.0 - vec3(1.0);
    mapped.xy *= constants.material_params.x;
    return normalize(mat3(t, b, n) * mapped);
}

vec3 material_clear_coat_normal(arc_material_inputs input, vec3 fallback)
{
    if (!has_advanced_texture(4.0))
        return fallback;
    vec3 t = normalize(input.tangent_ws.xyz);
    t = normalize(t - fallback * dot(fallback, t));
    vec3 b = normalize(cross(fallback, t) * input.tangent_ws.w);
    vec3 mapped = texture(clear_coat_normal_texture, input.uv0).xyz * 2.0 - vec3(1.0);
    return normalize(mat3(t, b, fallback) * mapped);
}

arc_material_surface arc_evaluate_legacy_material(arc_material_inputs input)
{
    arc_material_surface material = arc_default_material_surface(input);
    vec4 sampled_base = has_texture(1.0) ? texture(base_texture, input.uv0) : vec4(1.0);
    vec4 material_color = sampled_base * input.vertex_color * constants.base_color;
    vec4 mr = has_texture(2.0) ? texture(metallic_roughness_texture, input.uv0) : vec4(1.0);

    material.base_color = material_color.rgb;
    material.opacity = material_color.a;
    material.alpha_cutoff = constants.visualization.w;
    material.normal_ws = material_normal(input);
    material.clear_coat_normal_ws = material_clear_coat_normal(input, material.normal_ws);
    material.tangent_ws = input.tangent_ws.xyz;
    material.roughness = clamp(constants.visualization.z * mr.g, 0.04, 1.0);
    material.metallic = clamp(constants.visualization.y * mr.b, 0.0, 1.0);
    material.ambient_occlusion = has_texture(8.0)
        ? mix(1.0, texture(occlusion_texture, input.uv0).r, constants.material_params.y)
        : 1.0;
    material.emissive_radiance = has_texture(16.0)
        ? texture(emissive_texture, input.uv0).rgb *
            material_parameters.emissive_factor.rgb * material_parameters.emissive_factor.w
        : material_parameters.emissive_factor.rgb * material_parameters.emissive_factor.w;
    material.clear_coat = material_parameters.material_lobes.x *
        (has_advanced_texture(1.0) ? texture(clear_coat_texture, input.uv0).r : 1.0);
    material.clear_coat_roughness = material_parameters.material_lobes.y *
        (has_advanced_texture(2.0) ? texture(clear_coat_roughness_texture, input.uv0).g : 1.0);
    material.anisotropy = material_parameters.material_lobes.z *
        (has_advanced_texture(8.0) ? texture(anisotropy_texture, input.uv0).b : 1.0);
    material.transmission = material_parameters.material_lobes.w *
        (has_advanced_texture(64.0) ? texture(transmission_texture, input.uv0).r : 1.0);
    material.index_of_refraction = material_parameters.volume_params.y;
    material.thickness = material_parameters.volume_params.z *
        (has_advanced_texture(32.0) ? texture(thickness_texture, input.uv0).r : 1.0);
    material.attenuation_color = material_parameters.attenuation_color.rgb;
    material.attenuation_distance = material_parameters.volume_params.w;
    material.subsurface_color = material_parameters.subsurface_color_factor.rgb;
    material.subsurface = material_parameters.subsurface_color_factor.w *
        (has_advanced_texture(16.0) ? texture(subsurface_texture, input.uv0).r : 1.0);
    return material;
}

arc_material_surface arc_evaluate_material(arc_material_inputs input)
{
    return arc_evaluate_legacy_material(input);
}

void main()
{
    vec3 light_dir = normalize(-constants.light_direction_intensity.xyz);
    vec3 view_dir = normalize(constants.camera_position.xyz - in_world_position);

    arc_material_inputs material_input;
    material_input.position_ws = in_world_position;
    material_input.normal_ws = in_normal;
    material_input.tangent_ws = in_tangent;
    material_input.uv0 = in_texcoord;
    material_input.uv1 = in_texcoord;
    material_input.vertex_color = in_color;
    material_input.view_ws = view_dir;
    arc_material_surface material = arc_evaluate_material(material_input);

    int alpha_mode = int(constants.material_params.w + 0.5);
    if (alpha_mode == 1 && material.opacity < material.alpha_cutoff)
        discard;

    arc_surface_data surface = arc_material_to_pbr_surface(material);
    float key_n_dot_l = max(dot(material.normal_ws, light_dir), 0.0);
    float shadow = sample_shadow(in_world_position);
    vec3 radiance = constants.light_color.rgb * constants.light_direction_intensity.w;
    vec3 direct = arc_evaluate_scene_lights(surface, view_dir, in_world_position, shadow);
    vec3 ambient = arc_evaluate_split_sum_ibl(
        surface,
        view_dir,
        vec3(0.18),
        vec3(0.18) * mix(0.35, 1.0, 1.0 - material.roughness),
        vec2(1.0 - 0.5 * material.roughness, 0.04));
    int shading_model = int(material_parameters.volume_params.x + 0.5);
    if (shading_model == 1 && material.subsurface > 0.0)
    {
        float wrapped = clamp((dot(material.normal_ws, light_dir) + 0.45) / 1.45, 0.0, 1.0);
        float back_scatter = pow(clamp(dot(-material.normal_ws, light_dir), 0.0, 1.0), 2.0);
        direct += material.subsurface_color * material.subsurface * radiance *
            (wrapped * 0.22 + back_scatter * 0.18) * shadow;
    }
    vec3 lit_color = ambient + direct + material.emissive_radiance;
    if (shading_model == 2 && material.transmission > 0.0)
    {
        float ior = max(material.index_of_refraction, 1.0001);
        float fresnel = pow((ior - 1.0) / (ior + 1.0), 2.0);
        vec3 transmitted_environment = mix(
            constants.fog_color_density.rgb,
            constants.light_color.rgb * 0.18,
            clamp(refract(-view_dir, material.normal_ws, 1.0 / ior).y * 0.5 + 0.5, 0.0, 1.0));
        transmitted_environment *= arc_beer_lambert(
            material.attenuation_color,
            material.attenuation_distance,
            material.thickness);
        float transmission_weight = material.transmission * (1.0 - fresnel);
        lit_color = mix(lit_color, transmitted_environment + material.emissive_radiance, transmission_weight);
    }

    int mode = int(constants.visualization.x + 0.5);
    vec3 color = lit_color;
    if (mode == 1)
        color = material.base_color;
    else if (mode == 2)
        color = vec3(material.opacity);
    else if (mode == 3)
        color = material.normal_ws * 0.5 + vec3(0.5);
    else if (mode == 4)
        color = mix(vec3(0.04), material.base_color, material.metallic);
    else if (mode == 5)
        color = vec3(1.0 - material.roughness);
    else if (mode == 6)
        color = vec3(material.metallic);
    else if (mode == 7)
        color = vec3(material.ambient_occlusion);
    else if (mode == 8)
        color = material.emissive_radiance;
    else if (mode == 9)
        color = vec3(key_n_dot_l * shadow);
    else if (mode == 10)
        color = vec3(material_input.uv0, 0.0);
    else
        color = apply_height_fog(color);

    out_color = vec4(color, material.opacity);
}
