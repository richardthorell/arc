#version 450
#extension GL_GOOGLE_include_directive : require

#include "include/arc_pbr.glsl"
#define ARC_LIGHT_BUFFER_BINDING 7
#include "include/arc_lighting.glsl"

layout(location = 0) in vec2 in_uv;
layout(location = 0) out vec4 out_color;

layout(set = 0, binding = 0) uniform sampler2D gbuffer_albedo;
layout(set = 0, binding = 1) uniform sampler2D gbuffer_normal;
layout(set = 0, binding = 2) uniform sampler2D gbuffer_material;
layout(set = 0, binding = 3) uniform sampler2D gbuffer_emissive;
layout(set = 0, binding = 4) uniform usampler2D gbuffer_object_id;
layout(set = 0, binding = 5) uniform sampler2D gbuffer_motion;
layout(set = 0, binding = 6) uniform sampler2D scene_depth;
layout(set = 0, binding = 8) uniform sampler2D environment_radiance;

layout(push_constant) uniform deferred_constants
{
    mat4 inverse_view_projection;
    vec4 camera_position;
    vec4 light_direction_intensity;
    vec4 light_color;
    vec4 ambient_visualization;
} constants;

vec3 reconstruct_world_position(vec2 uv, float depth)
{
    vec4 clip_position = vec4(uv * 2.0 - vec2(1.0), depth, 1.0);
    vec4 world_position = constants.inverse_view_projection * clip_position;
    return world_position.xyz / max(abs(world_position.w), 1.0e-6);
}

vec3 evaluate_light(
    arc_surface_data surface,
    vec3 view_direction,
    vec3 light_direction,
    vec3 radiance,
    float visibility)
{
    arc_brdf_result brdf = arc_evaluate_brdf(surface, view_direction, light_direction);
    float n_dot_l = max(dot(surface.normal, light_direction), 0.0);
    return (brdf.diffuse + brdf.specular) * radiance * n_dot_l * visibility;
}

vec2 environment_uv(vec3 direction)
{
    direction = normalize(direction);
    return vec2(
        atan(direction.z, direction.x) / (2.0 * ARC_PI) + 0.5,
        asin(clamp(direction.y, -1.0, 1.0)) / ARC_PI + 0.5);
}

vec3 sample_environment(vec3 direction, float roughness)
{
    int levels = max(textureQueryLevels(environment_radiance), 1);
    return textureLod(
        environment_radiance,
        environment_uv(direction),
        roughness * float(levels - 1)).rgb;
}

void main()
{
    vec4 albedo = texture(gbuffer_albedo, in_uv);
    vec4 normal_ao = texture(gbuffer_normal, in_uv);
    vec4 material = texture(gbuffer_material, in_uv);
    vec3 emissive = texture(gbuffer_emissive, in_uv).rgb;
    uint object_id = texture(gbuffer_object_id, in_uv).r;
    vec2 motion = texture(gbuffer_motion, in_uv).rg;
    float depth = texture(scene_depth, in_uv).r;
    if (object_id == 0u && albedo.a <= 0.0)
        discard;

    vec3 world_position = reconstruct_world_position(in_uv, depth);
    vec3 view_direction = normalize(constants.camera_position.xyz - world_position);

    arc_surface_data surface;
    surface.base_color = albedo.rgb;
    surface.normal = normalize(normal_ao.xyz * 2.0 - vec3(1.0));
    surface.clear_coat_normal = surface.normal;
    surface.tangent = vec3(0.0);
    surface.emissive = emissive;
    surface.metallic = arc_saturate(material.x);
    surface.perceptual_roughness = clamp(material.y, 0.04, 1.0);
    surface.occlusion = normal_ao.w;
    surface.clear_coat = 0.0;
    surface.clear_coat_roughness = 0.0;
    surface.anisotropy = 0.0;

    float shadow = clamp(material.z, 0.0, 1.0);
    vec3 direct = vec3(0.0);
    for (uint index = 0u; index < min(lights.directional_count, 4u); ++index)
    {
        directional_light_data light = lights.directional_lights[index];
        vec3 light_direction = normalize(-light.direction_intensity.xyz);
        vec3 radiance = light.color_flags.rgb * light.direction_intensity.w;
        direct += evaluate_light(
            surface,
            view_direction,
            light_direction,
            radiance,
            index == 0u ? shadow : 1.0);
    }
    for (uint index = 0u; index < min(lights.point_count, 64u); ++index)
    {
        point_light_data light = lights.point_lights[index];
        vec3 to_light = light.position_range.xyz - world_position;
        float distance_squared = max(dot(to_light, to_light), 1.0e-4);
        float distance_to_light = sqrt(distance_squared);
        float normalized_range = clamp(distance_to_light / max(light.position_range.w, 1.0e-4), 0.0, 1.0);
        float smooth_cutoff = 1.0 - pow(normalized_range, 4.0);
        float attenuation = smooth_cutoff * smooth_cutoff / distance_squared;
        direct += evaluate_light(
            surface,
            view_direction,
            to_light / distance_to_light,
            light.color_intensity.rgb * light.color_intensity.w * attenuation,
            1.0);
    }
    for (uint index = 0u; index < min(lights.spot_count, 64u); ++index)
    {
        spot_light_data light = lights.spot_lights[index];
        vec3 to_light = light.position_range.xyz - world_position;
        float distance_squared = max(dot(to_light, to_light), 1.0e-4);
        float distance_to_light = sqrt(distance_squared);
        vec3 direction_to_light = to_light / distance_to_light;
        float normalized_range = clamp(distance_to_light / max(light.position_range.w, 1.0e-4), 0.0, 1.0);
        float smooth_cutoff = 1.0 - pow(normalized_range, 4.0);
        float outer_cosine = cos(light.params.x);
        float inner_cosine = cos(light.direction_inner_angle.w);
        float cone = smoothstep(
            outer_cosine,
            max(inner_cosine, outer_cosine + 1.0e-5),
            dot(-direction_to_light, normalize(light.direction_inner_angle.xyz)));
        float attenuation = smooth_cutoff * smooth_cutoff * cone / distance_squared;
        direct += evaluate_light(
            surface,
            view_direction,
            direction_to_light,
            light.color_intensity.rgb * light.color_intensity.w * attenuation,
            1.0);
    }
    for (uint index = 0u; index < min(lights.area_count, 32u); ++index)
    {
        area_light_data light = lights.area_lights[index];
        vec3 emitter_normal = normalize(light.direction_two_sided.xyz);
        vec3 emitter_tangent = normalize(light.tangent_width.xyz);
        vec3 emitter_bitangent = normalize(cross(emitter_normal, emitter_tangent));
        float width = max(light.tangent_width.w, 1.0e-4);
        float height = max(light.dimensions_shadow.y, 1.0e-4);
        vec3 from_center = world_position - light.position_shape.xyz;
        vec2 local = vec2(dot(from_center, emitter_tangent), dot(from_center, emitter_bitangent));
        if (light.position_shape.w > 0.5)
        {
            vec2 normalized_local = local / vec2(width * 0.5, height * 0.5);
            float radial = length(normalized_local);
            if (radial > 1.0)
                local /= radial;
        }
        else
            local = clamp(local, -vec2(width, height) * 0.5, vec2(width, height) * 0.5);
        vec3 closest_emitter_point = light.position_shape.xyz +
            emitter_tangent * local.x + emitter_bitangent * local.y;
        vec3 to_light = closest_emitter_point - world_position;
        float distance_squared = max(dot(to_light, to_light), 1.0e-4);
        float distance_to_light = sqrt(distance_squared);
        vec3 direction_to_light = to_light / distance_to_light;
        float facing = dot(emitter_normal, -direction_to_light);
        if (light.direction_two_sided.w > 0.5)
            facing = abs(facing);
        else
            facing = max(facing, 0.0);
        float area = light.position_shape.w > 0.5
            ? ARC_PI * width * 0.5 * height * 0.5
            : width * height;
        float solid_angle = min(area * facing / distance_squared, 2.0 * ARC_PI);
        direct += evaluate_light(
            surface,
            view_direction,
            direction_to_light,
            light.color_intensity.rgb * light.color_intensity.w * solid_angle,
            1.0);
    }

    vec3 f0 = mix(vec3(ARC_DIELECTRIC_F0), surface.base_color, surface.metallic);
    vec3 ambient_diffuse = constants.ambient_visualization.rgb;
    vec3 ambient_specular = constants.ambient_visualization.rgb *
        mix(0.35, 1.0, 1.0 - surface.perceptual_roughness);
    if (constants.light_color.w > 0.5)
    {
        ambient_diffuse = sample_environment(surface.normal, 1.0);
        vec3 reflected = reflect(-view_direction, surface.normal);
        vec3 sharp_specular = sample_environment(reflected, surface.perceptual_roughness);
        ambient_specular = mix(
            sharp_specular,
            constants.ambient_visualization.rgb,
            surface.perceptual_roughness * surface.perceptual_roughness);
    }
    vec3 ambient = arc_evaluate_split_sum_ibl(
        surface,
        view_direction,
        ambient_diffuse,
        ambient_specular,
        vec2(1.0 - 0.5 * surface.perceptual_roughness, 0.04));
    vec3 lit = ambient + direct + emissive;

    int mode = int(constants.ambient_visualization.w + 0.5);
    vec3 color = lit;
    if (mode == 1)
        color = albedo.rgb;
    else if (mode == 2)
        color = vec3(albedo.a);
    else if (mode == 3)
        color = surface.normal * 0.5 + vec3(0.5);
    else if (mode == 4)
        color = f0;
    else if (mode == 5)
        color = vec3(1.0 - surface.perceptual_roughness);
    else if (mode == 6)
        color = vec3(surface.metallic);
    else if (mode == 7)
        color = vec3(surface.occlusion);
    else if (mode == 8)
        color = emissive;
    else if (mode == 9)
        color = direct;
    else if (mode == 10)
        color = vec3(in_uv, 0.0);
    else if (mode == 11)
        color = vec3(motion * 0.5 + 0.5, 0.0);
    else if (mode == 12)
        color = vec3(shadow);

    out_color = vec4(color, 1.0);
}
