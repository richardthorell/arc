#ifndef ARC_LIGHTING_GLSL
#define ARC_LIGHTING_GLSL

#ifndef ARC_LIGHT_BUFFER_BINDING
#define ARC_LIGHT_BUFFER_BINDING 15
#endif

struct directional_light_data
{
    vec4 direction_intensity;
    vec4 color_flags;
};

struct point_light_data
{
    vec4 position_range;
    vec4 color_intensity;
};

struct spot_light_data
{
    vec4 position_range;
    vec4 direction_inner_angle;
    vec4 color_intensity;
    vec4 params;
};

struct area_light_data
{
    vec4 position_shape;
    vec4 direction_two_sided;
    vec4 tangent_width;
    vec4 color_intensity;
    vec4 dimensions_shadow;
};

layout(std430, set = 0, binding = ARC_LIGHT_BUFFER_BINDING) readonly buffer scene_light_buffer
{
    directional_light_data directional_lights[4];
    point_light_data point_lights[64];
    spot_light_data spot_lights[64];
    area_light_data area_lights[32];
    vec4 ambient_color_intensity;
    uint directional_count;
    uint point_count;
    uint spot_count;
    uint area_count;
    uint skipped_directional_count;
    uint skipped_point_count;
    uint skipped_spot_count;
    uint skipped_area_count;
} lights;

vec3 arc_evaluate_surface_light(
    arc_surface_data surface,
    vec3 view_direction,
    vec3 light_direction,
    vec3 radiance,
    float visibility)
{
    arc_brdf_result brdf = arc_evaluate_brdf(surface, view_direction, light_direction);
    return (brdf.diffuse + brdf.specular) * radiance *
        max(dot(surface.normal, light_direction), 0.0) * visibility;
}

vec3 arc_evaluate_scene_lights(
    arc_surface_data surface,
    vec3 view_direction,
    vec3 world_position,
    float primary_directional_visibility)
{
    vec3 direct = vec3(0.0);
    for (uint index = 0u; index < min(lights.directional_count, 4u); ++index)
    {
        vec3 direction_to_light = normalize(-lights.directional_lights[index].direction_intensity.xyz);
        vec3 radiance = lights.directional_lights[index].color_flags.rgb *
            lights.directional_lights[index].direction_intensity.w;
        direct += arc_evaluate_surface_light(
            surface,
            view_direction,
            direction_to_light,
            radiance,
            index == 0u ? primary_directional_visibility : 1.0);
    }
    for (uint index = 0u; index < min(lights.point_count, 64u); ++index)
    {
        vec3 to_light = lights.point_lights[index].position_range.xyz - world_position;
        float distance_squared = max(dot(to_light, to_light), 1.0e-4);
        float distance_to_light = sqrt(distance_squared);
        float normalized_range = clamp(
            distance_to_light / max(lights.point_lights[index].position_range.w, 1.0e-4), 0.0, 1.0);
        float cutoff = 1.0 - pow(normalized_range, 4.0);
        vec3 radiance = lights.point_lights[index].color_intensity.rgb *
            lights.point_lights[index].color_intensity.w * cutoff * cutoff / distance_squared;
        direct += arc_evaluate_surface_light(
            surface, view_direction, to_light / distance_to_light, radiance, 1.0);
    }
    for (uint index = 0u; index < min(lights.spot_count, 64u); ++index)
    {
        vec3 to_light = lights.spot_lights[index].position_range.xyz - world_position;
        float distance_squared = max(dot(to_light, to_light), 1.0e-4);
        float distance_to_light = sqrt(distance_squared);
        vec3 direction_to_light = to_light / distance_to_light;
        float normalized_range = clamp(
            distance_to_light / max(lights.spot_lights[index].position_range.w, 1.0e-4), 0.0, 1.0);
        float cutoff = 1.0 - pow(normalized_range, 4.0);
        float cone = smoothstep(
            cos(lights.spot_lights[index].params.x),
            cos(lights.spot_lights[index].direction_inner_angle.w),
            dot(-direction_to_light, normalize(lights.spot_lights[index].direction_inner_angle.xyz)));
        vec3 radiance = lights.spot_lights[index].color_intensity.rgb *
            lights.spot_lights[index].color_intensity.w * cutoff * cutoff * cone / distance_squared;
        direct += arc_evaluate_surface_light(
            surface, view_direction, direction_to_light, radiance, 1.0);
    }
    for (uint index = 0u; index < min(lights.area_count, 32u); ++index)
    {
        vec3 to_light = lights.area_lights[index].position_shape.xyz - world_position;
        float distance_squared = max(dot(to_light, to_light), 1.0e-4);
        float distance_to_light = sqrt(distance_squared);
        vec3 direction_to_light = to_light / distance_to_light;
        float facing = dot(normalize(lights.area_lights[index].direction_two_sided.xyz), -direction_to_light);
        facing = lights.area_lights[index].direction_two_sided.w > 0.5 ? abs(facing) : max(facing, 0.0);
        float width = max(lights.area_lights[index].tangent_width.w, 1.0e-4);
        float height = max(lights.area_lights[index].dimensions_shadow.y, 1.0e-4);
        float area = lights.area_lights[index].position_shape.w > 0.5
            ? ARC_PI * width * height * 0.25
            : width * height;
        vec3 radiance = lights.area_lights[index].color_intensity.rgb *
            lights.area_lights[index].color_intensity.w *
            min(area * facing / distance_squared, 2.0 * ARC_PI);
        direct += arc_evaluate_surface_light(
            surface, view_direction, direction_to_light, radiance, 1.0);
    }
    return direct;
}

#endif
