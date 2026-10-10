#ifndef ARC_LIGHTING_GLSL
#define ARC_LIGHTING_GLSL

#ifndef ARC_LIGHT_BUFFER_BINDING
#define ARC_LIGHT_BUFFER_BINDING 15
#endif
#ifndef ARC_LOCAL_SHADOW_BINDING
#define ARC_LOCAL_SHADOW_BINDING 17
#endif

struct directional_light_data
{
    vec4 direction_intensity;
    vec4 color_flags;
    vec4 source_shape;
};

struct point_light_data
{
    vec4 position_range;
    vec4 color_intensity;
    vec4 object_id_shadow;
    vec4 shadow_parameters;
    vec4 source_shape;
};

struct spot_light_data
{
    vec4 position_range;
    vec4 direction_inner_angle;
    vec4 color_intensity;
    vec4 params;
    vec4 object_id_shadow;
    vec4 shadow_parameters;
    vec4 source_shape;
};

struct local_shadow_face_data
{
    mat4 light_view_projection;
    vec4 atlas_rect;
    vec4 parameters;
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
    local_shadow_face_data local_shadow_faces[144];
    vec4 ambient_color_intensity;
    uint directional_count;
    uint point_count;
    uint spot_count;
    uint area_count;
    uint skipped_directional_count;
    uint skipped_point_count;
    uint skipped_spot_count;
    uint skipped_area_count;
    uint local_shadow_face_count;
    uint local_shadow_padding0;
    uint local_shadow_padding1;
    uint local_shadow_padding2;
} lights;

layout(set = 0, binding = ARC_LOCAL_SHADOW_BINDING) uniform sampler2DShadow arc_local_shadow_atlas;

float arc_sample_local_shadow_face(uint face_index, vec3 world_position, float filter_scale)
{
    if (face_index >= min(lights.local_shadow_face_count, 144u))
        return 1.0;
    local_shadow_face_data face = lights.local_shadow_faces[face_index];
    vec4 clip = face.light_view_projection * vec4(world_position, 1.0);
    if (clip.w <= 0.0)
        return 1.0;
    vec3 projected = clip.xyz / clip.w;
    vec2 uv = projected.xy * 0.5 + 0.5;
    if (projected.z <= 0.0 || projected.z >= 1.0 ||
        any(lessThan(uv, vec2(0.0))) || any(greaterThan(uv, vec2(1.0))))
        return 1.0;
    vec2 atlas_uv = face.atlas_rect.xy + uv * face.atlas_rect.zw;
    float comparison = projected.z - face.parameters.y;
    float texel = face.parameters.x;
    float visibility = 0.0;
    for (int y = -1; y <= 1; ++y)
        for (int x = -1; x <= 1; ++x)
            visibility += texture(
                arc_local_shadow_atlas,
                vec3(atlas_uv + vec2(x, y) * texel * max(filter_scale, 1.0), comparison));
    return visibility / 9.0;
}

uint arc_point_shadow_face(vec3 direction)
{
    vec3 absolute_direction = abs(direction);
    if (absolute_direction.x >= absolute_direction.y && absolute_direction.x >= absolute_direction.z)
        return direction.x >= 0.0 ? 0u : 1u;
    if (absolute_direction.y >= absolute_direction.z)
        return direction.y >= 0.0 ? 2u : 3u;
    return direction.z >= 0.0 ? 4u : 5u;
}

float arc_point_shadow_visibility(point_light_data light, vec3 world_position)
{
    int first_face = int(light.shadow_parameters.x + 0.5);
    if (first_face < 0 || light.shadow_parameters.y < 5.5)
        return 1.0;
    uint face = arc_point_shadow_face(world_position - light.position_range.xyz);
    float effective_source_radius = max(max(light.source_shape.x, 0.0), 0.5 * max(light.source_shape.y, 0.0));
    float filter_scale = 1.0 + 12.0 * effective_source_radius / max(light.position_range.w, 1.0e-4);
    float sampled = arc_sample_local_shadow_face(uint(first_face) + face, world_position, filter_scale);
    return mix(1.0, sampled, clamp(light.shadow_parameters.z, 0.0, 1.0));
}

float arc_spot_shadow_visibility(spot_light_data light, vec3 world_position)
{
    int face = int(light.shadow_parameters.x + 0.5);
    if (face < 0)
        return 1.0;
    float filter_scale = 1.0 + 12.0 * max(light.source_shape.x, 0.0) / max(light.position_range.w, 1.0e-4);
    float sampled = arc_sample_local_shadow_face(uint(face), world_position, filter_scale);
    return mix(1.0, sampled, clamp(light.shadow_parameters.z, 0.0, 1.0));
}

float arc_source_angular_radius(float source_radius, float distance_to_light)
{
    if (source_radius <= 0.0)
        return 0.0;
    return asin(clamp(source_radius / max(distance_to_light, source_radius), 0.0, 1.0));
}

arc_surface_data arc_surface_for_extended_source(arc_surface_data surface, float angular_radius)
{
    // A finite emitter occupies a cone of incident directions. Approximate that
    // convolution by broadening the microfacet lobe while preserving the exact
    // punctual-light result when the authored source size is zero.
    float source_roughness = clamp(angular_radius * (2.0 / ARC_PI), 0.0, 1.0);
    surface.perceptual_roughness = max(surface.perceptual_roughness, source_roughness);
    surface.clear_coat_roughness = max(surface.clear_coat_roughness, source_roughness);
    return surface;
}

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
        float source_angle = max(lights.directional_lights[index].source_shape.x, 0.0);
        direct += arc_evaluate_surface_light(
            arc_surface_for_extended_source(surface, source_angle),
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
        float source_radius = max(
            max(lights.point_lights[index].source_shape.x, 0.0),
            0.5 * max(lights.point_lights[index].source_shape.y, 0.0));
        float source_angle = arc_source_angular_radius(source_radius, distance_to_light);
        direct += arc_evaluate_surface_light(
            arc_surface_for_extended_source(surface, source_angle),
            view_direction,
            to_light / distance_to_light,
            radiance,
            arc_point_shadow_visibility(lights.point_lights[index], world_position));
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
        float source_radius = max(
            max(lights.spot_lights[index].source_shape.x, 0.0),
            0.5 * max(lights.spot_lights[index].source_shape.y, 0.0));
        float source_angle = arc_source_angular_radius(source_radius, distance_to_light);
        direct += arc_evaluate_surface_light(
            arc_surface_for_extended_source(surface, source_angle),
            view_direction,
            direction_to_light,
            radiance,
            arc_spot_shadow_visibility(lights.spot_lights[index], world_position));
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
