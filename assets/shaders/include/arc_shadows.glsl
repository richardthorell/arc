#ifndef ARC_SHADOWS_GLSL
#define ARC_SHADOWS_GLSL

#ifndef ARC_SHADOW_SET
#define ARC_SHADOW_SET 0
#endif
#ifndef ARC_SHADOW_TEXTURE_BINDING
#define ARC_SHADOW_TEXTURE_BINDING 5
#endif
#ifndef ARC_SHADOW_DATA_BINDING
#define ARC_SHADOW_DATA_BINDING 6
#endif

layout(set = ARC_SHADOW_SET, binding = ARC_SHADOW_TEXTURE_BINDING)
uniform sampler2DArrayShadow arc_directional_shadow_map;

layout(set = ARC_SHADOW_SET, binding = ARC_SHADOW_DATA_BINDING) uniform arc_shadow_data
{
    mat4 light_view_projection[4];
    vec4 cascade_splits;
    vec4 params;
    vec4 cascade_texel_size;
    vec4 cascade_blend_starts;
    vec4 configuration;
} arc_shadows;

int arc_shadow_cascade(float camera_distance)
{
    int cascade_count = clamp(int(arc_shadows.configuration.x + 0.5), 0, 4);
    for (int cascade = 0; cascade < cascade_count; ++cascade)
        if (camera_distance <= arc_shadows.cascade_splits[cascade])
            return cascade;
    return -1;
}

float arc_sample_shadow_cascade(
    int cascade,
    vec3 world_position,
    vec3 surface_normal,
    vec3 light_direction)
{
    vec4 light_clip = arc_shadows.light_view_projection[cascade] * vec4(world_position, 1.0);
    vec3 projected = light_clip.xyz / max(abs(light_clip.w), 1.0e-6);
    vec2 uv = projected.xy * 0.5 + vec2(0.5);
    if (any(lessThan(uv, vec2(0.0))) || any(greaterThan(uv, vec2(1.0))) ||
        projected.z < 0.0 || projected.z > 1.0)
        return 1.0;

    float normal_bias = arc_shadows.params.z *
        clamp(1.0 - dot(normalize(surface_normal), normalize(light_direction)), 0.0, 1.0);
    float compare_depth = projected.z - arc_shadows.params.y - normal_bias;
    int filter_mode = int(arc_shadows.params.w + 0.5);
    if (filter_mode == 0)
    {
        float static_visibility = texture(
            arc_directional_shadow_map, vec4(uv, float(cascade), compare_depth));
        float dynamic_visibility = texture(
            arc_directional_shadow_map, vec4(uv, float(cascade + 4), compare_depth));
        return min(static_visibility, dynamic_visibility);
    }

    int radius = filter_mode >= 2 ? 2 : 1;
    vec2 texel = vec2(1.0 / float(textureSize(arc_directional_shadow_map, 0).x));
    float visibility = 0.0;
    float sample_count = 0.0;
    for (int y = -radius; y <= radius; ++y)
        for (int x = -radius; x <= radius; ++x)
        {
            float static_visibility = texture(
                arc_directional_shadow_map,
                vec4(uv + vec2(x, y) * texel, float(cascade), compare_depth));
            float dynamic_visibility = texture(
                arc_directional_shadow_map,
                vec4(uv + vec2(x, y) * texel, float(cascade + 4), compare_depth));
            visibility += min(static_visibility, dynamic_visibility);
            sample_count += 1.0;
        }
    return visibility / max(sample_count, 1.0);
}

float arc_directional_shadow_visibility(
    vec3 world_position,
    vec3 surface_normal,
    vec3 camera_position,
    vec3 light_direction,
    out int resolved_cascade)
{
    resolved_cascade = -1;
    if (arc_shadows.params.x <= 0.0 || arc_shadows.configuration.x < 0.5)
        return 1.0;

    float camera_distance = length(camera_position - world_position);
    int cascade = arc_shadow_cascade(camera_distance);
    resolved_cascade = cascade;
    if (cascade < 0)
        return 1.0;

    float visibility = arc_sample_shadow_cascade(
        cascade, world_position, surface_normal, light_direction);
    int cascade_count = clamp(int(arc_shadows.configuration.x + 0.5), 0, 4);
    if (cascade + 1 < cascade_count)
    {
        float blend_start = arc_shadows.cascade_blend_starts[cascade];
        float blend_end = arc_shadows.cascade_splits[cascade];
        float blend = smoothstep(blend_start, max(blend_end, blend_start + 1.0e-5), camera_distance);
        if (blend > 0.0)
            visibility = mix(
                visibility,
                arc_sample_shadow_cascade(
                    cascade + 1, world_position, surface_normal, light_direction),
                blend);
    }
    return mix(1.0 - arc_shadows.params.x, 1.0, visibility);
}

#endif
