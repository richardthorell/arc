#ifndef ARC_SHADOWS_GLSL
#define ARC_SHADOWS_GLSL

#include "arc_virtual_shadows.glsl"

#ifndef ARC_SHADOW_SET
#define ARC_SHADOW_SET 0
#endif
#ifndef ARC_SHADOW_TEXTURE_BINDING
#define ARC_SHADOW_TEXTURE_BINDING 5
#endif
#ifndef ARC_SHADOW_DATA_BINDING
#define ARC_SHADOW_DATA_BINDING 6
#endif
#ifndef ARC_VIRTUAL_SHADOW_ADDRESS_BINDING
#define ARC_VIRTUAL_SHADOW_ADDRESS_BINDING 18
#endif
#ifndef ARC_VIRTUAL_SHADOW_VIEW_BINDING
#define ARC_VIRTUAL_SHADOW_VIEW_BINDING 19
#endif
#ifndef ARC_VIRTUAL_SHADOW_PAGE_TABLE_BINDING
#define ARC_VIRTUAL_SHADOW_PAGE_TABLE_BINDING 20
#endif
#ifndef ARC_VIRTUAL_SHADOW_STATIC_BINDING
#define ARC_VIRTUAL_SHADOW_STATIC_BINDING 21
#endif
#ifndef ARC_VIRTUAL_SHADOW_DYNAMIC_BINDING
#define ARC_VIRTUAL_SHADOW_DYNAMIC_BINDING 22
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
    vec4 source_shape;
} arc_shadows;

#ifndef ARC_DISABLE_VIRTUAL_SHADOW_SAMPLING
layout(std430, set = ARC_SHADOW_SET, binding = ARC_VIRTUAL_SHADOW_ADDRESS_BINDING) readonly buffer
arc_virtual_shadow_address_buffer
{
    ArcVirtualShadowAddressSpace arc_virtual_shadow_addresses[];
};

layout(std430, set = ARC_SHADOW_SET, binding = ARC_VIRTUAL_SHADOW_VIEW_BINDING) readonly buffer
arc_virtual_shadow_view_buffer
{
    ArcVirtualShadowView arc_virtual_shadow_views[];
};

layout(std430, set = ARC_SHADOW_SET, binding = ARC_VIRTUAL_SHADOW_PAGE_TABLE_BINDING) readonly buffer
arc_virtual_shadow_page_table_buffer
{
    ArcVirtualShadowPageTableEntry arc_virtual_shadow_page_table[];
};

layout(set = ARC_SHADOW_SET, binding = ARC_VIRTUAL_SHADOW_STATIC_BINDING)
uniform sampler2DShadow arc_virtual_shadow_static_atlas;
layout(set = ARC_SHADOW_SET, binding = ARC_VIRTUAL_SHADOW_DYNAMIC_BINDING)
uniform sampler2DShadow arc_virtual_shadow_dynamic_atlas;
#endif

const uint ARC_DIRECTIONAL_SHADOW_NONE = 0u;
const uint ARC_DIRECTIONAL_SHADOW_CONVENTIONAL = 1u;
const uint ARC_DIRECTIONAL_SHADOW_VIRTUALIZED = 2u;

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
    // Conventional CSM approximates a finite directional emitter by widening
    // its PCF footprint. source_shape.x is the shared angular-radius contract
    // consumed by both conventional shadows and future VSM filtering.
    float source_scale = 1.0 + clamp(0.5 * arc_shadows.source_shape.x, 0.0, 0.25) * 48.0;
    vec2 texel = vec2(1.0 / float(textureSize(arc_directional_shadow_map, 0).x)) * source_scale;
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

    vec3 camera_forward = normalize(arc_shadows.configuration.yzw);
    float camera_depth = max(dot(world_position - camera_position, camera_forward), 0.0);
    int cascade = arc_shadow_cascade(camera_depth);
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
        float blend = smoothstep(blend_start, max(blend_end, blend_start + 1.0e-5), camera_depth);
        if (blend > 0.0)
            visibility = mix(
                visibility,
                arc_sample_shadow_cascade(
                    cascade + 1, world_position, surface_normal, light_direction),
                blend);
    }
    return mix(1.0 - arc_shadows.params.x, 1.0, visibility);
}

#ifndef ARC_DISABLE_VIRTUAL_SHADOW_SAMPLING
bool arc_virtual_shadow_project(
    ArcVirtualShadowView view,
    vec3 world_position,
    out vec3 projected,
    out uvec2 page,
    out vec2 page_uv)
{
    vec4 clip = arcVirtualShadowTransform(view, world_position);
    if (view.pageRange.y == 0u || abs(clip.w) <= 1.0e-6 || any(isnan(clip)) || any(isinf(clip)))
        return false;
    projected = clip.xyz / clip.w;
    vec2 uv = projected.xy * 0.5 + 0.5;
    if (projected.z < 0.0 || projected.z > 1.0 ||
        any(lessThan(uv, vec2(0.0))) || any(greaterThanEqual(uv, vec2(1.0))))
        return false;
    vec2 page_position = uv * float(view.pageRange.y);
    page = uvec2(floor(page_position));
    page_uv = fract(page_position);
    return true;
}

float arc_sample_virtual_shadow_tap(
    ArcVirtualShadowPhysicalMapping mapping,
    vec2 page_uv,
    float receiver_depth,
    float comparison_bias,
    ivec2 tap,
    bool dynamic_layer)
{
    ivec2 atlas_size = textureSize(arc_virtual_shadow_static_atlas, 0);
    uint pages_per_axis = uint(atlas_size.x) / (ARC_VIRTUAL_SHADOW_PAGE_TEXELS +
                                                ARC_VIRTUAL_SHADOW_PAGE_GUARD_TEXELS * 2u);
    uint physical_page = mapping.value.x;
    uvec2 tile = uvec2(physical_page % max(pages_per_axis, 1u),
                       physical_page / max(pages_per_axis, 1u));
    vec2 atlas_pixel = vec2(tile * (ARC_VIRTUAL_SHADOW_PAGE_TEXELS +
                                    ARC_VIRTUAL_SHADOW_PAGE_GUARD_TEXELS * 2u) +
                            ARC_VIRTUAL_SHADOW_PAGE_GUARD_TEXELS) +
                       page_uv * float(ARC_VIRTUAL_SHADOW_PAGE_TEXELS);
    // Projected UVs already describe continuous pixel coordinates; adding a
    // half texel here would offset the comparison from the raster projection.
    vec2 tile_min = vec2(tile * 136u) + vec2(0.5);
    vec2 tile_max = tile_min + vec2(135.0);
    vec2 atlas_uv = clamp(atlas_pixel + vec2(tap), tile_min, tile_max) / vec2(atlas_size);
    return dynamic_layer
        ? texture(arc_virtual_shadow_dynamic_atlas, vec3(atlas_uv, receiver_depth - comparison_bias))
        : texture(arc_virtual_shadow_static_atlas, vec3(atlas_uv, receiver_depth - comparison_bias));
}

struct ArcResolvedVirtualShadowLayer
{
    ArcVirtualShadowPhysicalMapping mapping;
    vec2 page_uv;
    float depth;
};

bool arc_resolve_virtual_shadow_layer(
    ArcVirtualShadowAddressSpace address_space,
    vec3 world_position,
    bool dynamic_layer,
    out ArcResolvedVirtualShadowLayer resolved)
{
    uint level_count = min(arcVirtualShadowLevelCount(address_space), address_space.ranges.y);
    for (uint level = 0u; level < level_count; ++level)
    {
        uint view_index = arcVirtualShadowViewIndex(address_space, 0u, level);
        if (view_index >= arc_virtual_shadow_views.length())
            break;
        ArcVirtualShadowView view = arc_virtual_shadow_views[view_index];
        if (view.pageRange.z != 0u || view.pageRange.w != level)
            continue;
        vec3 projected;
        uvec2 page;
        vec2 page_uv;
        if (!arc_virtual_shadow_project(view, world_position, projected, page, page_uv))
            continue;
        uint dense_index = arcVirtualShadowDensePageIndex(address_space, view, page);
        if (dense_index < address_space.ranges.z ||
            dense_index - address_space.ranges.z >= address_space.ranges.w ||
            dense_index >= arc_virtual_shadow_page_table.length())
            continue;
        ArcVirtualShadowPageTableEntry entry = arc_virtual_shadow_page_table[dense_index];
        ArcVirtualShadowPhysicalMapping mapping = dynamic_layer ? entry.dynamicDepth : entry.staticDepth;
        uint atlas_axis = uint(textureSize(arc_virtual_shadow_static_atlas, 0).x) / 136u;
        if (mapping.value.x == ARC_VIRTUAL_SHADOW_INVALID_INDEX || mapping.value.y == 0u ||
            mapping.value.x >= atlas_axis * atlas_axis)
            continue;
        resolved.mapping = mapping;
        resolved.page_uv = page_uv;
        resolved.depth = projected.z;
        return true;
    }
    return false;
}

bool arc_virtual_directional_shadow_visibility(
    directional_light_data light,
    vec3 world_position,
    vec3 surface_normal,
    vec3 light_direction,
    out float visibility)
{
    uint address_index = light.shadow_identity.z;
    if (address_index == ARC_VIRTUAL_SHADOW_INVALID_INDEX ||
        address_index >= arc_virtual_shadow_addresses.length())
        return false;
    ArcVirtualShadowAddressSpace address_space = arc_virtual_shadow_addresses[address_index];
    if (address_space.identityTopology.x != light.shadow_identity.w ||
        address_space.identityTopology.y != 0u)
        return false;

    float normal_bias = light.shadow_parameters.z *
        clamp(1.0 - dot(normalize(surface_normal), normalize(light_direction)), 0.0, 1.0);
    float comparison_bias = light.shadow_parameters.y + normal_bias;
    uint mobility = address_space.requestMetadata.x;
    bool needs_static = mobility != 2u;
    bool needs_dynamic = mobility != 0u;
    ArcResolvedVirtualShadowLayer static_layer;
    ArcResolvedVirtualShadowLayer dynamic_layer;
    bool static_found = !needs_static || arc_resolve_virtual_shadow_layer(
        address_space, world_position, false, static_layer);
    bool dynamic_found = !needs_dynamic || arc_resolve_virtual_shadow_layer(
        address_space, world_position, true, dynamic_layer);
    if (!static_found || !dynamic_found)
        return false;
    // Combine layers before filtering: min of two independently averaged PCF
    // results misses disjoint static/dynamic occluders within the footprint.
    uint filter_mode = light.shadow_routing.y;
    int radius = filter_mode == 0u ? 0 : (filter_mode == 1u ? 1 : 2);
    float sampled = 0.0;
    for (int y = -radius; y <= radius; ++y)
        for (int x = -radius; x <= radius; ++x)
        {
            float static_visibility = needs_static ? arc_sample_virtual_shadow_tap(
                static_layer.mapping, static_layer.page_uv, static_layer.depth, comparison_bias, ivec2(x, y), false) : 1.0;
            float dynamic_visibility = needs_dynamic ? arc_sample_virtual_shadow_tap(
                dynamic_layer.mapping, dynamic_layer.page_uv, dynamic_layer.depth, comparison_bias, ivec2(x, y), true) : 1.0;
            sampled += min(static_visibility, dynamic_visibility);
        }
    sampled /= float((radius * 2 + 1) * (radius * 2 + 1));
    visibility = mix(1.0 - clamp(light.shadow_parameters.x, 0.0, 1.0), 1.0, sampled);
    return true;
}

#endif

float arc_directional_light_shadow_visibility(
    directional_light_data light,
    vec3 world_position,
    vec3 surface_normal,
    vec3 camera_position,
    vec3 light_direction,
    out int resolved_cascade)
{
    resolved_cascade = -1;
    if (light.shadow_routing.x == ARC_DIRECTIONAL_SHADOW_NONE)
        return 1.0;
#ifndef ARC_DISABLE_VIRTUAL_SHADOW_SAMPLING
    if (light.shadow_routing.x == ARC_DIRECTIONAL_SHADOW_VIRTUALIZED)
    {
        float virtual_visibility = 1.0;
        if (arc_virtual_directional_shadow_visibility(
                light, world_position, surface_normal, light_direction, virtual_visibility))
            return virtual_visibility;
    }
#endif
    return arc_directional_shadow_visibility(
        world_position, surface_normal, camera_position, light_direction, resolved_cascade);
}

vec4 arc_scene_directional_shadow_visibility(
    vec3 world_position, vec3 normal, vec3 camera_position, out int resolved_cascade)
{
    vec4 visibility = vec4(1.0);
    resolved_cascade = -1;
    for (uint index = 0u; index < min(lights.directional_count, 4u); ++index)
    {
        directional_light_data light = lights.directional_lights[index];
        if (light.shadow_routing.x == ARC_DIRECTIONAL_SHADOW_NONE)
            continue;
        visibility[index] = arc_directional_light_shadow_visibility(
            light, world_position, normal, camera_position,
            normalize(-light.direction_intensity.xyz), resolved_cascade);
    }
    return visibility;
}

#endif
