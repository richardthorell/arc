#ifndef ARC_TEXTURE_SAMPLING_GLSL
#define ARC_TEXTURE_SAMPLING_GLSL

// Material ABI v2 centralizes texture sampling here. Conventional streamed
// mip windows require no shader-side LOD correction because their image view
// begins at the original resident base mip. Virtual metadata is introduced at
// the same call site without changing generated material expressions.
vec4 arc_sample_texture_2d(sampler2D texture_resource, vec2 uv, uint texture_metadata_index)
{
    return texture(texture_resource, uv);
}

vec3 arc_texture_mip_debug_color(float mip)
{
    const vec3 palette[8] = vec3[8](
        vec3(0.12, 0.82, 1.0), vec3(0.12, 1.0, 0.42), vec3(0.88, 1.0, 0.10), vec3(1.0, 0.62, 0.08),
        vec3(1.0, 0.18, 0.12), vec3(0.92, 0.12, 0.68), vec3(0.52, 0.18, 1.0), vec3(0.18, 0.22, 1.0));
    return palette[int(clamp(floor(mip + 0.5), 0.0, 7.0))];
}

vec3 arc_texture_desired_mip_debug(sampler2D texture_resource, vec2 uv)
{
    return arc_texture_mip_debug_color(max(textureQueryLod(texture_resource, uv).x, 0.0));
}

vec3 arc_texture_resident_mip_debug(sampler2D texture_resource)
{
    return arc_texture_mip_debug_color(float(max(textureQueryLevels(texture_resource) - 1, 0)));
}

#endif
