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

#endif
