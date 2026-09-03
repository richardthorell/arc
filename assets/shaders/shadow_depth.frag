#version 450
#extension GL_GOOGLE_include_directive : require
#include "include/arc_texture_sampling.glsl"

layout(location = 0) in vec2 in_texcoord;
layout(set = 0, binding = 0) uniform sampler2D base_texture;

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

float blue_noise(vec2 pixel)
{
    // Deterministic interleaved gradient noise. This is stable in world/frame
    // space and provides the same authored opacity silhouette on every backend.
    return fract(52.9829189 * fract(dot(pixel, vec2(0.06711056, 0.00583715))));
}

void main()
{
    int alpha_mode = int(constants.material_params.w + 0.5);
    if (alpha_mode == 0)
        return;

    float alpha = arc_sample_texture_2d(base_texture, in_texcoord, 0u).a * constants.base_color.a;
    if (alpha_mode == 1)
    {
        if (alpha < constants.visualization.w)
            discard;
        return;
    }
    if (blue_noise(gl_FragCoord.xy) > alpha)
        discard;
}
