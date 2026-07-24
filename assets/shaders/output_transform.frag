#version 450
#extension GL_GOOGLE_include_directive : require

#include "include/arc_color.glsl"

layout(location = 0) in vec2 in_uv;
layout(location = 0) out vec4 out_color;

layout(set = 0, binding = 0) uniform sampler2D scene_color;
layout(std430, set = 0, binding = 1) readonly buffer exposure_buffer
{
    uint bins[256];
    float ev100;
    uint valid;
    uint reserved0;
    uint reserved1;
} exposure;

layout(push_constant) uniform output_constants
{
    vec4 exposure_output;
} constants;

void main()
{
    vec4 hdr = texture(scene_color, in_uv);
    float exposure_multiplier = constants.exposure_output.x;
    if (constants.exposure_output.y > 0.5 && exposure.valid != 0u)
        exposure_multiplier = exp2(constants.exposure_output.z - exposure.ev100) / 1.2;
    vec3 exposed = hdr.rgb * max(exposure_multiplier, 0.0);
    vec3 display_linear = arc_aces_fitted(exposed);
    out_color = vec4(arc_linear_to_srgb(display_linear), hdr.a);
}
