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
    // x: apply FXAA, yz: inverse output extent.
    vec4 post_process;
} constants;

vec3 arc_output_linear(vec2 uv, float exposure_multiplier)
{
    vec3 exposed = texture(scene_color, uv).rgb * max(exposure_multiplier, 0.0);
    return arc_aces_fitted(exposed);
}

void main()
{
    vec4 hdr = texture(scene_color, in_uv);
    if (constants.exposure_output.w > 0.5)
    {
        out_color = vec4(arc_linear_to_srgb(clamp(hdr.rgb, vec3(0.0), vec3(1.0))), hdr.a);
        return;
    }
    float exposure_multiplier = constants.exposure_output.x;
    if (constants.exposure_output.y > 0.5 && exposure.valid != 0u)
        exposure_multiplier = exp2(constants.exposure_output.z - exposure.ev100) / 1.2;
    vec3 display_linear = arc_output_linear(in_uv, exposure_multiplier);
    if (constants.post_process.x > 0.5)
    {
        vec2 texel = constants.post_process.yz;
        vec3 north_west = arc_output_linear(in_uv + texel * vec2(-1.0, -1.0), exposure_multiplier);
        vec3 north_east = arc_output_linear(in_uv + texel * vec2(1.0, -1.0), exposure_multiplier);
        vec3 south_west = arc_output_linear(in_uv + texel * vec2(-1.0, 1.0), exposure_multiplier);
        vec3 south_east = arc_output_linear(in_uv + texel * vec2(1.0, 1.0), exposure_multiplier);
        const vec3 luminance_weights = vec3(0.299, 0.587, 0.114);
        float luma_center = dot(display_linear, luminance_weights);
        float luma_nw = dot(north_west, luminance_weights);
        float luma_ne = dot(north_east, luminance_weights);
        float luma_sw = dot(south_west, luminance_weights);
        float luma_se = dot(south_east, luminance_weights);
        vec2 direction = vec2(-((luma_nw + luma_ne) - (luma_sw + luma_se)),
                               (luma_nw + luma_sw) - (luma_ne + luma_se));
        float direction_reduce = max((luma_nw + luma_ne + luma_sw + luma_se) * 0.03125, 0.0078125);
        float reciprocal_minimum = 1.0 / (min(abs(direction.x), abs(direction.y)) + direction_reduce);
        direction = clamp(direction * reciprocal_minimum, vec2(-8.0), vec2(8.0)) * texel;
        vec3 sample_a = 0.5 *
                        (arc_output_linear(in_uv + direction * (1.0 / 3.0 - 0.5), exposure_multiplier) +
                         arc_output_linear(in_uv + direction * (2.0 / 3.0 - 0.5), exposure_multiplier));
        vec3 sample_b = sample_a * 0.5 +
                        0.25 * (arc_output_linear(in_uv + direction * -0.5, exposure_multiplier) +
                                arc_output_linear(in_uv + direction * 0.5, exposure_multiplier));
        float minimum_luma = min(luma_center, min(min(luma_nw, luma_ne), min(luma_sw, luma_se)));
        float maximum_luma = max(luma_center, max(max(luma_nw, luma_ne), max(luma_sw, luma_se)));
        float sample_b_luma = dot(sample_b, luminance_weights);
        display_linear = sample_b_luma < minimum_luma || sample_b_luma > maximum_luma ? sample_a : sample_b;
    }
    out_color = vec4(arc_linear_to_srgb(display_linear), hdr.a);
}
