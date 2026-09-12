#version 450
#extension GL_GOOGLE_include_directive : require

#include "include/arc_color.glsl"

layout(location = 0) in vec2 in_uv;
layout(location = 0) out vec4 out_color;

layout(set = 0, binding = 0) uniform sampler2D scene_color;
layout(set = 0, binding = 2) uniform sampler2D selection_mask;
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

const vec3 outline_accent = vec3(1.0, 0.48, 0.04);

vec3 arc_output_linear(vec2 uv, float exposure_multiplier)
{
    vec3 exposed = texture(scene_color, uv).rgb * max(exposure_multiplier, 0.0);
    return arc_aces_fitted(exposed);
}

float selection_outline()
{
    ivec2 extent = textureSize(selection_mask, 0);
    ivec2 pixel = clamp(ivec2(gl_FragCoord.xy), ivec2(0), extent - ivec2(1));
    float center = texelFetch(selection_mask, pixel, 0).r;
    float neighbor = 0.0;
    for (int y = -2; y <= 2; ++y)
        for (int x = -2; x <= 2; ++x)
        {
            if (x == 0 && y == 0) continue;
            ivec2 sample_pixel = clamp(pixel + ivec2(x, y), ivec2(0), extent - ivec2(1));
            neighbor = max(neighbor, texelFetch(selection_mask, sample_pixel, 0).r);
        }
    return max(0.0, neighbor - center);
}

void main()
{
    vec4 hdr = texture(scene_color, in_uv);
    if (constants.exposure_output.w > 0.5)
    {
        vec3 display_color = arc_linear_to_srgb(clamp(hdr.rgb, vec3(0.0), vec3(1.0)));
        display_color = mix(display_color, arc_linear_to_srgb(outline_accent), selection_outline());
        out_color = vec4(display_color, hdr.a);
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
    // ARC's accent is applied after scene anti-aliasing so the silhouette
    // remains a stable two output pixels regardless of camera distance.
    display_linear = mix(display_linear, outline_accent, selection_outline());
    out_color = vec4(arc_linear_to_srgb(display_linear), hdr.a);
}
