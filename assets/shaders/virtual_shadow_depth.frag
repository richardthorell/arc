#version 450
#extension GL_GOOGLE_include_directive : require
#extension GL_EXT_nonuniform_qualifier : require

#include "include/gpu_scene_bindless_material.glsl"

layout(location = 0) in vec2 in_texcoord;
layout(location = 1) in vec4 in_color;
layout(location = 2) flat in uint in_material_index;
layout(location = 3) flat in uint in_material_generation;

float arc_shadow_noise(vec2 pixel)
{
    return fract(52.9829189 * fract(dot(pixel, vec2(0.06711056, 0.00583715))));
}

void main()
{
    if (in_material_index >= material_words.length() / gpu_material_word_stride)
        discard;
    uint material_base = in_material_index * gpu_material_word_stride;
    if (material_words[material_base] != in_material_generation ||
        (material_words[material_base + 1u] & (1u << 9u)) != 0u)
        discard;

    uint alpha_mode = material_words[material_base + 1u] & 0xfu;
    if (alpha_mode == 0u)
        return;
    vec4 base_factor = vec4(gpu_material_float(material_base, 2u), gpu_material_float(material_base, 3u),
                            gpu_material_float(material_base, 4u), gpu_material_float(material_base, 5u));
    float alpha = gpu_sample_material_texture(material_base, 0u, in_texcoord, vec4(1.0)).a *
                  base_factor.a * in_color.a;
    if (alpha_mode == 1u)
    {
        if (alpha < gpu_material_float(material_base, 12u))
            discard;
        return;
    }
    if (arc_shadow_noise(gl_FragCoord.xy) > alpha)
        discard;
}
