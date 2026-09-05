#version 450
#extension GL_GOOGLE_include_directive : require
#extension GL_EXT_nonuniform_qualifier : require

#include "include/gpu_scene_bindless_material.glsl"

layout(location = 0) in vec3 in_normal;
layout(location = 1) in vec3 in_world_position;
layout(location = 2) in vec4 in_color;
layout(location = 3) in vec2 in_texcoord;
layout(location = 4) in vec4 in_tangent;
layout(location = 8) flat in uint in_material_index;
layout(location = 9) flat in uint in_material_generation;
layout(location = 0) out vec4 out_color;

void main()
{
    if (in_material_index >= material_words.length() / gpu_material_word_stride)
        discard;
    uint material_base = in_material_index * gpu_material_word_stride;
    if (material_words[material_base] != in_material_generation ||
        (material_words[material_base + 1u] & (1u << 9u)) != 0u ||
        (material_words[material_base + 1u] & 0xfu) != 2u)
        discard;
    vec4 base_factor = vec4(gpu_material_float(material_base, 2u), gpu_material_float(material_base, 3u),
                            gpu_material_float(material_base, 4u), gpu_material_float(material_base, 5u));
    vec4 base_color = gpu_sample_material_texture(material_base, 0u, in_texcoord, vec4(1.0)) * base_factor * in_color;
    vec3 normal = normalize(in_normal);
    vec3 light_direction = normalize(vec3(0.35, 0.85, 0.40));
    float diffuse = max(dot(normal, light_direction), 0.0);
    vec3 emissive_factor = vec3(gpu_material_float(material_base, 6u), gpu_material_float(material_base, 7u),
                                gpu_material_float(material_base, 8u));
    vec3 emissive = gpu_sample_material_texture(material_base, 4u, in_texcoord, vec4(1.0)).rgb * emissive_factor *
                    gpu_material_float(material_base, 9u);
    out_color = vec4(base_color.rgb * (0.18 + diffuse * 0.82) + emissive, base_color.a);
}
