#version 450
#extension GL_GOOGLE_include_directive : require
#extension GL_EXT_nonuniform_qualifier : require

#include "include/gpu_scene_bindless_material.glsl"

layout(location = 0) in vec3 in_normal;
layout(location = 1) in vec3 in_world_position;
layout(location = 2) in vec4 in_color;
layout(location = 3) in vec2 in_texcoord;
layout(location = 4) in vec4 in_tangent;
layout(location = 5) in vec4 in_clip_position;
layout(location = 6) in vec4 in_previous_clip_position;
layout(location = 7) flat in uint in_object_id;
layout(location = 8) flat in uint in_material_index;
layout(location = 9) flat in uint in_material_generation;

layout(location = 0) out vec4 out_albedo;
layout(location = 1) out vec4 out_normal;
layout(location = 2) out vec4 out_material;
layout(location = 3) out vec4 out_emissive;
layout(location = 4) out vec2 out_motion;
layout(location = 5) out uint out_object_id;

void main()
{
    if (in_material_index >= material_words.length() / gpu_material_word_stride)
        discard;
    uint material_base = in_material_index * gpu_material_word_stride;
    if (material_words[material_base] != in_material_generation ||
        (material_words[material_base + 1u] & (1u << 9u)) != 0u)
        discard;

    uint flags = material_words[material_base + 1u];
    uint alpha_mode = flags & 0xfu;
    if (alpha_mode == 2u)
        discard;
    vec4 base_factor = vec4(gpu_material_float(material_base, 2u), gpu_material_float(material_base, 3u),
                            gpu_material_float(material_base, 4u), gpu_material_float(material_base, 5u));
    vec4 base_color = gpu_sample_material_texture(material_base, 0u, in_texcoord, vec4(1.0)) * base_factor * in_color;
    if (alpha_mode == 1u && base_color.a < gpu_material_float(material_base, 12u))
        discard;

    vec3 normal = normalize(in_normal);
    uint normal_descriptor = gpu_invalid_index;
    if (gpu_material_texture_valid(material_base, 2u, normal_descriptor))
    {
        vec3 tangent = normalize(in_tangent.xyz - normal * dot(normal, in_tangent.xyz));
        vec3 bitangent = normalize(cross(normal, tangent) * in_tangent.w);
        vec3 mapped = texture(gpu_textures[nonuniformEXT(normal_descriptor)], in_texcoord).xyz * 2.0 - 1.0;
        mapped.xy *= gpu_material_float(material_base, 13u);
        normal = normalize(mat3(tangent, bitangent, normal) * mapped);
    }
    vec4 mr = gpu_sample_material_texture(material_base, 1u, in_texcoord, vec4(1.0));
    float metallic = clamp(gpu_material_float(material_base, 10u) * mr.b, 0.0, 1.0);
    float roughness = clamp(gpu_material_float(material_base, 11u) * mr.g, 0.04, 1.0);
    float ao = gpu_sample_material_texture(material_base, 3u, in_texcoord, vec4(1.0)).r;
    vec3 emissive_factor = vec3(gpu_material_float(material_base, 6u), gpu_material_float(material_base, 7u),
                                gpu_material_float(material_base, 8u));
    vec3 emissive = gpu_sample_material_texture(material_base, 4u, in_texcoord, vec4(1.0)).rgb * emissive_factor *
                    gpu_material_float(material_base, 9u);
    vec2 current_ndc = in_clip_position.xy / max(in_clip_position.w, 1.0e-6);
    vec2 previous_ndc = in_previous_clip_position.xy / max(in_previous_clip_position.w, 1.0e-6);

    out_albedo = base_color;
    out_normal = vec4(normal * 0.5 + 0.5, ao);
    out_material = vec4(metallic, roughness, 1.0, 0.0);
    out_emissive = vec4(emissive, 1.0);
    out_motion = (current_ndc - previous_ndc) * 0.5;
    out_object_id = in_object_id;
}
