layout(std430, set = 0, binding = 2) readonly buffer gpu_material_buffer
{
    uint material_words[];
};
layout(std430, set = 0, binding = 3) readonly buffer gpu_texture_buffer
{
    uint texture_words[];
};
layout(set = 0, binding = 4) uniform sampler2D gpu_textures[];

const uint gpu_invalid_index = 0xffffffffu;
const uint gpu_material_word_stride = 40u;
const uint gpu_texture_word_stride = 8u;

float gpu_material_float(uint base, uint word)
{
    return uintBitsToFloat(material_words[base + word]);
}

bool gpu_material_texture_valid(uint material_base, uint slot, out uint descriptor_index)
{
    uint texture_index = material_words[material_base + 14u + slot];
    uint generation = material_words[material_base + 26u + slot];
    if (texture_index == gpu_invalid_index || texture_index >= texture_words.length() / gpu_texture_word_stride)
        return false;
    uint texture_base = texture_index * gpu_texture_word_stride;
    descriptor_index = texture_words[texture_base + 1u];
    return texture_words[texture_base] == generation && descriptor_index != gpu_invalid_index;
}

vec4 gpu_sample_material_texture(uint material_base, uint slot, vec2 uv, vec4 fallback_value)
{
    uint descriptor_index = gpu_invalid_index;
    if (!gpu_material_texture_valid(material_base, slot, descriptor_index))
        return fallback_value;
    return texture(gpu_textures[nonuniformEXT(descriptor_index)], uv);
}
