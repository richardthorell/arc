#ifndef ARC_MATERIAL_PARAMETERS_GLSL
#define ARC_MATERIAL_PARAMETERS_GLSL

layout(set = 0, binding = 16) uniform arc_material_parameters
{
    vec4 emissive_factor;
    vec4 material_lobes;
    vec4 volume_params;
    vec4 subsurface_color_factor;
    vec4 attenuation_color;
} material_parameters;

#endif
