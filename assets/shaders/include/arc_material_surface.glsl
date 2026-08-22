#ifndef ARC_MATERIAL_SURFACE_GLSL
#define ARC_MATERIAL_SURFACE_GLSL

#include "arc_pbr.glsl"

const uint ARC_MATERIAL_ABI_VERSION = 1u;

struct arc_material_inputs
{
    vec3 position_ws;
    vec3 normal_ws;
    vec4 tangent_ws;
    vec2 uv0;
    vec2 uv1;
    vec4 vertex_color;
    vec3 view_ws;
};

struct arc_material_surface
{
    vec3 base_color;
    float metallic;
    float roughness;
    vec3 normal_ws;
    vec3 clear_coat_normal_ws;
    vec3 tangent_ws;
    float ambient_occlusion;
    vec3 emissive_radiance;
    float opacity;
    float alpha_cutoff;
    float index_of_refraction;
    float clear_coat;
    float clear_coat_roughness;
    float sheen;
    vec3 sheen_color;
    float sheen_roughness;
    float anisotropy;
    float anisotropy_rotation;
    float transmission;
    float thickness;
    vec3 attenuation_color;
    float attenuation_distance;
    vec3 subsurface_color;
    float subsurface;
};

arc_material_surface arc_default_material_surface(arc_material_inputs input)
{
    arc_material_surface surface;
    surface.base_color = vec3(0.8);
    surface.metallic = 0.0;
    surface.roughness = 0.6;
    surface.normal_ws = normalize(input.normal_ws);
    surface.clear_coat_normal_ws = surface.normal_ws;
    surface.tangent_ws = input.tangent_ws.xyz;
    surface.ambient_occlusion = 1.0;
    surface.emissive_radiance = vec3(0.0);
    surface.opacity = 1.0;
    surface.alpha_cutoff = 0.5;
    surface.index_of_refraction = 1.5;
    surface.clear_coat = 0.0;
    surface.clear_coat_roughness = 0.1;
    surface.sheen = 0.0;
    surface.sheen_color = vec3(0.0);
    surface.sheen_roughness = 0.5;
    surface.anisotropy = 0.0;
    surface.anisotropy_rotation = 0.0;
    surface.transmission = 0.0;
    surface.thickness = 0.0;
    surface.attenuation_color = vec3(1.0);
    surface.attenuation_distance = 1.0;
    surface.subsurface_color = vec3(1.0, 0.35, 0.2);
    surface.subsurface = 0.0;
    return surface;
}

arc_surface_data arc_material_to_pbr_surface(arc_material_surface material)
{
    arc_surface_data surface;
    surface.base_color = material.base_color;
    surface.normal = material.normal_ws;
    surface.clear_coat_normal = material.clear_coat_normal_ws;
    surface.tangent = material.tangent_ws;
    surface.emissive = material.emissive_radiance;
    surface.metallic = material.metallic;
    surface.perceptual_roughness = material.roughness;
    surface.occlusion = material.ambient_occlusion;
    surface.clear_coat = material.clear_coat;
    surface.clear_coat_roughness = material.clear_coat_roughness;
    surface.anisotropy = material.anisotropy;
    return surface;
}

#endif
