#ifndef ARC_COLOR_GLSL
#define ARC_COLOR_GLSL

vec3 arc_linear_to_srgb(vec3 linear_color)
{
    vec3 low = linear_color * 12.92;
    vec3 high = 1.055 * pow(max(linear_color, vec3(0.0)), vec3(1.0 / 2.4)) - 0.055;
    return mix(high, low, lessThanEqual(linear_color, vec3(0.0031308)));
}

vec3 arc_srgb_to_linear(vec3 srgb_color)
{
    vec3 low = srgb_color / 12.92;
    vec3 high = pow((max(srgb_color, vec3(0.0)) + 0.055) / 1.055, vec3(2.4));
    return mix(high, low, lessThanEqual(srgb_color, vec3(0.04045)));
}

vec3 arc_aces_fitted(vec3 scene_linear)
{
    const mat3 input_transform = mat3(
        0.59719, 0.35458, 0.04823,
        0.07600, 0.90834, 0.01566,
        0.02840, 0.13383, 0.83777);
    const mat3 output_transform = mat3(
         1.60475, -0.53108, -0.07367,
        -0.10208,  1.10813, -0.00605,
        -0.00327, -0.07276,  1.07602);
    vec3 color = input_transform * max(scene_linear, vec3(0.0));
    vec3 numerator = color * (color + 0.0245786) - 0.000090537;
    vec3 denominator = color * (0.983729 * color + 0.4329510) + 0.238081;
    return clamp(output_transform * (numerator / denominator), 0.0, 1.0);
}

#endif
