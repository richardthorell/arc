#version 450

layout(location = 0) in vec3 in_position;
layout(location = 1) in vec3 in_normal;
layout(location = 2) in vec2 in_texcoord;
layout(location = 3) in vec4 in_color;
layout(location = 4) in vec4 in_tangent;

layout(location = 0) out vec3 out_normal;
layout(location = 1) out vec3 out_world_position;
layout(location = 2) out vec4 out_color;
layout(location = 3) out vec2 out_texcoord;
layout(location = 4) out float out_view_depth;
layout(location = 5) out vec4 out_tangent;

layout(push_constant) uniform mesh_constants
{
    mat4 model_view_projection;
    mat4 model;
    vec4 base_color;
    vec4 light_direction_intensity;
    vec4 light_color;
    vec4 camera_position;
    vec4 visualization;
    vec4 fog_color_density;
    vec4 fog_params;
    vec4 material_params;
} constants;

struct surface_sample
{
    vec4 displacement;
    vec4 normal;
    vec4 velocity;
};

layout(std140, set = 1, binding = 0) uniform water_metadata_buffer
{
    uvec4 resolutions;
    vec4 physical_lengths;
    uvec4 configuration;
} water_metadata;
layout(std430, set = 1, binding = 1) readonly buffer water_surface_buffer_0 { surface_sample values[]; } water_surface_0;
layout(std430, set = 1, binding = 2) readonly buffer water_surface_buffer_1 { surface_sample values[]; } water_surface_1;
layout(std430, set = 1, binding = 3) readonly buffer water_surface_buffer_2 { surface_sample values[]; } water_surface_2;
layout(std430, set = 1, binding = 4) readonly buffer water_surface_buffer_3 { surface_sample values[]; } water_surface_3;

surface_sample load_surface(uint cascade, uint index)
{
    if (cascade == 0u) return water_surface_0.values[index];
    if (cascade == 1u) return water_surface_1.values[index];
    if (cascade == 2u) return water_surface_2.values[index];
    return water_surface_3.values[index];
}

surface_sample sample_surface(uint cascade, vec2 world_position)
{
    uint resolution = water_metadata.resolutions[cascade];
    float physical_length = water_metadata.physical_lengths[cascade];
    vec2 coordinate = fract(world_position / physical_length) * float(resolution);
    uvec2 base = uvec2(floor(coordinate));
    vec2 weight = fract(coordinate);
    uvec2 next = (base + uvec2(1u)) % resolution;
    base %= resolution;
    surface_sample p00 = load_surface(cascade, base.y * resolution + base.x);
    surface_sample p10 = load_surface(cascade, base.y * resolution + next.x);
    surface_sample p01 = load_surface(cascade, next.y * resolution + base.x);
    surface_sample p11 = load_surface(cascade, next.y * resolution + next.x);
    surface_sample result;
    result.displacement = mix(mix(p00.displacement, p10.displacement, weight.x),
                              mix(p01.displacement, p11.displacement, weight.x), weight.y);
    result.normal = mix(mix(p00.normal, p10.normal, weight.x),
                        mix(p01.normal, p11.normal, weight.x), weight.y);
    result.velocity = mix(mix(p00.velocity, p10.velocity, weight.x),
                          mix(p01.velocity, p11.velocity, weight.x), weight.y);
    return result;
}

void main()
{
    vec4 world_position = constants.model * vec4(in_position, 1.0);
    vec3 displacement = vec3(0.0);
    vec2 combined_slope = vec2(0.0);
    float foam = 0.0;
    for (uint cascade = 0u; cascade < water_metadata.configuration.x; ++cascade)
    {
        surface_sample surface = sample_surface(cascade, world_position.xz);
        displacement += surface.displacement.xyz;
        float inverse_y = 1.0 / max(surface.normal.y, 1.0e-4);
        combined_slope += -surface.normal.xz * inverse_y;
        foam = max(foam, surface.displacement.w);
    }
    world_position.xyz += displacement;
    out_normal = normalize(vec3(-combined_slope.x, 1.0, -combined_slope.y));
    out_world_position = world_position.xyz;
    out_color = vec4(in_color.rgb * mix(1.0, 5.0, smoothstep(0.0, 1.0, foam)), in_color.a);
    out_texcoord = in_texcoord;
    out_view_depth = length(constants.camera_position.xyz - world_position.xyz);
    out_tangent = vec4(normalize(vec3(1.0, combined_slope.x, 0.0)), in_tangent.w);
    mat4 view_projection = constants.model_view_projection * inverse(constants.model);
    gl_Position = view_projection * world_position;
}
