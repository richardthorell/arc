#version 450

layout(location = 0) in vec3 in_position;
layout(location = 1) in vec3 in_normal;
layout(location = 2) in vec2 in_texcoord;
layout(location = 3) in vec4 in_color;
layout(location = 4) in vec4 in_tangent;

layout(location = 0) out vec2 out_texcoord;
layout(location = 1) out vec4 out_color;
layout(location = 2) flat out uint out_material_index;
layout(location = 3) flat out uint out_material_generation;

struct gpu_scene_instance
{
    vec4 bounds_min;
    vec4 bounds_max;
    uvec4 geometry;
    uvec4 material_flags;
    uvec4 draw_metadata;
    vec2 distance_error;
    uvec2 material_attribute;
};
struct gpu_scene_transform
{
    mat4 model;
    mat4 previous_model;
};

layout(std430, set = 0, binding = 0) readonly buffer gpu_scene_buffer
{
    gpu_scene_instance instances[];
};
layout(std430, set = 0, binding = 1) readonly buffer gpu_transform_buffer
{
    gpu_scene_transform transforms[];
};

layout(push_constant) uniform virtual_shadow_page_constants
{
    float world_to_page_clip[16];
} constants;

vec4 transform_page(vec3 world_position)
{
    vec4 position = vec4(world_position, 1.0);
    return vec4(dot(vec4(constants.world_to_page_clip[0], constants.world_to_page_clip[1],
                         constants.world_to_page_clip[2], constants.world_to_page_clip[3]), position),
                dot(vec4(constants.world_to_page_clip[4], constants.world_to_page_clip[5],
                         constants.world_to_page_clip[6], constants.world_to_page_clip[7]), position),
                dot(vec4(constants.world_to_page_clip[8], constants.world_to_page_clip[9],
                         constants.world_to_page_clip[10], constants.world_to_page_clip[11]), position),
                dot(vec4(constants.world_to_page_clip[12], constants.world_to_page_clip[13],
                         constants.world_to_page_clip[14], constants.world_to_page_clip[15]), position));
}

void main()
{
    uint instance_index = gl_InstanceIndex;
    gpu_scene_instance instance = instances[instance_index];
    vec4 world = transforms[instance_index].model * vec4(in_position, 1.0);
    gl_Position = transform_page(world.xyz);
    out_texcoord = in_texcoord;
    out_color = in_color;
    out_material_index = instance.material_flags.x;
    out_material_generation = instance.material_flags.y;
}
