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
layout(location = 4) out vec4 out_tangent;
layout(location = 5) out vec4 out_clip_position;
layout(location = 6) out vec4 out_previous_clip_position;
layout(location = 7) flat out uint out_object_id;
layout(location = 8) flat out uint out_material_index;
layout(location = 9) flat out uint out_material_generation;

struct gpu_scene_instance
{
    vec4 bounds_min;
    vec4 bounds_max;
    uvec4 geometry;
    uvec4 material_flags;
    uvec4 draw_metadata;
    vec4 distance_error;
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

layout(push_constant) uniform gpu_scene_constants
{
    mat4 view_projection;
    mat4 previous_view_projection;
} constants;

void main()
{
    uint instance_index = gl_InstanceIndex;
    gpu_scene_instance instance = instances[instance_index];
    gpu_scene_transform transform = transforms[instance_index];
    vec4 world = transform.model * vec4(in_position, 1.0);
    vec3 normal = normalize(mat3(transform.model) * in_normal);
    vec3 tangent = normalize(mat3(transform.model) * in_tangent.xyz);
    out_world_position = world.xyz;
    out_normal = normal;
    out_tangent = vec4(normalize(tangent - normal * dot(normal, tangent)), in_tangent.w);
    out_color = in_color;
    out_texcoord = in_texcoord;
    out_clip_position = constants.view_projection * world;
    out_previous_clip_position = constants.previous_view_projection * transform.previous_model * vec4(in_position, 1.0);
    out_object_id = instance.draw_metadata.w;
    out_material_index = instance.material_flags.x;
    out_material_generation = instance.material_flags.y;
    gl_Position = out_clip_position;
}
