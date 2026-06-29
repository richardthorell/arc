#version 450

layout(location = 0) in vec3 in_position;
layout(location = 1) in vec3 in_normal;
layout(location = 2) in vec2 in_texcoord;
layout(location = 3) in vec4 in_color;

layout(location = 0) out vec3 out_normal;
layout(location = 1) out vec3 out_world_position;
layout(location = 2) out vec4 out_color;
layout(location = 3) out vec2 out_texcoord;

layout(push_constant) uniform mesh_constants
{
    mat4 model_view_projection;
    mat4 model;
    vec4 base_color;
    vec4 light_direction_intensity;
    vec4 light_color;
    vec4 visualization;
} constants;

void main()
{
    vec4 world_position = constants.model * vec4(in_position, 1.0);
    out_normal = mat3(constants.model) * in_normal;
    out_world_position = world_position.xyz;
    out_color = in_color * constants.base_color;
    out_texcoord = in_texcoord;
    gl_Position = constants.model_view_projection * vec4(in_position, 1.0);
}
