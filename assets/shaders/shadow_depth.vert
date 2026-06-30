#version 450

layout(location = 0) in vec3 in_position;

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
} constants;

void main()
{
    gl_Position = constants.model_view_projection * vec4(in_position, 1.0);
}
