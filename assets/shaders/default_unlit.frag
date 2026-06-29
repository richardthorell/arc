#version 450

layout(location = 0) in vec3 in_normal;
layout(location = 1) in vec3 in_world_position;
layout(location = 2) in vec4 in_color;
layout(location = 3) in vec2 in_texcoord;

layout(location = 0) out vec4 out_color;

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
    vec3 normal = normalize(in_normal);
    vec3 light_dir = normalize(-constants.light_direction_intensity.xyz);
    vec3 view_dir = normalize(-in_world_position);
    vec3 half_dir = normalize(light_dir + view_dir);

    float diffuse = max(dot(normal, light_dir), 0.0);
    float specular = pow(max(dot(normal, half_dir), 0.0), 32.0);
    float intensity = constants.light_direction_intensity.w;
    vec3 light = constants.light_color.rgb * intensity;
    vec3 ambient = vec3(0.12);
    float metallic = clamp(constants.visualization.y, 0.0, 1.0);
    float roughness = clamp(constants.visualization.z, 0.04, 1.0);
    vec3 dielectric_f0 = vec3(0.04);
    vec3 f0 = mix(dielectric_f0, in_color.rgb, metallic);
    vec3 diffuse_color = in_color.rgb * (1.0 - metallic);
    float rough_specular = pow(max(dot(normal, half_dir), 0.0), mix(96.0, 8.0, roughness));
    vec3 lit_color = diffuse_color * (ambient + diffuse * light) + f0 * rough_specular * light;

    int mode = int(constants.visualization.x + 0.5);
    vec3 color = lit_color;
    if (mode == 1)
        color = in_color.rgb;
    else if (mode == 2)
        color = vec3(in_color.a);
    else if (mode == 3)
        color = normal * 0.5 + vec3(0.5);
    else if (mode == 4)
        color = f0;
    else if (mode == 5)
        color = vec3(1.0 - roughness);
    else if (mode == 6)
        color = vec3(metallic);
    else if (mode == 7)
        color = vec3(1.0);
    else if (mode == 8)
        color = vec3(0.0);
    else if (mode == 9)
        color = vec3(diffuse);
    else if (mode == 10)
        color = vec3(in_texcoord, 0.0);
    out_color = vec4(color, in_color.a);
}
