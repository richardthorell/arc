#version 450

layout(location = 0) in vec3 in_normal;
layout(location = 1) in vec3 in_world_position;
layout(location = 2) in vec4 in_color;
layout(location = 3) in vec2 in_texcoord;
layout(location = 4) in float in_view_depth;

layout(location = 0) out vec4 out_color;

layout(set = 0, binding = 0) uniform sampler2D base_texture;
layout(set = 0, binding = 1) uniform sampler2DArrayShadow shadow_map;

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

layout(set = 0, binding = 2) uniform shadow_data
{
    mat4 light_view_projection[4];
    vec4 cascade_splits;
    vec4 params;
} shadows;

float sample_shadow(vec3 world_position)
{
    if (shadows.params.x <= 0.0)
        return 1.0;

    int cascade = 0;
    if (in_view_depth > shadows.cascade_splits.x)
        cascade = 1;
    if (in_view_depth > shadows.cascade_splits.y)
        cascade = 2;
    if (in_view_depth > shadows.cascade_splits.z)
        cascade = 3;

    vec4 shadow_position = shadows.light_view_projection[cascade] * vec4(world_position, 1.0);
    vec3 projected = shadow_position.xyz / shadow_position.w;
    vec2 uv = projected.xy * 0.5 + vec2(0.5);
    float depth = projected.z;
    if (uv.x < 0.0 || uv.y < 0.0 || uv.x > 1.0 || uv.y > 1.0 || depth < 0.0 || depth > 1.0)
        return 1.0;

    float radius = shadows.params.z;
    float bias = shadows.params.y;
    vec2 texel = vec2(1.0 / float(textureSize(shadow_map, 0).x));
    float visibility = 0.0;
    float samples = 0.0;
    for (float y = -radius; y <= radius; y += 1.0)
    {
        for (float x = -radius; x <= radius; x += 1.0)
        {
            visibility += texture(shadow_map, vec4(uv + vec2(x, y) * texel, float(cascade), depth - bias));
            samples += 1.0;
        }
    }
    visibility = samples > 0.0 ? visibility / samples : texture(shadow_map, vec4(uv, float(cascade), depth - bias));
    return mix(1.0 - shadows.params.x, 1.0, visibility);
}

vec3 apply_height_fog(vec3 color)
{
    float density = constants.fog_color_density.w;
    if (density <= 0.0)
        return color;

    float distance_from_camera = length(constants.camera_position.xyz - in_world_position);
    float start_distance = max(constants.fog_params.x, 0.0);
    float height_falloff = max(constants.fog_params.y, 0.0);
    float max_opacity = clamp(constants.fog_params.z, 0.0, 1.0);
    float sun_scattering = max(constants.fog_params.w, 0.0);

    float distance_term = max(distance_from_camera - start_distance, 0.0) * density;
    float height_term = exp(-max(in_world_position.y, 0.0) * height_falloff);
    float fog_amount = clamp(1.0 - exp(-distance_term * height_term), 0.0, max_opacity);

    vec3 fog_color = constants.fog_color_density.rgb;
    vec3 light_dir = normalize(-constants.light_direction_intensity.xyz);
    vec3 view_dir = normalize(constants.camera_position.xyz - in_world_position);
    float sun_term = pow(max(dot(view_dir, light_dir), 0.0), 8.0) * sun_scattering;
    fog_color += constants.light_color.rgb * sun_term;
    return mix(color, fog_color, fog_amount);
}

void main()
{
    vec4 material_color = texture(base_texture, in_texcoord) * in_color;
    vec3 normal = normalize(in_normal);
    vec3 light_dir = normalize(-constants.light_direction_intensity.xyz);
    vec3 view_dir = normalize(constants.camera_position.xyz - in_world_position);
    vec3 half_dir = normalize(light_dir + view_dir);

    float diffuse = max(dot(normal, light_dir), 0.0);
    float roughness = clamp(constants.visualization.z, 0.04, 1.0);
    float metallic = clamp(constants.visualization.y, 0.0, 1.0);
    float spec_power = mix(96.0, 8.0, roughness);
    float specular_term = pow(max(dot(normal, half_dir), 0.0), spec_power);
    vec3 f0 = mix(vec3(0.04), material_color.rgb, metallic);
    vec3 diffuse_color = material_color.rgb * (1.0 - metallic);
    vec3 light = constants.light_color.rgb * constants.light_direction_intensity.w;
    float shadow = sample_shadow(in_world_position);
    vec3 ambient = vec3(0.12) * material_color.rgb;
    vec3 direct = (diffuse_color * diffuse + f0 * specular_term) * light * shadow;
    vec3 lit_color = ambient + direct;

    int mode = int(constants.visualization.x + 0.5);
    vec3 color = lit_color;
    if (mode == 1)
        color = material_color.rgb;
    else if (mode == 2)
        color = vec3(material_color.a);
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
        color = vec3(diffuse * shadow);
    else if (mode == 10)
        color = vec3(in_texcoord, 0.0);
    else
        color = apply_height_fog(color);

    out_color = vec4(color, material_color.a);
}
