#version 450

layout(set = 1, binding = 0, std430) readonly buffer TerrainHeights { float heights[]; } terrain_heights;
layout(set = 1, binding = 1, std430) readonly buffer TerrainWeights { uint weights[]; } terrain_weights;
layout(set = 1, binding = 2, std140) uniform TerrainResource { uvec4 dimensions; vec4 extent; } terrain;

layout(location = 0) out vec3 out_normal;
layout(location = 1) out vec3 out_world_position;
layout(location = 2) out vec4 out_weights;
layout(location = 3) out vec2 out_texcoord;
layout(location = 4) out float out_view_depth;
layout(location = 5) out vec4 out_tangent;

layout(push_constant) uniform mesh_constants {
    mat4 model_view_projection; mat4 model; vec4 patch_samples; vec4 light_direction_intensity;
    vec4 light_color; vec4 camera_position; vec4 visualization; vec4 fog_color_density;
    vec4 fog_params; vec4 material_params;
} constants;

float sample_height(ivec2 sample_coord)
{
    int resolution = int(terrain.dimensions.x);
    sample_coord = clamp(sample_coord, ivec2(0), ivec2(resolution - 1));
    return terrain_heights.heights[sample_coord.y * resolution + sample_coord.x];
}

void main()
{
    uint patch_quads = terrain.dimensions.y;
    uint row = patch_quads + 1u;
    uvec2 grid = uvec2(uint(gl_VertexIndex) % row, uint(gl_VertexIndex) / row);
    vec2 uv = vec2(grid) / float(patch_quads);
    vec2 source_sample = mix(constants.patch_samples.xy, constants.patch_samples.zw, uv);
    ivec2 sample_coord = ivec2(round(source_sample));
    float height = sample_height(sample_coord);
    float half_width = terrain.extent.x * 0.5;
    float half_depth = terrain.extent.y * 0.5;
    float source_quads = float(terrain.dimensions.x - 1u);
    vec3 local_position = vec3(-half_width + terrain.extent.x * source_sample.x / source_quads,
                              height,
                              -half_depth + terrain.extent.y * source_sample.y / source_quads);
    float spacing_x = terrain.extent.x / source_quads;
    float spacing_z = terrain.extent.y / source_quads;
    vec3 local_normal = normalize(vec3(sample_height(sample_coord + ivec2(-1, 0)) -
                                       sample_height(sample_coord + ivec2(1, 0)),
                                       2.0 * max(spacing_x, spacing_z),
                                       sample_height(sample_coord + ivec2(0, -1)) -
                                       sample_height(sample_coord + ivec2(0, 1))));
    vec4 world_position = constants.model * vec4(local_position, 1.0);
    mat3 normal_matrix = transpose(inverse(mat3(constants.model)));
    uint packed = terrain_weights.weights[sample_coord.y * int(terrain.dimensions.x) + sample_coord.x];
    out_weights = vec4(float(packed & 255u), float((packed >> 8u) & 255u),
                       float((packed >> 16u) & 255u), float((packed >> 24u) & 255u)) / 255.0;
    out_normal = normalize(normal_matrix * local_normal);
    out_world_position = world_position.xyz;
    out_texcoord = source_sample / source_quads;
    out_view_depth = length(constants.camera_position.xyz - world_position.xyz);
    out_tangent = vec4(normalize(mat3(constants.model) * vec3(1.0, 0.0, 0.0)), 1.0);
    gl_Position = constants.model_view_projection * vec4(local_position, 1.0);
}
