#version 450

layout(set = 1, binding = 0, std430) readonly buffer TerrainHeights { float heights[]; } terrain_heights;
layout(set = 1, binding = 2, std140) uniform TerrainResource { uvec4 dimensions; vec4 extent; } terrain;
struct TerrainPatch { uvec4 samples; uvec4 metadata; };
layout(set = 1, binding = 3, std430) readonly buffer TerrainPatches { TerrainPatch patches[]; } terrain_patches;

layout(push_constant) uniform mesh_constants
{
    mat4 model_view_projection;
    mat4 model;
    vec4 patch_samples;
    vec4 light_direction_intensity;
    vec4 light_color;
    vec4 camera_position;
    vec4 visualization;
    vec4 fog_color_density;
    vec4 fog_params;
    vec4 material_params;
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
    vec4 patch_samples = constants.patch_samples;
    if (patch_samples.x < 0.0)
    {
        TerrainPatch patch_record = terrain_patches.patches[gl_InstanceIndex];
        uint stitch_mask = patch_record.metadata.z;
        if ((stitch_mask & 1u) != 0u && grid.x == 0u && (grid.y & 1u) != 0u) --grid.y;
        if ((stitch_mask & 2u) != 0u && grid.x == patch_quads && (grid.y & 1u) != 0u) --grid.y;
        if ((stitch_mask & 4u) != 0u && grid.y == 0u && (grid.x & 1u) != 0u) --grid.x;
        if ((stitch_mask & 8u) != 0u && grid.y == patch_quads && (grid.x & 1u) != 0u) --grid.x;
        patch_samples = vec4(patch_record.samples);
    }
    vec2 patch_uv = vec2(grid) / float(patch_quads);
    vec2 source_sample = mix(patch_samples.xy, patch_samples.zw, patch_uv);
    float source_quads = float(terrain.dimensions.x - 1u);
    vec3 local_position = vec3(-terrain.extent.x * 0.5 + terrain.extent.x * source_sample.x / source_quads,
                               sample_height(ivec2(round(source_sample))),
                               -terrain.extent.y * 0.5 + terrain.extent.y * source_sample.y / source_quads);
    gl_Position = constants.model_view_projection * vec4(local_position, 1.0);
}
