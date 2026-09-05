struct virtual_visible_cluster
{
    uint instance_index;
    uint resource_index;
    uint cluster_index;
    uint page_index;
    uint material_index;
    uint hierarchy_level;
    uint flags;
    float view_depth;
};

struct virtual_cluster
{
    vec4 sphere;
    vec4 normal_cone;
    vec4 bounds_min_error;
    vec4 bounds_max;
    uint page_index;
    uint page_byte_offset;
    uint vertex_count;
    uint triangle_count;
    uint material_section;
    uint hierarchy_node;
    uint flags;
    uint reserved;
};

struct virtual_page
{
    uint heap_index;
    uint heap_byte_offset;
    uint stored_size;
    uint decoded_size;
    uint resource_generation;
    uint flags;
    uint last_used_frame;
    uint reserved;
};

struct gpu_scene_transform
{
    mat4 model;
    mat4 previous_model;
};

struct virtual_raster_bin
{
    uint visible_index;
    uint minimum_tile_x;
    uint minimum_tile_y;
    uint maximum_tile_x;
    uint maximum_tile_y;
    uint flags;
    uint reserved0;
    uint reserved1;
};

layout(std430, set = 0, binding = 0) readonly buffer visible_buffer
{
    virtual_visible_cluster visible_clusters[];
};
layout(std430, set = 0, binding = 1) readonly buffer cluster_buffer
{
    virtual_cluster clusters[];
};
layout(std430, set = 0, binding = 2) readonly buffer transform_buffer
{
    gpu_scene_transform transforms[];
};
layout(std430, set = 0, binding = 3) readonly buffer page_buffer
{
    virtual_page pages[];
};
layout(std430, set = 0, binding = 4) readonly buffer page_heap_buffer
{
    uint page_words[];
};
layout(std430, set = 0, binding = 5) buffer traversal_counter_buffer
{
    uint visible_count;
    uint request_count;
    uint frustum_rejected;
    uint cone_rejected;
    uint hzb_rejected;
    uint projected_size_rejected;
    uint visible_overflow;
    uint request_overflow;
    uint fallback_instances;
    uint parent_fallbacks;
    uint traversal_overflow;
    uint bin_count;
} counters;
layout(std430, set = 0, binding = 6) buffer bin_buffer
{
    virtual_raster_bin bins[];
};
layout(set = 0, binding = 7, r32ui) uniform uimage2D encoded_depth;
layout(set = 0, binding = 8, r32ui) uniform uimage2D visibility_ids;

layout(push_constant) uniform raster_constants
{
    mat4 view_projection;
    uvec4 viewport_capacities;
} constants;

uint load_byte(uint byte_offset)
{
    uint word = page_words[byte_offset >> 2u];
    return (word >> ((byte_offset & 3u) * 8u)) & 0xffu;
}

uint load_u16(uint byte_offset)
{
    return load_byte(byte_offset) | (load_byte(byte_offset + 1u) << 8u);
}

vec3 decode_position(virtual_cluster cluster, virtual_page page, uint vertex_index)
{
    uint vertex_offset = page.heap_byte_offset + cluster.page_byte_offset + 16u + vertex_index * 24u;
    vec3 quantized = vec3(load_u16(vertex_offset), load_u16(vertex_offset + 2u), load_u16(vertex_offset + 4u));
    return mix(cluster.bounds_min_error.xyz, cluster.bounds_max.xyz, quantized / 65535.0);
}

uint load_triangle_index(virtual_cluster cluster, virtual_page page, uint triangle_index, uint corner)
{
    uint index_offset = page.heap_byte_offset + cluster.page_byte_offset + 16u + cluster.vertex_count * 24u +
                        triangle_index * 3u + corner;
    return load_byte(index_offset);
}

vec2 screen_position(vec4 clip)
{
    vec2 ndc = clip.xy / max(abs(clip.w), 1.0e-6);
    vec2 normalized = vec2(ndc.x * 0.5 + 0.5, 0.5 - ndc.y * 0.5);
    return normalized * vec2(constants.viewport_capacities.xy);
}

float edge_function(vec2 a, vec2 b, vec2 point)
{
    return (point.x - a.x) * (b.y - a.y) - (point.y - a.y) * (b.x - a.x);
}
