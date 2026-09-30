#version 450
#extension GL_GOOGLE_include_directive : require
#define ARC_VIRTUAL_GEOMETRY_HARDWARE 1
#include "include/virtual_geometry_raster.glsl"

layout(location = 0) flat out uint output_visible_index;
layout(location = 1) flat out uint output_triangle_index;

void main()
{
    uint visible_index = gl_InstanceIndex;
    virtual_visible_cluster visible = visible_clusters[visible_index];
    virtual_cluster cluster = clusters[visible.cluster_index];
    virtual_page page = pages[cluster.page_index];
    uint triangle = uint(gl_VertexIndex) / 3u;
    uint corner = uint(gl_VertexIndex) % 3u;
    uint vertex_index = load_triangle_index(cluster, page, triangle, corner);
    vec3 position = decode_position(cluster, page, vertex_index);
    gl_Position = constants.view_projection * transforms[visible.instance_index].model * vec4(position, 1.0);
    output_visible_index = visible_index;
    output_triangle_index = triangle;
}
