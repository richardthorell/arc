#version 450

layout(location = 0) flat in uint input_visible_index;
layout(location = 1) flat in uint input_triangle_index;
layout(location = 0) out uint output_encoded_depth;
layout(location = 1) out uint output_visibility_id;

void main()
{
    output_encoded_depth = floatBitsToUint(clamp(gl_FragCoord.z, 0.0, 1.0));
    output_visibility_id = (input_visible_index << 8u) | min(input_triangle_index, 255u);
}
