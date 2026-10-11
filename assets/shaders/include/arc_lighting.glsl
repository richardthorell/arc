#ifndef ARC_LIGHTING_GLSL
#define ARC_LIGHTING_GLSL

#ifndef ARC_LIGHT_BUFFER_BINDING
#define ARC_LIGHT_BUFFER_BINDING 15
#endif
#ifndef ARC_LOCAL_SHADOW_BINDING
#define ARC_LOCAL_SHADOW_BINDING 17
#endif
#ifndef ARC_LTC_LOOKUP_BINDING
#define ARC_LTC_LOOKUP_BINDING 23
#endif

struct directional_light_data
{
    vec4 direction_intensity;
    vec4 color_flags;
    vec4 source_shape;
    uvec4 shadow_identity;
    uvec4 shadow_routing;
    vec4 shadow_parameters;
};

struct point_light_data
{
    vec4 position_range;
    vec4 color_intensity;
    vec4 object_id_shadow;
    vec4 shadow_parameters;
    vec4 source_shape;
};

struct spot_light_data
{
    vec4 position_range;
    vec4 direction_inner_angle;
    vec4 color_intensity;
    vec4 params;
    vec4 object_id_shadow;
    vec4 shadow_parameters;
    vec4 source_shape;
};

struct local_shadow_face_data
{
    mat4 light_view_projection;
    vec4 atlas_rect;
    vec4 parameters;
};

struct area_light_data
{
    vec4 position_shape;
    vec4 direction_two_sided;
    vec4 tangent_width;
    vec4 color_intensity;
    vec4 dimensions_shadow;
};

layout(std430, set = 0, binding = ARC_LIGHT_BUFFER_BINDING) readonly buffer scene_light_buffer
{
    directional_light_data directional_lights[4];
    point_light_data point_lights[64];
    spot_light_data spot_lights[64];
    area_light_data area_lights[32];
    local_shadow_face_data local_shadow_faces[144];
    vec4 ambient_color_intensity;
    uint directional_count;
    uint point_count;
    uint spot_count;
    uint area_count;
    uint skipped_directional_count;
    uint skipped_point_count;
    uint skipped_spot_count;
    uint skipped_area_count;
    uint local_shadow_face_count;
    uint local_shadow_padding0;
    uint local_shadow_padding1;
    uint local_shadow_padding2;
} lights;

struct arc_ltc_sample
{
    vec4 inverse_transform;
    vec4 amplitude_fresnel;
};

layout(std430, set = 0, binding = ARC_LTC_LOOKUP_BINDING) readonly buffer arc_ltc_lookup_buffer
{
    arc_ltc_sample samples[];
} arc_ltc_lookup;

layout(set = 0, binding = ARC_LOCAL_SHADOW_BINDING) uniform sampler2DShadow arc_local_shadow_atlas;

float arc_sample_local_shadow_face(uint face_index, vec3 world_position, float filter_scale)
{
    if (face_index >= min(lights.local_shadow_face_count, 144u))
        return 1.0;
    local_shadow_face_data face = lights.local_shadow_faces[face_index];
    vec4 clip = face.light_view_projection * vec4(world_position, 1.0);
    if (clip.w <= 0.0)
        return 1.0;
    vec3 projected = clip.xyz / clip.w;
    vec2 uv = projected.xy * 0.5 + 0.5;
    if (projected.z <= 0.0 || projected.z >= 1.0 ||
        any(lessThan(uv, vec2(0.0))) || any(greaterThan(uv, vec2(1.0))))
        return 1.0;
    vec2 atlas_uv = face.atlas_rect.xy + uv * face.atlas_rect.zw;
    float comparison = projected.z - face.parameters.y;
    float texel = face.parameters.x;
    float visibility = 0.0;
    for (int y = -1; y <= 1; ++y)
        for (int x = -1; x <= 1; ++x)
            visibility += texture(
                arc_local_shadow_atlas,
                vec3(atlas_uv + vec2(x, y) * texel * max(filter_scale, 1.0), comparison));
    return visibility / 9.0;
}

uint arc_point_shadow_face(vec3 direction)
{
    vec3 absolute_direction = abs(direction);
    if (absolute_direction.x >= absolute_direction.y && absolute_direction.x >= absolute_direction.z)
        return direction.x >= 0.0 ? 0u : 1u;
    if (absolute_direction.y >= absolute_direction.z)
        return direction.y >= 0.0 ? 2u : 3u;
    return direction.z >= 0.0 ? 4u : 5u;
}

float arc_point_shadow_visibility(point_light_data light, vec3 world_position)
{
    int first_face = int(light.shadow_parameters.x + 0.5);
    if (first_face < 0 || light.shadow_parameters.y < 5.5)
        return 1.0;
    uint face = arc_point_shadow_face(world_position - light.position_range.xyz);
    float effective_source_radius = max(max(light.source_shape.x, 0.0), 0.5 * max(light.source_shape.y, 0.0));
    float filter_scale = 1.0 + 12.0 * effective_source_radius / max(light.position_range.w, 1.0e-4);
    float sampled = arc_sample_local_shadow_face(uint(first_face) + face, world_position, filter_scale);
    return mix(1.0, sampled, clamp(light.shadow_parameters.z, 0.0, 1.0));
}

float arc_spot_shadow_visibility(spot_light_data light, vec3 world_position)
{
    int face = int(light.shadow_parameters.x + 0.5);
    if (face < 0)
        return 1.0;
    float filter_scale = 1.0 + 12.0 * max(light.source_shape.x, 0.0) / max(light.position_range.w, 1.0e-4);
    float sampled = arc_sample_local_shadow_face(uint(face), world_position, filter_scale);
    return mix(1.0, sampled, clamp(light.shadow_parameters.z, 0.0, 1.0));
}

float arc_source_angular_radius(float source_radius, float distance_to_light)
{
    if (source_radius <= 0.0)
        return 0.0;
    return asin(clamp(source_radius / max(distance_to_light, source_radius), 0.0, 1.0));
}

arc_surface_data arc_surface_for_extended_source(arc_surface_data surface, float angular_radius)
{
    // A finite emitter occupies a cone of incident directions. Approximate that
    // convolution by broadening the microfacet lobe while preserving the exact
    // punctual-light result when the authored source size is zero.
    float source_roughness = clamp(angular_radius * (2.0 / ARC_PI), 0.0, 1.0);
    surface.perceptual_roughness = max(surface.perceptual_roughness, source_roughness);
    surface.clear_coat_roughness = max(surface.clear_coat_roughness, source_roughness);
    return surface;
}

vec3 arc_evaluate_surface_light(
    arc_surface_data surface,
    vec3 view_direction,
    vec3 light_direction,
    vec3 radiance,
    float visibility)
{
    arc_brdf_result brdf = arc_evaluate_brdf(surface, view_direction, light_direction);
    return (brdf.diffuse + brdf.specular) * radiance *
        max(dot(surface.normal, light_direction), 0.0) * visibility;
}


arc_ltc_sample arc_sample_ltc(float perceptual_roughness, float n_dot_v)
{
    const uint resolution = 32u;
    float x = clamp(perceptual_roughness, 0.0, 1.0) * float(resolution - 1u);
    float y = sqrt(clamp(1.0 - n_dot_v, 0.0, 1.0)) * float(resolution - 1u);
    uvec2 p0 = uvec2(uint(floor(x)), uint(floor(y)));
    uvec2 p1 = min(p0 + uvec2(1u), uvec2(resolution - 1u));
    vec2 f = fract(vec2(x, y));

    arc_ltc_sample s00 = arc_ltc_lookup.samples[p0.y * resolution + p0.x];
    arc_ltc_sample s10 = arc_ltc_lookup.samples[p0.y * resolution + p1.x];
    arc_ltc_sample s01 = arc_ltc_lookup.samples[p1.y * resolution + p0.x];
    arc_ltc_sample s11 = arc_ltc_lookup.samples[p1.y * resolution + p1.x];

    arc_ltc_sample result;
    result.inverse_transform =
        mix(mix(s00.inverse_transform, s10.inverse_transform, f.x),
            mix(s01.inverse_transform, s11.inverse_transform, f.x), f.y);
    result.amplitude_fresnel =
        mix(mix(s00.amplitude_fresnel, s10.amplitude_fresnel, f.x),
            mix(s01.amplitude_fresnel, s11.amplitude_fresnel, f.x), f.y);
    return result;
}

mat3 arc_ltc_inverse_matrix(vec4 packed)
{
    // packed = (m11, m13, m31, m33), matching the CPU lookup contract.
    return mat3(
        packed.x, 0.0, packed.z,
        0.0,      1.0, 0.0,
        packed.y, 0.0, packed.w);
}

vec3 arc_ltc_edge_integral(vec3 a, vec3 b)
{
    float cosine = clamp(dot(a, b), -0.9999, 0.9999);
    float theta = acos(cosine);
    float sine = max(sin(theta), 1.0e-5);
    return cross(a, b) * (theta / sine);
}

float arc_ltc_polygon_integral(
    vec3 vertices[8],
    int vertex_count,
    vec3 basis_x,
    vec3 basis_y,
    vec3 basis_z,
    mat3 inverse_transform)
{
    vec3 transformed[8];
    for (int index = 0; index < vertex_count; ++index)
    {
        vec3 local = vec3(
            dot(vertices[index], basis_x),
            dot(vertices[index], basis_y),
            dot(vertices[index], basis_z));
        transformed[index] = normalize(inverse_transform * local);
    }

    vec3 sum = vec3(0.0);
    for (int index = 0; index < vertex_count; ++index)
        sum += arc_ltc_edge_integral(transformed[index], transformed[(index + 1) % vertex_count]);

    // Emitter-facing culling is handled separately; absolute value keeps rectangle/disk
    // winding conventions from changing the finite-emitter energy.
    return clamp(abs(sum.z), 0.0, 2.0 * ARC_PI);
}

vec3 arc_evaluate_area_ltc(
    area_light_data light,
    arc_surface_data surface,
    vec3 view_direction,
    vec3 world_position)
{
    vec3 emitter_normal = normalize(light.direction_two_sided.xyz);
    vec3 center_vector = light.position_shape.xyz - world_position;
    float center_distance = length(center_vector);
    if (center_distance <= 1.0e-5)
        return vec3(0.0);

    vec3 center_direction = center_vector / center_distance;
    float emitter_facing = dot(emitter_normal, -center_direction);
    if (light.direction_two_sided.w <= 0.5 && emitter_facing <= 0.0)
        return vec3(0.0);

    vec3 tangent = light.tangent_width.xyz;
    tangent -= emitter_normal * dot(tangent, emitter_normal);
    if (dot(tangent, tangent) < 1.0e-6)
        tangent = abs(emitter_normal.y) < 0.99
            ? normalize(cross(vec3(0.0, 1.0, 0.0), emitter_normal))
            : normalize(cross(vec3(1.0, 0.0, 0.0), emitter_normal));
    else
        tangent = normalize(tangent);
    vec3 bitangent = normalize(cross(emitter_normal, tangent));

    float width = max(light.tangent_width.w, 1.0e-4);
    float height = max(light.dimensions_shadow.y, 1.0e-4);
    vec3 vertices[8];
    int vertex_count = 4;

    if (light.position_shape.w > 0.5)
    {
        vertex_count = 8;
        float radius = 0.5 * width;
        for (int index = 0; index < 8; ++index)
        {
            float angle = (2.0 * ARC_PI * float(index)) / 8.0;
            vertices[index] = center_vector +
                tangent * (cos(angle) * radius) +
                bitangent * (sin(angle) * radius);
        }
    }
    else
    {
        vec3 half_tangent = tangent * (0.5 * width);
        vec3 half_bitangent = bitangent * (0.5 * height);
        vertices[0] = center_vector - half_tangent - half_bitangent;
        vertices[1] = center_vector + half_tangent - half_bitangent;
        vertices[2] = center_vector + half_tangent + half_bitangent;
        vertices[3] = center_vector - half_tangent + half_bitangent;
    }

    vec3 normal = normalize(surface.normal);
    vec3 view = normalize(view_direction);
    float n_dot_v = clamp(dot(normal, view), 0.0, 1.0);
    vec3 basis_x = view - normal * n_dot_v;
    if (dot(basis_x, basis_x) < 1.0e-6)
        basis_x = abs(normal.y) < 0.99
            ? normalize(cross(vec3(0.0, 1.0, 0.0), normal))
            : normalize(cross(vec3(1.0, 0.0, 0.0), normal));
    else
        basis_x = normalize(basis_x);
    vec3 basis_y = normalize(cross(normal, basis_x));

    arc_ltc_sample lookup = arc_sample_ltc(surface.perceptual_roughness, n_dot_v);
    float diffuse_integral = arc_ltc_polygon_integral(
        vertices, vertex_count, basis_x, basis_y, normal, mat3(1.0));
    float specular_integral = arc_ltc_polygon_integral(
        vertices, vertex_count, basis_x, basis_y, normal, arc_ltc_inverse_matrix(lookup.inverse_transform));

    vec3 diffuse_color = surface.base_color * (1.0 - surface.metallic) / ARC_PI;
    vec3 f0 = mix(vec3(0.04), surface.base_color, surface.metallic);
    vec3 specular_color = lookup.amplitude_fresnel.x *
        mix(f0, vec3(1.0), clamp(lookup.amplitude_fresnel.y, 0.0, 1.0));

    vec3 emitter_radiance = light.color_intensity.rgb * light.color_intensity.w;
    return emitter_radiance *
        (diffuse_color * diffuse_integral + specular_color * specular_integral);
}

vec3 arc_evaluate_scene_lights(
    arc_surface_data surface,
    vec3 view_direction,
    vec3 world_position,
    vec4 directional_visibility)
{
    vec3 direct = vec3(0.0);
    for (uint index = 0u; index < min(lights.directional_count, 4u); ++index)
    {
        vec3 direction_to_light = normalize(-lights.directional_lights[index].direction_intensity.xyz);
        vec3 radiance = lights.directional_lights[index].color_flags.rgb *
            lights.directional_lights[index].direction_intensity.w;
        float source_angle = 0.5 * max(lights.directional_lights[index].source_shape.x, 0.0);
        direct += arc_evaluate_surface_light(
            arc_surface_for_extended_source(surface, source_angle),
            view_direction,
            direction_to_light,
            radiance,
            directional_visibility[index]);
    }
    for (uint index = 0u; index < min(lights.point_count, 64u); ++index)
    {
        vec3 to_light = lights.point_lights[index].position_range.xyz - world_position;
        float distance_squared = max(dot(to_light, to_light), 1.0e-4);
        float distance_to_light = sqrt(distance_squared);
        float normalized_range = clamp(
            distance_to_light / max(lights.point_lights[index].position_range.w, 1.0e-4), 0.0, 1.0);
        float cutoff = 1.0 - pow(normalized_range, 4.0);
        vec3 radiance = lights.point_lights[index].color_intensity.rgb *
            lights.point_lights[index].color_intensity.w * cutoff * cutoff / distance_squared;
        float source_radius = max(
            max(lights.point_lights[index].source_shape.x, 0.0),
            0.5 * max(lights.point_lights[index].source_shape.y, 0.0));
        float source_angle = arc_source_angular_radius(source_radius, distance_to_light);
        direct += arc_evaluate_surface_light(
            arc_surface_for_extended_source(surface, source_angle),
            view_direction,
            to_light / distance_to_light,
            radiance,
            arc_point_shadow_visibility(lights.point_lights[index], world_position));
    }
    for (uint index = 0u; index < min(lights.spot_count, 64u); ++index)
    {
        vec3 to_light = lights.spot_lights[index].position_range.xyz - world_position;
        float distance_squared = max(dot(to_light, to_light), 1.0e-4);
        float distance_to_light = sqrt(distance_squared);
        vec3 direction_to_light = to_light / distance_to_light;
        float normalized_range = clamp(
            distance_to_light / max(lights.spot_lights[index].position_range.w, 1.0e-4), 0.0, 1.0);
        float cutoff = 1.0 - pow(normalized_range, 4.0);
        float cone = smoothstep(
            cos(lights.spot_lights[index].params.x),
            cos(lights.spot_lights[index].direction_inner_angle.w),
            dot(-direction_to_light, normalize(lights.spot_lights[index].direction_inner_angle.xyz)));
        vec3 radiance = lights.spot_lights[index].color_intensity.rgb *
            lights.spot_lights[index].color_intensity.w * cutoff * cutoff * cone / distance_squared;
        float source_radius = max(
            max(lights.spot_lights[index].source_shape.x, 0.0),
            0.5 * max(lights.spot_lights[index].source_shape.y, 0.0));
        float source_angle = arc_source_angular_radius(source_radius, distance_to_light);
        direct += arc_evaluate_surface_light(
            arc_surface_for_extended_source(surface, source_angle),
            view_direction,
            direction_to_light,
            radiance,
            arc_spot_shadow_visibility(lights.spot_lights[index], world_position));
    }
    for (uint index = 0u; index < min(lights.area_count, 32u); ++index)
        direct += arc_evaluate_area_ltc(lights.area_lights[index], surface, view_direction, world_position);
    return direct;
}

#endif
