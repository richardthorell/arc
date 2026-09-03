#include <arc/editor/model_preview.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <numbers>
#include <vector>

namespace arc::editor
{
namespace
{

struct vec3
{
    float x{};
    float y{};
    float z{};
};

vec3 add(vec3 a, vec3 b) noexcept
{
    return {a.x + b.x, a.y + b.y, a.z + b.z};
}

vec3 sub(vec3 a, vec3 b) noexcept
{
    return {a.x - b.x, a.y - b.y, a.z - b.z};
}

vec3 mul(vec3 value, float scale) noexcept
{
    return {value.x * scale, value.y * scale, value.z * scale};
}

float dot(vec3 a, vec3 b) noexcept
{
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

vec3 cross(vec3 a, vec3 b) noexcept
{
    return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x};
}

float length(vec3 value) noexcept
{
    return std::sqrt(std::max(dot(value, value), 0.0f));
}

vec3 normalize(vec3 value) noexcept
{
    const float magnitude = length(value);
    return magnitude > 1.0e-6f ? mul(value, 1.0f / magnitude) : vec3{};
}

vec3 rotate(const math::quatf& rotation, vec3 value) noexcept
{
    const vec3 q{rotation[0], rotation[1], rotation[2]};
    const vec3 t = mul(cross(q, value), 2.0f);
    return add(add(value, mul(t, rotation[3])), cross(q, t));
}

vec3 transform_position(const render::scene_import_node& node, const render::mesh_vertex& vertex) noexcept
{
    const vec3 scaled{vertex.position[0] * node.scale[0], vertex.position[1] * node.scale[1],
                      vertex.position[2] * node.scale[2]};
    return add(rotate(node.rotation, scaled), {node.position[0], node.position[1], node.position[2]});
}

vec3 transform_normal(const render::scene_import_node& node, const render::mesh_vertex& vertex) noexcept
{
    return normalize(rotate(node.rotation, {vertex.normal[0], vertex.normal[1], vertex.normal[2]}));
}

struct bounds3
{
    vec3 minimum{std::numeric_limits<float>::max(), std::numeric_limits<float>::max(),
                 std::numeric_limits<float>::max()};
    vec3 maximum{std::numeric_limits<float>::lowest(), std::numeric_limits<float>::lowest(),
                 std::numeric_limits<float>::lowest()};
    bool valid{};

    void include(vec3 value) noexcept
    {
        minimum.x = std::min(minimum.x, value.x);
        minimum.y = std::min(minimum.y, value.y);
        minimum.z = std::min(minimum.z, value.z);
        maximum.x = std::max(maximum.x, value.x);
        maximum.y = std::max(maximum.y, value.y);
        maximum.z = std::max(maximum.z, value.z);
        valid = true;
    }
};

struct projected_vertex
{
    float x{};
    float y{};
    float depth{};
    vec3 world{};
    vec3 normal{};
    bool visible{};
};

struct projected_bounds
{
    float minimum_x{std::numeric_limits<float>::max()};
    float minimum_y{std::numeric_limits<float>::max()};
    float maximum_x{std::numeric_limits<float>::lowest()};
    float maximum_y{std::numeric_limits<float>::lowest()};
    bool valid{};

    void include(float x, float y) noexcept
    {
        minimum_x = std::min(minimum_x, x);
        minimum_y = std::min(minimum_y, y);
        maximum_x = std::max(maximum_x, x);
        maximum_y = std::max(maximum_y, y);
        valid = true;
    }
};

float edge(float ax, float ay, float bx, float by, float px, float py) noexcept
{
    return (px - ax) * (by - ay) - (py - ay) * (bx - ax);
}

std::uint8_t to_byte(float value) noexcept
{
    value = std::clamp(value, 0.0f, 1.0f);
    const float srgb = value <= 0.0031308f ? value * 12.92f : 1.055f * std::pow(value, 1.0f / 2.4f) - 0.055f;
    return static_cast<std::uint8_t>(std::lround(std::clamp(srgb, 0.0f, 1.0f) * 255.0f));
}

render::material_descriptor default_preview_material()
{
    render::material_descriptor material;
    material.name = "default_phong_preview";
    material.base_color = math::vector4f{0.64f, 0.68f, 0.74f, 1.0f};
    material.metallic = 0.0f;
    material.roughness = 0.48f;
    return material;
}

vec3 shade(const render::material_descriptor& material, vec3 position, vec3 normal, vec3 camera_position) noexcept
{
    const vec3 light_direction = normalize(vec3{0.45f, 0.82f, 0.36f});
    const vec3 view_direction = normalize(sub(camera_position, position));
    const vec3 half_vector = normalize(add(light_direction, view_direction));
    const float diffuse = std::max(dot(normal, light_direction), 0.0f);
    const float spec_power = 8.0f + (1.0f - std::clamp(material.roughness, 0.0f, 1.0f)) * 88.0f;
    const float specular = std::pow(std::max(dot(normal, half_vector), 0.0f), spec_power);
    const vec3 base{material.base_color[0], material.base_color[1], material.base_color[2]};
    const float ambient = 0.2f;
    const float key = 0.78f * diffuse;
    const float rim = 0.12f * std::pow(1.0f - std::max(dot(normal, view_direction), 0.0f), 2.0f);
    const float specular_strength = 0.18f + material.metallic * 0.35f;
    return {base.x * (ambient + key + rim) + specular * specular_strength,
            base.y * (ambient + key + rim) + specular * specular_strength,
            base.z * (ambient + key + rim) + specular * specular_strength};
}

} // namespace

model_preview_result render_model_preview(const render::scene_import_result& scene,
                                          const model_preview_options& options)
{
    const std::uint32_t size = std::clamp(options.size, 32u, 256u);
    if (!scene.succeeded()) return {.message = "model preview requires imported mesh nodes"};

    bounds3 bounds;
    for (const auto& node : scene.nodes)
    {
        if (node.mesh_index >= scene.meshes.size()) continue;
        for (const auto& vertex : scene.meshes[node.mesh_index].vertices)
            bounds.include(transform_position(node, vertex));
    }
    if (!bounds.valid) return {.message = "model preview has no renderable vertices"};

    const vec3 center = mul(add(bounds.minimum, bounds.maximum), 0.5f);
    const vec3 extent = sub(bounds.maximum, bounds.minimum);
    const float radius = std::max(length(extent) * 0.5f, 0.001f);
    const float fov = 35.0f * std::numbers::pi_v<float> / 180.0f;
    const float focal = 1.0f / std::tan(fov * 0.5f);
    const float distance = radius * focal * 1.12f;
    const vec3 camera_direction = normalize(vec3{1.0f, 0.72f, 1.0f});
    vec3 camera_position = add(center, mul(camera_direction, distance));
    const vec3 forward = normalize(sub(center, camera_position));
    vec3 right = normalize(cross(forward, {0.0f, 1.0f, 0.0f}));
    if (length(right) < 1.0e-4f) right = {1.0f, 0.0f, 0.0f};
    const vec3 up = normalize(cross(right, forward));

    // Perspective can shift asymmetric geometry away from the apparent center even when the
    // camera looks at the world-space AABB center. Measure the projected bounds and translate
    // the camera parallel to its image plane so the visible model is centered in the thumbnail.
    projected_bounds screen_bounds;
    for (const auto& node : scene.nodes)
    {
        if (node.mesh_index >= scene.meshes.size()) continue;
        for (const auto& vertex : scene.meshes[node.mesh_index].vertices)
        {
            const vec3 relative = sub(transform_position(node, vertex), camera_position);
            const float depth = dot(relative, forward);
            if (depth <= 1.0e-4f) continue;
            screen_bounds.include(dot(relative, right) * focal / depth, dot(relative, up) * focal / depth);
        }
    }
    if (screen_bounds.valid)
    {
        const float center_x = (screen_bounds.minimum_x + screen_bounds.maximum_x) * 0.5f;
        const float center_y = (screen_bounds.minimum_y + screen_bounds.maximum_y) * 0.5f;
        camera_position =
            add(camera_position, add(mul(right, center_x * distance / focal), mul(up, center_y * distance / focal)));
    }

    render::texture_data texture;
    texture.name = "model-thumbnail";
    texture.width = size;
    texture.height = size;
    texture.format = render::texture_format::rgba8_srgb;
    texture.color_space = render::texture_color_space::srgb;
    texture.semantic = render::texture_semantic::generic_color;
    texture.mip_levels = 1;
    // Value-initialized RGBA pixels stay fully transparent until geometry writes them.
    texture.pixels.resize(static_cast<std::size_t>(size) * size * 4u);

    std::vector<float> depth_buffer(static_cast<std::size_t>(size) * size, std::numeric_limits<float>::max());
    const auto material = options.material_override.value_or(default_preview_material());

    const auto project = [&](vec3 world, vec3 normal)
    {
        const vec3 relative = sub(world, camera_position);
        const float depth = dot(relative, forward);
        if (depth <= 1.0e-4f) return projected_vertex{};
        const float ndc_x = dot(relative, right) * focal / depth;
        const float ndc_y = dot(relative, up) * focal / depth;
        return projected_vertex{.x = (ndc_x * 0.5f + 0.5f) * static_cast<float>(size - 1u),
                                .y = (0.5f - ndc_y * 0.5f) * static_cast<float>(size - 1u),
                                .depth = depth,
                                .world = world,
                                .normal = normal,
                                .visible = true};
    };

    for (const auto& node : scene.nodes)
    {
        if (node.mesh_index >= scene.meshes.size()) continue;
        const auto& mesh = scene.meshes[node.mesh_index];
        for (std::size_t index = 0; index + 2u < mesh.indices.size(); index += 3u)
        {
            const auto i0 = mesh.indices[index];
            const auto i1 = mesh.indices[index + 1u];
            const auto i2 = mesh.indices[index + 2u];
            if (i0 >= mesh.vertices.size() || i1 >= mesh.vertices.size() || i2 >= mesh.vertices.size()) continue;

            const auto world0 = transform_position(node, mesh.vertices[i0]);
            const auto world1 = transform_position(node, mesh.vertices[i1]);
            const auto world2 = transform_position(node, mesh.vertices[i2]);
            auto normal0 = transform_normal(node, mesh.vertices[i0]);
            auto normal1 = transform_normal(node, mesh.vertices[i1]);
            auto normal2 = transform_normal(node, mesh.vertices[i2]);
            const vec3 face_normal = normalize(cross(sub(world1, world0), sub(world2, world0)));
            if (length(normal0) < 1.0e-4f) normal0 = face_normal;
            if (length(normal1) < 1.0e-4f) normal1 = face_normal;
            if (length(normal2) < 1.0e-4f) normal2 = face_normal;

            const auto a = project(world0, normal0);
            const auto b = project(world1, normal1);
            const auto c = project(world2, normal2);
            if (!a.visible || !b.visible || !c.visible) continue;

            const float area = edge(a.x, a.y, b.x, b.y, c.x, c.y);
            if (std::abs(area) < 1.0e-6f) continue;
            const int min_x = std::max(0, static_cast<int>(std::floor(std::min({a.x, b.x, c.x}))));
            const int max_x =
                std::min(static_cast<int>(size) - 1, static_cast<int>(std::ceil(std::max({a.x, b.x, c.x}))));
            const int min_y = std::max(0, static_cast<int>(std::floor(std::min({a.y, b.y, c.y}))));
            const int max_y =
                std::min(static_cast<int>(size) - 1, static_cast<int>(std::ceil(std::max({a.y, b.y, c.y}))));

            for (int y = min_y; y <= max_y; ++y)
            {
                for (int x = min_x; x <= max_x; ++x)
                {
                    const float px = static_cast<float>(x) + 0.5f;
                    const float py = static_cast<float>(y) + 0.5f;
                    const float w0 = edge(b.x, b.y, c.x, c.y, px, py) / area;
                    const float w1 = edge(c.x, c.y, a.x, a.y, px, py) / area;
                    const float w2 = 1.0f - w0 - w1;
                    if (w0 < 0.0f || w1 < 0.0f || w2 < 0.0f) continue;
                    const float depth = a.depth * w0 + b.depth * w1 + c.depth * w2;
                    const auto pixel = static_cast<std::size_t>(y) * size + static_cast<std::size_t>(x);
                    if (depth >= depth_buffer[pixel]) continue;
                    depth_buffer[pixel] = depth;

                    const vec3 world = add(add(mul(a.world, w0), mul(b.world, w1)), mul(c.world, w2));
                    const vec3 normal = normalize(add(add(mul(a.normal, w0), mul(b.normal, w1)), mul(c.normal, w2)));
                    const vec3 color = shade(material, world, normal, camera_position);
                    const auto offset = pixel * 4u;
                    texture.pixels[offset] = static_cast<std::byte>(to_byte(color.x));
                    texture.pixels[offset + 1u] = static_cast<std::byte>(to_byte(color.y));
                    texture.pixels[offset + 2u] = static_cast<std::byte>(to_byte(color.z));
                    texture.pixels[offset + 3u] = static_cast<std::byte>(255u);
                }
            }
        }
    }

    return {.texture = std::move(texture), .message = "rendered model thumbnail"};
}

} // namespace arc::editor
