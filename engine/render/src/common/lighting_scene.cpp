#include <arc/render/lighting_scene.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <limits>
#include <queue>
#include <unordered_map>
#include <unordered_set>

namespace arc::render
{
namespace
{

constexpr float distance_epsilon = 1.0e-5f;
constexpr std::uint32_t invalid_grid_distance = std::numeric_limits<std::uint32_t>::max();

math::vector3f position_of(const mesh_vertex& vertex) noexcept
{
    return {vertex.position[0], vertex.position[1], vertex.position[2]};
}

geometric::box3f mesh_bounds(const mesh_data& mesh)
{
    if (mesh.vertices.empty()) return {};
    auto minimum = position_of(mesh.vertices.front());
    auto maximum = minimum;
    for (const auto& vertex : mesh.vertices)
    {
        const auto position = position_of(vertex);
        for (std::size_t axis = 0; axis < 3; ++axis)
        {
            minimum[axis] = std::min(minimum[axis], position[axis]);
            maximum[axis] = std::max(maximum[axis], position[axis]);
        }
    }
    return {geometric::point3f{minimum}, geometric::point3f{maximum}};
}

float point_triangle_distance_squared(const math::vector3f& point, const math::vector3f& a, const math::vector3f& b,
                                      const math::vector3f& c) noexcept
{
    const math::vector3f ab = b - a;
    const math::vector3f ac = c - a;
    const math::vector3f ap = point - a;
    const float d1 = math::dot(ab, ap);
    const float d2 = math::dot(ac, ap);
    if (d1 <= 0.0f && d2 <= 0.0f) return math::dot(ap, ap);

    const math::vector3f bp = point - b;
    const float d3 = math::dot(ab, bp);
    const float d4 = math::dot(ac, bp);
    if (d3 >= 0.0f && d4 <= d3) return math::dot(bp, bp);

    const float vc = d1 * d4 - d3 * d2;
    if (vc <= 0.0f && d1 >= 0.0f && d3 <= 0.0f)
    {
        const float value = d1 / (d1 - d3);
        const auto delta = point - (a + ab * value);
        return math::dot(delta, delta);
    }

    const math::vector3f cp = point - c;
    const float d5 = math::dot(ab, cp);
    const float d6 = math::dot(ac, cp);
    if (d6 >= 0.0f && d5 <= d6) return math::dot(cp, cp);

    const float vb = d5 * d2 - d1 * d6;
    if (vb <= 0.0f && d2 >= 0.0f && d6 <= 0.0f)
    {
        const float value = d2 / (d2 - d6);
        const auto delta = point - (a + ac * value);
        return math::dot(delta, delta);
    }

    const float va = d3 * d6 - d5 * d4;
    if (va <= 0.0f && (d4 - d3) >= 0.0f && (d5 - d6) >= 0.0f)
    {
        const float value = (d4 - d3) / ((d4 - d3) + (d5 - d6));
        const auto delta = point - (b + (c - b) * value);
        return math::dot(delta, delta);
    }

    const auto normal = math::normalize(math::cross(ab, ac));
    const float distance = math::dot(point - a, normal);
    return distance * distance;
}

std::uint64_t edge_key(std::uint32_t first, std::uint32_t second) noexcept
{
    if (first > second) std::swap(first, second);
    return (static_cast<std::uint64_t>(first) << 32u) | second;
}

std::uint64_t hash_bytes(std::span<const std::byte> bytes) noexcept
{
    std::uint64_t result = 1469598103934665603ull;
    for (const auto value : bytes)
    {
        result ^= std::to_integer<std::uint8_t>(value);
        result *= 1099511628211ull;
    }
    return result;
}

std::size_t flatten(std::uint32_t x, std::uint32_t y, std::uint32_t z,
                    const std::array<std::uint32_t, 3>& dimensions) noexcept
{
    return (static_cast<std::size_t>(z) * dimensions[1] + y) * dimensions[0] + x;
}

math::vector3f voxel_center(const geometric::box3f& bounds, const math::vector3f& voxel_size, std::uint32_t x,
                            std::uint32_t y, std::uint32_t z) noexcept
{
    return {bounds.min[0] + (static_cast<float>(x) + 0.5f) * voxel_size[0],
            bounds.min[1] + (static_cast<float>(y) + 0.5f) * voxel_size[1],
            bounds.min[2] + (static_cast<float>(z) + 0.5f) * voxel_size[2]};
}

std::vector<surface_card_descriptor> build_cards(const geometric::box3f& bounds, float density)
{
    const auto center = geometric::center(bounds);
    const auto size = geometric::size(bounds);
    const std::array<math::vector3f, 6> normals = {math::vector3f{1.0f, 0.0f, 0.0f}, math::vector3f{-1.0f, 0.0f, 0.0f},
                                                   math::vector3f{0.0f, 1.0f, 0.0f}, math::vector3f{0.0f, -1.0f, 0.0f},
                                                   math::vector3f{0.0f, 0.0f, 1.0f}, math::vector3f{0.0f, 0.0f, -1.0f}};
    std::vector<surface_card_descriptor> cards;
    cards.reserve(normals.size());
    for (std::uint32_t index = 0; index < normals.size(); ++index)
    {
        const std::uint32_t axis = index / 2u;
        const std::uint32_t tangent_axis = axis == 0 ? 2u : 0u;
        const std::uint32_t bitangent_axis = axis == 1 ? 2u : 1u;
        math::vector3f tangent{};
        tangent[tangent_axis] = 1.0f;
        auto card_center = math::vector3f{center[0], center[1], center[2]};
        card_center[axis] += normals[index][axis] * size[axis] * 0.5f;
        cards.push_back({.center = card_center,
                         .normal = normals[index],
                         .tangent = tangent,
                         .extent = {std::max(size[tangent_axis] * 0.5f, distance_epsilon),
                                    std::max(size[bitangent_axis] * 0.5f, distance_epsilon)},
                         .depth_extent = std::max(size[axis], distance_epsilon),
                         .texel_density = density,
                         .material_section = 0,
                         .fallback_card = index});
    }
    return cards;
}

float sample_field(const mesh_distance_field_descriptor& field, const math::vector3f& position)
{
    if (field.bricks.empty() || field.pages.empty()) return std::numeric_limits<float>::infinity();
    const auto size = geometric::size(field.bounds);
    std::array<std::uint32_t, 3> coordinate{};
    for (std::size_t axis = 0; axis < 3; ++axis)
    {
        if (position[axis] < field.bounds.min[axis] || position[axis] > field.bounds.max[axis])
        {
            const float delta = position[axis] < field.bounds.min[axis] ? field.bounds.min[axis] - position[axis]
                                                                        : position[axis] - field.bounds.max[axis];
            return std::max(delta, std::min({field.voxel_size[0], field.voxel_size[1], field.voxel_size[2]}));
        }
        const float normalized =
            size[axis] > distance_epsilon ? (position[axis] - field.bounds.min[axis]) / size[axis] : 0.0f;
        coordinate[axis] =
            std::min(field.dimensions[axis] - 1u, static_cast<std::uint32_t>(normalized * field.dimensions[axis]));
    }

    const std::array<std::uint16_t, 3> brick_coordinate = {
        static_cast<std::uint16_t>(coordinate[0] / mesh_distance_field_descriptor::brick_dimension),
        static_cast<std::uint16_t>(coordinate[1] / mesh_distance_field_descriptor::brick_dimension),
        static_cast<std::uint16_t>(coordinate[2] / mesh_distance_field_descriptor::brick_dimension)};
    const auto found = std::find_if(field.bricks.begin(), field.bricks.end(),
                                    [&](const auto& brick) { return brick.coordinate == brick_coordinate; });
    if (found == field.bricks.end()) return field.distance_scale;

    const std::uint32_t local_x = coordinate[0] % mesh_distance_field_descriptor::brick_dimension;
    const std::uint32_t local_y = coordinate[1] % mesh_distance_field_descriptor::brick_dimension;
    const std::uint32_t local_z = coordinate[2] % mesh_distance_field_descriptor::brick_dimension;
    const std::size_t sample_index =
        (static_cast<std::size_t>(local_z) * mesh_distance_field_descriptor::brick_dimension + local_y) *
            mesh_distance_field_descriptor::brick_dimension +
        local_x;
    std::int16_t encoded{};
    const std::size_t byte_offset = found->page_offset + sample_index * sizeof(encoded);
    const std::size_t absolute_offset = field.page_offsets[found->page_index] + byte_offset;
    if (absolute_offset + sizeof(encoded) > field.pages.size()) return field.distance_scale;
    std::memcpy(&encoded, field.pages.data() + absolute_offset, sizeof(encoded));
    return static_cast<float>(encoded) / 32767.0f * field.distance_scale;
}

} // namespace

lighting_geometry_build_result build_lighting_geometry(const mesh_data& mesh,
                                                       const lighting_geometry_build_options& options)
{
    lighting_geometry_build_result result;
    result.geometry.name = mesh.name;
    result.geometry.bounds = mesh_bounds(mesh);
    result.statistics.source_triangles = static_cast<std::uint32_t>(mesh.indices.size() / 3u);
    if (mesh.vertices.empty() || mesh.indices.size() < 3)
    {
        result.diagnostics.emplace_back("lighting geometry requires indexed triangle data");
        return result;
    }

    result.geometry.cards = build_cards(result.geometry.bounds, std::max(options.card_texel_density, 1.0f));
    result.statistics.card_count = static_cast<std::uint32_t>(result.geometry.cards.size());

    std::unordered_map<std::uint64_t, std::uint32_t> edge_counts;
    std::vector<std::array<std::uint32_t, 3>> valid_triangles;
    valid_triangles.reserve(result.statistics.source_triangles);
    for (std::size_t index = 0; index + 2 < mesh.indices.size(); index += 3)
    {
        const std::array triangle = {mesh.indices[index], mesh.indices[index + 1], mesh.indices[index + 2]};
        if (triangle[0] >= mesh.vertices.size() || triangle[1] >= mesh.vertices.size() ||
            triangle[2] >= mesh.vertices.size() || triangle[0] == triangle[1] || triangle[1] == triangle[2] ||
            triangle[0] == triangle[2])
        {
            ++result.statistics.rejected_triangles;
            continue;
        }
        const auto a = position_of(mesh.vertices[triangle[0]]);
        const auto b = position_of(mesh.vertices[triangle[1]]);
        const auto c = position_of(mesh.vertices[triangle[2]]);
        if (math::length(math::cross(b - a, c - a)) <= distance_epsilon)
        {
            ++result.statistics.rejected_triangles;
            continue;
        }
        valid_triangles.push_back(triangle);
        ++edge_counts[edge_key(triangle[0], triangle[1])];
        ++edge_counts[edge_key(triangle[1], triangle[2])];
        ++edge_counts[edge_key(triangle[2], triangle[0])];
    }
    result.statistics.watertight =
        !valid_triangles.empty() &&
        std::all_of(edge_counts.begin(), edge_counts.end(), [](const auto& edge) { return edge.second == 2u; });

    auto& field = result.geometry.distance_field;
    field.bounds = geometric::expand(result.geometry.bounds, 0.01f);
    field.mode = result.statistics.watertight ? distance_field_mode::signed_distance
                                              : distance_field_mode::two_sided_unsigned_distance;
    const auto bounds_size = geometric::size(field.bounds);
    const float longest = std::max({bounds_size[0], bounds_size[1], bounds_size[2], distance_epsilon});
    const std::uint32_t minimum_resolution = std::clamp(options.minimum_distance_field_resolution, 8u, 128u);
    const std::uint32_t maximum_resolution =
        std::max(minimum_resolution, std::clamp(options.maximum_distance_field_resolution, 8u, 128u));
    const std::uint32_t longest_resolution = std::clamp(
        static_cast<std::uint32_t>(std::round(32.0f * std::max(options.distance_field_resolution_scale, 0.25f))),
        minimum_resolution, maximum_resolution);
    for (std::size_t axis = 0; axis < 3; ++axis)
    {
        field.dimensions[axis] =
            std::max(8u, static_cast<std::uint32_t>(std::ceil(longest_resolution * bounds_size[axis] / longest)));
        field.dimensions[axis] = (field.dimensions[axis] + 7u) & ~7u;
        field.voxel_size[axis] = bounds_size[axis] / static_cast<float>(field.dimensions[axis]);
    }
    const float voxel_length = std::max({field.voxel_size[0], field.voxel_size[1], field.voxel_size[2]});
    const float surface_threshold = voxel_length * 0.8f;
    const std::size_t voxel_count =
        static_cast<std::size_t>(field.dimensions[0]) * field.dimensions[1] * field.dimensions[2];
    std::vector<std::uint8_t> surface(voxel_count);

    for (const auto& triangle : valid_triangles)
    {
        const auto a = position_of(mesh.vertices[triangle[0]]);
        const auto b = position_of(mesh.vertices[triangle[1]]);
        const auto c = position_of(mesh.vertices[triangle[2]]);
        std::array<std::uint32_t, 3> lower{};
        std::array<std::uint32_t, 3> upper{};
        for (std::size_t axis = 0; axis < 3; ++axis)
        {
            const float minimum = std::min({a[axis], b[axis], c[axis]}) - surface_threshold;
            const float maximum = std::max({a[axis], b[axis], c[axis]}) + surface_threshold;
            lower[axis] = static_cast<std::uint32_t>(
                std::clamp<float>(std::floor((minimum - field.bounds.min[axis]) / field.voxel_size[axis]), 0.0,
                                  static_cast<float>(field.dimensions[axis] - 1u)));
            upper[axis] = static_cast<std::uint32_t>(
                std::clamp<float>(std::ceil((maximum - field.bounds.min[axis]) / field.voxel_size[axis]), 0.0,
                                  static_cast<float>(field.dimensions[axis] - 1u)));
        }
        for (std::uint32_t z = lower[2]; z <= upper[2]; ++z)
            for (std::uint32_t y = lower[1]; y <= upper[1]; ++y)
                for (std::uint32_t x = lower[0]; x <= upper[0]; ++x)
                {
                    const auto point = voxel_center(field.bounds, field.voxel_size, x, y, z);
                    if (point_triangle_distance_squared(point, a, b, c) <= surface_threshold * surface_threshold)
                        surface[flatten(x, y, z, field.dimensions)] = 1;
                }
    }

    std::vector<std::uint32_t> grid_distance(voxel_count, invalid_grid_distance);
    std::queue<std::array<std::uint32_t, 3>> distance_queue;
    for (std::uint32_t z = 0; z < field.dimensions[2]; ++z)
        for (std::uint32_t y = 0; y < field.dimensions[1]; ++y)
            for (std::uint32_t x = 0; x < field.dimensions[0]; ++x)
                if (surface[flatten(x, y, z, field.dimensions)] != 0)
                {
                    grid_distance[flatten(x, y, z, field.dimensions)] = 0;
                    distance_queue.push({x, y, z});
                }

    constexpr std::array<std::array<int, 3>, 6> neighbors = {
        std::array<int, 3>{1, 0, 0}, {-1, 0, 0}, {0, 1, 0}, {0, -1, 0}, {0, 0, 1}, {0, 0, -1}};
    while (!distance_queue.empty())
    {
        const auto current = distance_queue.front();
        distance_queue.pop();
        const auto current_distance = grid_distance[flatten(current[0], current[1], current[2], field.dimensions)];
        for (const auto& delta : neighbors)
        {
            const int x = static_cast<int>(current[0]) + delta[0];
            const int y = static_cast<int>(current[1]) + delta[1];
            const int z = static_cast<int>(current[2]) + delta[2];
            if (x < 0 || y < 0 || z < 0 || x >= static_cast<int>(field.dimensions[0]) ||
                y >= static_cast<int>(field.dimensions[1]) || z >= static_cast<int>(field.dimensions[2]))
                continue;
            auto& candidate = grid_distance[flatten(static_cast<std::uint32_t>(x), static_cast<std::uint32_t>(y),
                                                    static_cast<std::uint32_t>(z), field.dimensions)];
            if (candidate <= current_distance + 1u) continue;
            candidate = current_distance + 1u;
            distance_queue.push(
                {static_cast<std::uint32_t>(x), static_cast<std::uint32_t>(y), static_cast<std::uint32_t>(z)});
        }
    }

    std::vector<std::uint8_t> outside(voxel_count);
    if (result.statistics.watertight)
    {
        std::queue<std::array<std::uint32_t, 3>> outside_queue;
        const auto seed = [&](std::uint32_t x, std::uint32_t y, std::uint32_t z)
        {
            const auto index = flatten(x, y, z, field.dimensions);
            if (surface[index] == 0 && outside[index] == 0)
            {
                outside[index] = 1;
                outside_queue.push({x, y, z});
            }
        };
        for (std::uint32_t z = 0; z < field.dimensions[2]; ++z)
            for (std::uint32_t y = 0; y < field.dimensions[1]; ++y)
            {
                seed(0, y, z);
                seed(field.dimensions[0] - 1u, y, z);
            }
        for (std::uint32_t z = 0; z < field.dimensions[2]; ++z)
            for (std::uint32_t x = 0; x < field.dimensions[0]; ++x)
            {
                seed(x, 0, z);
                seed(x, field.dimensions[1] - 1u, z);
            }
        for (std::uint32_t y = 0; y < field.dimensions[1]; ++y)
            for (std::uint32_t x = 0; x < field.dimensions[0]; ++x)
            {
                seed(x, y, 0);
                seed(x, y, field.dimensions[2] - 1u);
            }
        while (!outside_queue.empty())
        {
            const auto current = outside_queue.front();
            outside_queue.pop();
            for (const auto& delta : neighbors)
            {
                const int x = static_cast<int>(current[0]) + delta[0];
                const int y = static_cast<int>(current[1]) + delta[1];
                const int z = static_cast<int>(current[2]) + delta[2];
                if (x < 0 || y < 0 || z < 0 || x >= static_cast<int>(field.dimensions[0]) ||
                    y >= static_cast<int>(field.dimensions[1]) || z >= static_cast<int>(field.dimensions[2]))
                    continue;
                const auto index = flatten(static_cast<std::uint32_t>(x), static_cast<std::uint32_t>(y),
                                           static_cast<std::uint32_t>(z), field.dimensions);
                if (surface[index] != 0 || outside[index] != 0) continue;
                outside[index] = 1;
                outside_queue.push(
                    {static_cast<std::uint32_t>(x), static_cast<std::uint32_t>(y), static_cast<std::uint32_t>(z)});
            }
        }
    }

    field.distance_scale = std::max(
        voxel_length,
        voxel_length * static_cast<float>(std::max({field.dimensions[0], field.dimensions[1], field.dimensions[2]})));
    const std::uint32_t bricks_x = field.dimensions[0] / mesh_distance_field_descriptor::brick_dimension;
    const std::uint32_t bricks_y = field.dimensions[1] / mesh_distance_field_descriptor::brick_dimension;
    const std::uint32_t bricks_z = field.dimensions[2] / mesh_distance_field_descriptor::brick_dimension;
    constexpr std::size_t brick_sample_count = mesh_distance_field_descriptor::brick_dimension *
                                               mesh_distance_field_descriptor::brick_dimension *
                                               mesh_distance_field_descriptor::brick_dimension;
    constexpr std::size_t brick_bytes = brick_sample_count * sizeof(std::int16_t);
    std::vector<std::byte> encoded;
    for (std::uint32_t brick_z = 0; brick_z < bricks_z; ++brick_z)
        for (std::uint32_t brick_y = 0; brick_y < bricks_y; ++brick_y)
            for (std::uint32_t brick_x = 0; brick_x < bricks_x; ++brick_x)
            {
                std::array<std::int16_t, brick_sample_count> samples{};
                float minimum = std::numeric_limits<float>::infinity();
                float maximum = -std::numeric_limits<float>::infinity();
                for (std::uint32_t local_z = 0; local_z < mesh_distance_field_descriptor::brick_dimension; ++local_z)
                    for (std::uint32_t local_y = 0; local_y < mesh_distance_field_descriptor::brick_dimension;
                         ++local_y)
                        for (std::uint32_t local_x = 0; local_x < mesh_distance_field_descriptor::brick_dimension;
                             ++local_x)
                        {
                            const std::uint32_t x = brick_x * mesh_distance_field_descriptor::brick_dimension + local_x;
                            const std::uint32_t y = brick_y * mesh_distance_field_descriptor::brick_dimension + local_y;
                            const std::uint32_t z = brick_z * mesh_distance_field_descriptor::brick_dimension + local_z;
                            const auto index = flatten(x, y, z, field.dimensions);
                            float distance = grid_distance[index] == invalid_grid_distance
                                                 ? field.distance_scale
                                                 : static_cast<float>(grid_distance[index]) * voxel_length;
                            if (result.statistics.watertight && outside[index] == 0 && surface[index] == 0)
                                distance = -distance;
                            minimum = std::min(minimum, distance);
                            maximum = std::max(maximum, distance);
                            const auto sample_index =
                                (static_cast<std::size_t>(local_z) * mesh_distance_field_descriptor::brick_dimension +
                                 local_y) *
                                    mesh_distance_field_descriptor::brick_dimension +
                                local_x;
                            samples[sample_index] = static_cast<std::int16_t>(
                                std::round(std::clamp(distance / field.distance_scale, -1.0f, 1.0f) * 32767.0f));
                        }
                const bool near_surface = minimum <= options.narrow_band_voxels * voxel_length || minimum < 0.0f;
                if (!near_surface) continue;
                if (field.page_offsets.empty() || encoded.size() - field.page_offsets.back() + brick_bytes >
                                                      mesh_distance_field_descriptor::page_size)
                    field.page_offsets.push_back(static_cast<std::uint32_t>(encoded.size()));
                const std::uint32_t page_index = static_cast<std::uint32_t>(field.page_offsets.size() - 1u);
                const std::uint32_t page_offset =
                    static_cast<std::uint32_t>(encoded.size() - field.page_offsets[page_index]);
                const auto* bytes = reinterpret_cast<const std::byte*>(samples.data());
                encoded.insert(encoded.end(), bytes, bytes + brick_bytes);
                field.bricks.push_back(
                    {.coordinate = {static_cast<std::uint16_t>(brick_x), static_cast<std::uint16_t>(brick_y),
                                    static_cast<std::uint16_t>(brick_z)},
                     .page_index = page_index,
                     .page_offset = page_offset,
                     .byte_size = static_cast<std::uint32_t>(brick_bytes),
                     .minimum_distance = minimum,
                     .maximum_distance = maximum});
            }
    field.pages = std::move(encoded);
    field.content_hash = hash_bytes(field.pages);
    result.statistics.brick_count = static_cast<std::uint32_t>(field.bricks.size());
    result.statistics.page_count = static_cast<std::uint32_t>(field.page_offsets.size());
    result.statistics.encoded_bytes = field.pages.size();
    if (!result.statistics.watertight)
        result.diagnostics.emplace_back("non-watertight mesh uses conservative two-sided unsigned distances");
    return result;
}

struct lighting_scene::implementation
{
    struct slot
    {
        lighting_scene_instance instance{};
        std::uint64_t last_seen_frame{};
        std::uint32_t generation{1};
        bool alive{};
    };

    lighting_scene_config config{};
    std::vector<slot> slots;
    std::vector<std::uint32_t> free_slots;
    std::unordered_map<std::uint64_t, std::uint32_t> lookup;
    lighting_scene_snapshot snapshot{};
};

lighting_scene::lighting_scene(lighting_scene_config config) : implementation_(std::make_unique<implementation>())
{
    implementation_->config = config;
}

lighting_scene::~lighting_scene() = default;
lighting_scene::lighting_scene(lighting_scene&&) noexcept = default;
lighting_scene& lighting_scene::operator=(lighting_scene&&) noexcept = default;

void lighting_scene::configure(lighting_scene_config config)
{
    implementation_->config = config;
}

lighting_scene_update_batch lighting_scene::synchronize(std::uint64_t world_id, std::uint64_t world_epoch,
                                                        std::uint64_t frame_index,
                                                        std::span<const lighting_scene_instance> instances)
{
    auto& state = *implementation_;
    lighting_scene_update_batch result{.frame_index = frame_index, .world_id = world_id, .world_epoch = world_epoch};
    if (state.snapshot.world_id != 0 &&
        (state.snapshot.world_id != world_id || state.snapshot.world_epoch != world_epoch))
    {
        result.updates.push_back({.kind = lighting_scene_update_kind::reset});
        state.slots.clear();
        state.free_slots.clear();
        state.lookup.clear();
        ++state.snapshot.cache_generation;
    }

    for (const auto& instance : instances)
    {
        if (!instance.affects_indirect_lighting || instance.stable_id == 0 || !instance.geometry.valid()) continue;
        auto found = state.lookup.find(instance.stable_id);
        std::uint32_t slot_index{};
        bool created{};
        if (found == state.lookup.end())
        {
            if (!state.free_slots.empty())
            {
                slot_index = state.free_slots.back();
                state.free_slots.pop_back();
            }
            else
            {
                slot_index = static_cast<std::uint32_t>(state.slots.size());
                state.slots.emplace_back();
            }
            state.lookup.emplace(instance.stable_id, slot_index);
            state.slots[slot_index].alive = true;
            created = true;
        }
        else
        {
            slot_index = found->second;
        }
        auto& slot = state.slots[slot_index];
        const bool transform_dirty = created || slot.instance.transform_revision != instance.transform_revision;
        const bool material_dirty = created || slot.instance.material_revision != instance.material_revision ||
                                    slot.instance.material != instance.material;
        const bool geometry_dirty = created || slot.instance.geometry != instance.geometry ||
                                    slot.instance.geometry_generation != instance.geometry_generation;
        slot.instance = instance;
        slot.last_seen_frame = frame_index;
        if (transform_dirty || material_dirty || geometry_dirty)
        {
            result.updates.push_back({.kind = lighting_scene_update_kind::upsert,
                                      .handle = {slot_index, slot.generation},
                                      .instance = instance,
                                      .transform_dirty = transform_dirty,
                                      .material_dirty = material_dirty,
                                      .geometry_dirty = geometry_dirty});
            result.dirty_world_regions.push_back(instance.world_bounds);
            ++state.snapshot.cache_generation;
        }
    }

    for (std::uint32_t index = 0; index < state.slots.size(); ++index)
    {
        auto& slot = state.slots[index];
        if (!slot.alive || slot.last_seen_frame == frame_index) continue;
        result.updates.push_back({.kind = lighting_scene_update_kind::destroy,
                                  .handle = {index, slot.generation},
                                  .instance = slot.instance,
                                  .transform_dirty = true,
                                  .material_dirty = true,
                                  .geometry_dirty = true});
        result.dirty_world_regions.push_back(slot.instance.world_bounds);
        state.lookup.erase(slot.instance.stable_id);
        slot.alive = false;
        ++slot.generation;
        if (slot.generation == 0) slot.generation = 1;
        state.free_slots.push_back(index);
        ++state.snapshot.cache_generation;
    }

    result.active_instances = static_cast<std::uint32_t>(state.lookup.size());
    state.snapshot.frame_index = frame_index;
    state.snapshot.world_id = world_id;
    state.snapshot.world_epoch = world_epoch;
    state.snapshot.gpu_budget_bytes = state.config.gpu_budget_bytes;
    state.snapshot.active_instances = result.active_instances;
    state.snapshot.dirty_regions = static_cast<std::uint32_t>(result.dirty_world_regions.size());
    return result;
}

void lighting_scene::reset()
{
    const auto config = implementation_->config;
    implementation_ = std::make_unique<implementation>();
    implementation_->config = config;
}

const lighting_scene_instance* lighting_scene::find(lighting_scene_instance_handle handle) const noexcept
{
    if (!handle.valid() || handle.index >= implementation_->slots.size()) return nullptr;
    const auto& slot = implementation_->slots[handle.index];
    return slot.alive && slot.generation == handle.generation ? &slot.instance : nullptr;
}

lighting_scene_snapshot lighting_scene::snapshot() const noexcept
{
    return implementation_->snapshot;
}

void lighting_scene::update_residency_statistics(std::uint32_t surface_cards, std::uint32_t surface_pages,
                                                 std::uint32_t distance_field_pages, std::uint64_t resident_bytes,
                                                 std::uint32_t evictions) noexcept
{
    implementation_->snapshot.surface_cards = surface_cards;
    implementation_->snapshot.resident_surface_pages = surface_pages;
    implementation_->snapshot.resident_distance_field_pages = distance_field_pages;
    implementation_->snapshot.gpu_resident_bytes = resident_bytes;
    implementation_->snapshot.evictions = evictions;
}

lighting_trace_result trace_mesh_distance_field(const mesh_distance_field_descriptor& field,
                                                const lighting_trace_ray& ray, std::uint32_t maximum_steps)
{
    lighting_trace_result result;
    const auto direction = math::normalize(ray.direction);
    if (math::length(direction) <= distance_epsilon || field.bricks.empty()) return result;
    float distance = std::max(ray.minimum_distance, 0.0f);
    const float minimum_step =
        std::max(std::min({field.voxel_size[0], field.voxel_size[1], field.voxel_size[2]}) * 0.5f, 0.001f);
    for (std::uint32_t step = 0; step < maximum_steps && distance <= ray.maximum_distance; ++step)
    {
        const auto position = ray.origin + direction * distance;
        const float sampled = sample_field(field, position);
        result.steps = step + 1u;
        if (std::abs(sampled) <= minimum_step)
        {
            result.hit = true;
            result.distance = distance;
            result.position = position;
            result.source = lighting_trace_source::software_distance_field;
            return result;
        }
        distance += std::max(std::abs(sampled) * 0.8f, minimum_step);
    }
    return result;
}

} // namespace arc::render
