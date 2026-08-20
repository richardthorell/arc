#include <arc/editor/procedural_mesh.h>

#include <arc/editor/editor_state.h>
#include <arc/geometric/box.h>
#include <arc/math/constants.h>
#include <arc/render/primitives.h>

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cmath>
#include <limits>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace arc::editor
{
namespace
{
using json = nlohmann::json;

constexpr std::string_view procedural_component_name = "ProceduralMesh";
constexpr float minimum_dimension = 0.001f;
constexpr float maximum_dimension = 100000.0f;
constexpr std::uint32_t maximum_segments = 512u;

struct point3
{
    float x{};
    float y{};
    float z{};
};

constexpr point3 operator+(point3 lhs, point3 rhs) noexcept
{
    return {lhs.x + rhs.x, lhs.y + rhs.y, lhs.z + rhs.z};
}

constexpr point3 operator-(point3 lhs, point3 rhs) noexcept
{
    return {lhs.x - rhs.x, lhs.y - rhs.y, lhs.z - rhs.z};
}

constexpr point3 operator*(point3 value, float scalar) noexcept
{
    return {value.x * scalar, value.y * scalar, value.z * scalar};
}

point3 lerp(point3 start, point3 end, float amount) noexcept
{
    return start + (end - start) * amount;
}

float safe_dimension(float value) noexcept
{
    return std::clamp(value, minimum_dimension, maximum_dimension);
}

float safe_dimension(double value) noexcept
{
    return std::isfinite(value) ? safe_dimension(static_cast<float>(value)) : minimum_dimension;
}

std::uint32_t safe_segments(std::uint32_t value, std::uint32_t minimum) noexcept
{
    return std::clamp(value, minimum, maximum_segments);
}

std::uint32_t safe_segments(double value, std::uint32_t minimum) noexcept
{
    if (!std::isfinite(value)) return minimum;
    const auto rounded = static_cast<long long>(std::llround(value));
    return static_cast<std::uint32_t>(
        std::clamp<long long>(rounded, static_cast<long long>(minimum), static_cast<long long>(maximum_segments)));
}

render::mesh_vertex make_vertex(point3 position, point3 normal, float u, float v)
{
    return {.position = {position.x, position.y, position.z},
            .normal = {normal.x, normal.y, normal.z},
            .texcoord = {u, v},
            .color = {1.0f, 1.0f, 1.0f, 1.0f}};
}

void append_grid_face(render::mesh_data& mesh, point3 p00, point3 p10, point3 p01, point3 normal,
                      std::uint32_t segments_u, std::uint32_t segments_v)
{
    segments_u = safe_segments(segments_u, 1u);
    segments_v = safe_segments(segments_v, 1u);
    const point3 p11 = p10 + (p01 - p00);
    const auto base = static_cast<std::uint32_t>(mesh.vertices.size());
    const std::uint32_t stride = segments_u + 1u;

    for (std::uint32_t row = 0; row <= segments_v; ++row)
    {
        const float v = static_cast<float>(row) / static_cast<float>(segments_v);
        const point3 left = lerp(p00, p01, v);
        const point3 right = lerp(p10, p11, v);
        for (std::uint32_t column = 0; column <= segments_u; ++column)
        {
            const float u = static_cast<float>(column) / static_cast<float>(segments_u);
            mesh.vertices.push_back(make_vertex(lerp(left, right, u), normal, u, v));
        }
    }

    for (std::uint32_t row = 0; row < segments_v; ++row)
    {
        for (std::uint32_t column = 0; column < segments_u; ++column)
        {
            const std::uint32_t a = base + row * stride + column;
            const std::uint32_t b = a + 1u;
            const std::uint32_t d = a + stride;
            const std::uint32_t c = d + 1u;
            mesh.indices.insert(mesh.indices.end(), {a, b, c, a, c, d});
        }
    }
}

render::mesh_data make_plane(const plane_mesh_parameters& source)
{
    const float width = safe_dimension(source.width);
    const float depth = safe_dimension(source.depth);
    const std::uint32_t segments_x = safe_segments(source.segments_x, 1u);
    const std::uint32_t segments_z = safe_segments(source.segments_z, 1u);
    const float half_width = width * 0.5f;
    const float half_depth = depth * 0.5f;
    const std::uint32_t stride = segments_x + 1u;

    render::mesh_data mesh;
    mesh.name = "Plane";
    for (std::uint32_t z = 0; z <= segments_z; ++z)
    {
        const float v = static_cast<float>(z) / static_cast<float>(segments_z);
        const float pz = -half_depth + depth * v;
        for (std::uint32_t x = 0; x <= segments_x; ++x)
        {
            const float u = static_cast<float>(x) / static_cast<float>(segments_x);
            mesh.vertices.push_back(
                make_vertex({-half_width + width * u, 0.0f, pz}, {0.0f, 1.0f, 0.0f}, u, v));
        }
    }
    for (std::uint32_t z = 0; z < segments_z; ++z)
    {
        for (std::uint32_t x = 0; x < segments_x; ++x)
        {
            const std::uint32_t a = z * stride + x;
            const std::uint32_t b = a + 1u;
            const std::uint32_t d = a + stride;
            const std::uint32_t c = d + 1u;
            mesh.indices.insert(mesh.indices.end(), {a, b, c, a, c, d});
        }
    }
    return mesh;
}

render::mesh_data make_cube(const cube_mesh_parameters& source)
{
    const float x = safe_dimension(source.width) * 0.5f;
    const float y = safe_dimension(source.height) * 0.5f;
    const float z = safe_dimension(source.depth) * 0.5f;
    const std::uint32_t segments_x = safe_segments(source.segments_x, 1u);
    const std::uint32_t segments_y = safe_segments(source.segments_y, 1u);
    const std::uint32_t segments_z = safe_segments(source.segments_z, 1u);

    render::mesh_data mesh;
    mesh.name = "Cube";
    append_grid_face(mesh, {-x, -y, z}, {x, -y, z}, {-x, y, z}, {0.0f, 0.0f, 1.0f}, segments_x, segments_y);
    append_grid_face(mesh, {x, -y, -z}, {-x, -y, -z}, {x, y, -z}, {0.0f, 0.0f, -1.0f}, segments_x,
                     segments_y);
    append_grid_face(mesh, {-x, -y, -z}, {-x, -y, z}, {-x, y, -z}, {-1.0f, 0.0f, 0.0f}, segments_z,
                     segments_y);
    append_grid_face(mesh, {x, -y, z}, {x, -y, -z}, {x, y, z}, {1.0f, 0.0f, 0.0f}, segments_z, segments_y);
    append_grid_face(mesh, {-x, y, z}, {x, y, z}, {-x, y, -z}, {0.0f, 1.0f, 0.0f}, segments_x, segments_z);
    append_grid_face(mesh, {-x, -y, -z}, {x, -y, -z}, {-x, -y, z}, {0.0f, -1.0f, 0.0f}, segments_x,
                     segments_z);
    return mesh;
}

void append_disc(render::mesh_data& mesh, float radius, float y, float normal_y, std::uint32_t segments)
{
    const std::uint32_t center = static_cast<std::uint32_t>(mesh.vertices.size());
    mesh.vertices.push_back(make_vertex({0.0f, y, 0.0f}, {0.0f, normal_y, 0.0f}, 0.5f, 0.5f));
    const std::uint32_t rim = static_cast<std::uint32_t>(mesh.vertices.size());
    for (std::uint32_t segment = 0; segment <= segments; ++segment)
    {
        const float amount = static_cast<float>(segment) / static_cast<float>(segments);
        const float theta = amount * math::tau<float>;
        const float x = std::cos(theta);
        const float z = std::sin(theta);
        mesh.vertices.push_back(
            make_vertex({x * radius, y, z * radius}, {0.0f, normal_y, 0.0f}, x * 0.5f + 0.5f, z * 0.5f + 0.5f));
    }
    for (std::uint32_t segment = 0; segment < segments; ++segment)
    {
        if (normal_y > 0.0f)
            mesh.indices.insert(mesh.indices.end(), {center, rim + segment + 1u, rim + segment});
        else
            mesh.indices.insert(mesh.indices.end(), {center, rim + segment, rim + segment + 1u});
    }
}

render::mesh_data make_cylinder(const cylinder_mesh_parameters& source)
{
    const float radius = safe_dimension(source.radius);
    const float height = safe_dimension(source.height);
    const std::uint32_t radial_segments = safe_segments(source.radial_segments, 3u);
    const std::uint32_t height_segments = safe_segments(source.height_segments, 1u);
    const float half_height = height * 0.5f;
    const std::uint32_t stride = radial_segments + 1u;

    render::mesh_data mesh;
    mesh.name = "Cylinder";
    for (std::uint32_t row = 0; row <= height_segments; ++row)
    {
        const float v = static_cast<float>(row) / static_cast<float>(height_segments);
        const float y = -half_height + height * v;
        for (std::uint32_t segment = 0; segment <= radial_segments; ++segment)
        {
            const float u = static_cast<float>(segment) / static_cast<float>(radial_segments);
            const float theta = u * math::tau<float>;
            const float nx = std::cos(theta);
            const float nz = std::sin(theta);
            mesh.vertices.push_back(make_vertex({nx * radius, y, nz * radius}, {nx, 0.0f, nz}, u, v));
        }
    }
    for (std::uint32_t row = 0; row < height_segments; ++row)
    {
        for (std::uint32_t segment = 0; segment < radial_segments; ++segment)
        {
            const std::uint32_t a = row * stride + segment;
            const std::uint32_t b = a + 1u;
            const std::uint32_t d = a + stride;
            const std::uint32_t c = d + 1u;
            mesh.indices.insert(mesh.indices.end(), {a, d, c, a, c, b});
        }
    }
    append_disc(mesh, radius, half_height, 1.0f, radial_segments);
    append_disc(mesh, radius, -half_height, -1.0f, radial_segments);
    return mesh;
}

render::mesh_data make_cone(const cone_mesh_parameters& source)
{
    const float radius = safe_dimension(source.radius);
    const float height = safe_dimension(source.height);
    const std::uint32_t radial_segments = safe_segments(source.radial_segments, 3u);
    const std::uint32_t height_segments = safe_segments(source.height_segments, 1u);
    const float half_height = height * 0.5f;
    const float normal_y = radius / height;
    const std::uint32_t stride = radial_segments + 1u;

    render::mesh_data mesh;
    mesh.name = "Cone";
    for (std::uint32_t row = 0; row <= height_segments; ++row)
    {
        const float v = static_cast<float>(row) / static_cast<float>(height_segments);
        const float y = -half_height + height * v;
        const float row_radius = radius * (1.0f - v);
        for (std::uint32_t segment = 0; segment <= radial_segments; ++segment)
        {
            const float u = static_cast<float>(segment) / static_cast<float>(radial_segments);
            const float theta = u * math::tau<float>;
            const float x = std::cos(theta);
            const float z = std::sin(theta);
            const auto normal = math::normalize(math::vector3f{x, normal_y, z});
            mesh.vertices.push_back(
                make_vertex({x * row_radius, y, z * row_radius}, {normal[0], normal[1], normal[2]}, u, v));
        }
    }
    for (std::uint32_t row = 0; row < height_segments; ++row)
    {
        const bool apex_band = row + 1u == height_segments;
        for (std::uint32_t segment = 0; segment < radial_segments; ++segment)
        {
            const std::uint32_t a = row * stride + segment;
            const std::uint32_t b = a + 1u;
            const std::uint32_t d = a + stride;
            const std::uint32_t c = d + 1u;
            if (apex_band)
                mesh.indices.insert(mesh.indices.end(), {a, d, b});
            else
                mesh.indices.insert(mesh.indices.end(), {a, d, c, a, c, b});
        }
    }
    append_disc(mesh, radius, -half_height, -1.0f, radial_segments);
    return mesh;
}

render::mesh_data make_capsule(const capsule_mesh_parameters& source)
{
    const float radius = safe_dimension(source.radius);
    const float height = safe_dimension(source.height);
    const std::uint32_t radial_segments = safe_segments(source.radial_segments, 3u);
    const std::uint32_t hemisphere_rings = safe_segments(source.hemisphere_rings, 2u);
    const std::uint32_t height_segments = safe_segments(source.height_segments, 1u);
    const float half_height = height * 0.5f;

    struct ring
    {
        float center_y{};
        float normal_y{};
        float radial{1.0f};
    };
    std::vector<ring> rings;
    rings.reserve(static_cast<std::size_t>(hemisphere_rings) * 2u + height_segments + 1u);
    for (std::uint32_t index = 0; index <= hemisphere_rings; ++index)
    {
        const float amount = static_cast<float>(index) / static_cast<float>(hemisphere_rings);
        const float latitude = math::pi<float> * 0.5f * (1.0f - amount);
        rings.push_back({half_height, std::sin(latitude), std::cos(latitude)});
    }
    for (std::uint32_t row = 1; row < height_segments; ++row)
    {
        const float amount = static_cast<float>(row) / static_cast<float>(height_segments);
        rings.push_back({half_height - height * amount, 0.0f, 1.0f});
    }
    for (std::uint32_t index = 0; index <= hemisphere_rings; ++index)
    {
        const float amount = static_cast<float>(index) / static_cast<float>(hemisphere_rings);
        const float latitude = -math::pi<float> * 0.5f * amount;
        rings.push_back({-half_height, std::sin(latitude), std::cos(latitude)});
    }

    render::mesh_data mesh;
    mesh.name = "Capsule";
    const std::uint32_t stride = radial_segments + 1u;
    for (std::size_t ring_index = 0; ring_index < rings.size(); ++ring_index)
    {
        const auto& current = rings[ring_index];
        const float v = rings.size() > 1u ? static_cast<float>(ring_index) / static_cast<float>(rings.size() - 1u) : 0.0f;
        for (std::uint32_t segment = 0; segment <= radial_segments; ++segment)
        {
            const float u = static_cast<float>(segment) / static_cast<float>(radial_segments);
            const float theta = u * math::tau<float>;
            const float nx = std::cos(theta) * current.radial;
            const float nz = std::sin(theta) * current.radial;
            mesh.vertices.push_back(make_vertex({nx * radius, current.center_y + current.normal_y * radius, nz * radius},
                                                {nx, current.normal_y, nz}, u, v));
        }
    }
    for (std::size_t ring_index = 0; ring_index + 1u < rings.size(); ++ring_index)
    {
        const auto row = static_cast<std::uint32_t>(ring_index);
        for (std::uint32_t segment = 0; segment < radial_segments; ++segment)
        {
            const std::uint32_t a = row * stride + segment;
            const std::uint32_t b = a + 1u;
            const std::uint32_t d = a + stride;
            const std::uint32_t c = d + 1u;
            mesh.indices.insert(mesh.indices.end(), {a, b, c, a, c, d});
        }
    }
    return mesh;
}

geometric::box3f bounds_for_mesh(const render::mesh_data& mesh)
{
    if (mesh.vertices.empty())
        return geometric::box3f{geometric::point3f{-0.5f, -0.5f, -0.5f}, geometric::point3f{0.5f, 0.5f, 0.5f}};

    math::vector3f minimum{std::numeric_limits<float>::max(), std::numeric_limits<float>::max(),
                           std::numeric_limits<float>::max()};
    math::vector3f maximum{std::numeric_limits<float>::lowest(), std::numeric_limits<float>::lowest(),
                           std::numeric_limits<float>::lowest()};
    for (const auto& value : mesh.vertices)
    {
        for (std::size_t axis = 0; axis < 3; ++axis)
        {
            minimum[axis] = std::min(minimum[axis], value.position[axis]);
            maximum[axis] = std::max(maximum[axis], value.position[axis]);
        }
    }
    return geometric::box3f{geometric::point3f{minimum}, geometric::point3f{maximum}};
}

json parameter_json(const procedural_mesh_parameters& parameters)
{
    return std::visit(
        [](const auto& value) -> json
        {
            using type = std::decay_t<decltype(value)>;
            if constexpr (std::is_same_v<type, plane_mesh_parameters>)
                return {{"width", value.width},
                        {"depth", value.depth},
                        {"segmentsX", value.segments_x},
                        {"segmentsZ", value.segments_z}};
            else if constexpr (std::is_same_v<type, cube_mesh_parameters>)
                return {{"width", value.width},
                        {"height", value.height},
                        {"depth", value.depth},
                        {"segmentsX", value.segments_x},
                        {"segmentsY", value.segments_y},
                        {"segmentsZ", value.segments_z}};
            else if constexpr (std::is_same_v<type, sphere_mesh_parameters>)
                return {{"radius", value.radius}, {"segments", value.segments}, {"rings", value.rings}};
            else if constexpr (std::is_same_v<type, cylinder_mesh_parameters> || std::is_same_v<type, cone_mesh_parameters>)
                return {{"radius", value.radius},
                        {"height", value.height},
                        {"radialSegments", value.radial_segments},
                        {"heightSegments", value.height_segments}};
            else
                return {{"radius", value.radius},
                        {"height", value.height},
                        {"radialSegments", value.radial_segments},
                        {"hemisphereRings", value.hemisphere_rings},
                        {"heightSegments", value.height_segments}};
        },
        parameters);
}

std::optional<procedural_mesh_parameters> deserialize_parameters(const json& serialized)
{
    if (!serialized.is_object()) return std::nullopt;
    const auto type = procedural_mesh_type_from_token(serialized.value("type", std::string{}));
    if (!type) return std::nullopt;

    auto component = procedural_mesh_component{default_procedural_mesh_parameters(*type)};
    const auto& values = serialized.contains("parameters") && serialized["parameters"].is_object()
                             ? serialized["parameters"]
                             : serialized;
    for (const std::string_view name : {"width", "height", "depth", "radius", "segmentsX", "segmentsY",
                                        "segmentsZ", "segments", "rings", "radialSegments", "hemisphereRings",
                                        "heightSegments"})
    {
        const auto found = values.find(std::string{name});
        if (found != values.end() && found->is_number())
            (void)set_procedural_mesh_parameter(component, name, found->get<double>());
    }
    return component.parameters;
}

json unknown_components_for(const editor_scene_state& scene, ecs::entity_guid guid)
{
    const auto found = std::find_if(scene.unknown_component_records.begin(), scene.unknown_component_records.end(),
                                    [guid](const auto& record) { return record.first == guid; });
    if (found == scene.unknown_component_records.end()) return json::object();
    auto parsed = json::parse(found->second, nullptr, false);
    return parsed.is_object() ? std::move(parsed) : json::object();
}

void store_unknown_components(editor_scene_state& scene, ecs::entity_guid guid, json value)
{
    const auto found = std::find_if(scene.unknown_component_records.begin(), scene.unknown_component_records.end(),
                                    [guid](const auto& record) { return record.first == guid; });
    if (value.empty())
    {
        if (found != scene.unknown_component_records.end()) scene.unknown_component_records.erase(found);
        return;
    }
    const auto text = value.dump();
    if (found == scene.unknown_component_records.end())
        scene.unknown_component_records.emplace_back(guid, text);
    else
        found->second = text;
}

std::optional<procedural_mesh_parameters> persisted_parameters(const editor_scene_state& scene, ecs::entity entity)
{
    const auto guid = entity_guid_of(scene, entity);
    if (!guid.valid()) return std::nullopt;
    const auto unknown = unknown_components_for(scene, guid);
    const auto found = unknown.find(std::string{procedural_component_name});
    return found == unknown.end() ? std::nullopt : deserialize_parameters(*found);
}

} // namespace

std::optional<editor_primitive_type> procedural_mesh_type_from_token(std::string_view token) noexcept
{
    if (token == "plane") return editor_primitive_type::plane;
    if (token == "cube") return editor_primitive_type::cube;
    if (token == "sphere") return editor_primitive_type::sphere;
    if (token == "cylinder") return editor_primitive_type::cylinder;
    if (token == "cone") return editor_primitive_type::cone;
    if (token == "capsule") return editor_primitive_type::capsule;
    return std::nullopt;
}

std::optional<editor_primitive_type> procedural_mesh_type_from_name(std::string_view name) noexcept
{
    if (name == "Plane" || name == "plane") return editor_primitive_type::plane;
    if (name == "Cube" || name == "cube") return editor_primitive_type::cube;
    if (name == "Sphere" || name == "sphere") return editor_primitive_type::sphere;
    if (name == "Cylinder" || name == "cylinder") return editor_primitive_type::cylinder;
    if (name == "Cone" || name == "cone") return editor_primitive_type::cone;
    if (name == "Capsule" || name == "capsule") return editor_primitive_type::capsule;
    return std::nullopt;
}

editor_primitive_type procedural_mesh_type(const procedural_mesh_parameters& parameters) noexcept
{
    return std::visit(
        [](const auto& value) noexcept
        {
            using type = std::decay_t<decltype(value)>;
            if constexpr (std::is_same_v<type, plane_mesh_parameters>)
                return editor_primitive_type::plane;
            else if constexpr (std::is_same_v<type, cube_mesh_parameters>)
                return editor_primitive_type::cube;
            else if constexpr (std::is_same_v<type, sphere_mesh_parameters>)
                return editor_primitive_type::sphere;
            else if constexpr (std::is_same_v<type, cylinder_mesh_parameters>)
                return editor_primitive_type::cylinder;
            else if constexpr (std::is_same_v<type, cone_mesh_parameters>)
                return editor_primitive_type::cone;
            else
                return editor_primitive_type::capsule;
        },
        parameters);
}

const char* procedural_mesh_token(editor_primitive_type type) noexcept
{
    switch (type)
    {
        case editor_primitive_type::plane:
            return "plane";
        case editor_primitive_type::cube:
            return "cube";
        case editor_primitive_type::sphere:
            return "sphere";
        case editor_primitive_type::cylinder:
            return "cylinder";
        case editor_primitive_type::cone:
            return "cone";
        case editor_primitive_type::capsule:
            return "capsule";
    }
    return "cube";
}

procedural_mesh_parameters default_procedural_mesh_parameters(editor_primitive_type type)
{
    switch (type)
    {
        case editor_primitive_type::plane:
            return plane_mesh_parameters{};
        case editor_primitive_type::cube:
            return cube_mesh_parameters{};
        case editor_primitive_type::sphere:
            return sphere_mesh_parameters{};
        case editor_primitive_type::cylinder:
            return cylinder_mesh_parameters{};
        case editor_primitive_type::cone:
            return cone_mesh_parameters{};
        case editor_primitive_type::capsule:
            return capsule_mesh_parameters{};
    }
    return cube_mesh_parameters{};
}

render::mesh_data make_procedural_mesh(const procedural_mesh_parameters& parameters)
{
    return std::visit(
        [](const auto& value) -> render::mesh_data
        {
            using type = std::decay_t<decltype(value)>;
            if constexpr (std::is_same_v<type, plane_mesh_parameters>)
                return make_plane(value);
            else if constexpr (std::is_same_v<type, cube_mesh_parameters>)
                return make_cube(value);
            else if constexpr (std::is_same_v<type, sphere_mesh_parameters>)
                return render::make_uv_sphere_mesh(value.radius, safe_segments(value.segments, 3u),
                                                   safe_segments(value.rings, 2u));
            else if constexpr (std::is_same_v<type, cylinder_mesh_parameters>)
                return make_cylinder(value);
            else if constexpr (std::is_same_v<type, cone_mesh_parameters>)
                return make_cone(value);
            else
                return make_capsule(value);
        },
        parameters);
}

bool set_procedural_mesh_parameter(procedural_mesh_component& component, std::string_view parameter, double value)
{
    return std::visit(
        [&](auto& parameters)
        {
            using type = std::decay_t<decltype(parameters)>;
            if constexpr (std::is_same_v<type, plane_mesh_parameters>)
            {
                if (parameter == "width")
                    parameters.width = safe_dimension(value);
                else if (parameter == "depth")
                    parameters.depth = safe_dimension(value);
                else if (parameter == "segmentsX")
                    parameters.segments_x = safe_segments(value, 1u);
                else if (parameter == "segmentsZ")
                    parameters.segments_z = safe_segments(value, 1u);
                else
                    return false;
            }
            else if constexpr (std::is_same_v<type, cube_mesh_parameters>)
            {
                if (parameter == "width")
                    parameters.width = safe_dimension(value);
                else if (parameter == "height")
                    parameters.height = safe_dimension(value);
                else if (parameter == "depth")
                    parameters.depth = safe_dimension(value);
                else if (parameter == "segmentsX")
                    parameters.segments_x = safe_segments(value, 1u);
                else if (parameter == "segmentsY")
                    parameters.segments_y = safe_segments(value, 1u);
                else if (parameter == "segmentsZ")
                    parameters.segments_z = safe_segments(value, 1u);
                else
                    return false;
            }
            else if constexpr (std::is_same_v<type, sphere_mesh_parameters>)
            {
                if (parameter == "radius")
                    parameters.radius = safe_dimension(value);
                else if (parameter == "segments")
                    parameters.segments = safe_segments(value, 3u);
                else if (parameter == "rings")
                    parameters.rings = safe_segments(value, 2u);
                else
                    return false;
            }
            else if constexpr (std::is_same_v<type, cylinder_mesh_parameters> ||
                               std::is_same_v<type, cone_mesh_parameters>)
            {
                if (parameter == "radius")
                    parameters.radius = safe_dimension(value);
                else if (parameter == "height")
                    parameters.height = safe_dimension(value);
                else if (parameter == "radialSegments")
                    parameters.radial_segments = safe_segments(value, 3u);
                else if (parameter == "heightSegments")
                    parameters.height_segments = safe_segments(value, 1u);
                else
                    return false;
            }
            else
            {
                if (parameter == "radius")
                    parameters.radius = safe_dimension(value);
                else if (parameter == "height")
                    parameters.height = safe_dimension(value);
                else if (parameter == "radialSegments")
                    parameters.radial_segments = safe_segments(value, 3u);
                else if (parameter == "hemisphereRings")
                    parameters.hemisphere_rings = safe_segments(value, 2u);
                else if (parameter == "heightSegments")
                    parameters.height_segments = safe_segments(value, 1u);
                else
                    return false;
            }
            return true;
        },
        component.parameters);
}

std::string procedural_mesh_snapshot_json(const procedural_mesh_component& component)
{
    auto result = parameter_json(component.parameters);
    result["type"] = procedural_mesh_token(procedural_mesh_type(component.parameters));
    return result.dump();
}

void persist_procedural_mesh_component(editor_scene_state& scene, ecs::entity entity)
{
    const auto* component = std::as_const(scene.scene).try_get<procedural_mesh_component>(entity);
    if (!component) return;
    ensure_scene_authoring_metadata(scene);
    const auto guid = entity_guid_of(scene, entity);
    if (!guid.valid()) return;

    auto unknown = unknown_components_for(scene, guid);
    unknown[std::string{procedural_component_name}] = {
        {"version", 1u},
        {"type", procedural_mesh_token(procedural_mesh_type(component->parameters))},
        {"parameters", parameter_json(component->parameters)},
    };
    store_unknown_components(scene, guid, std::move(unknown));
}

void clear_procedural_mesh_component(editor_scene_state& scene, ecs::entity entity)
{
    scene.scene.remove<procedural_mesh_component>(entity);
    const auto guid = entity_guid_of(scene, entity);
    if (!guid.valid()) return;
    auto unknown = unknown_components_for(scene, guid);
    unknown.erase(std::string{procedural_component_name});
    store_unknown_components(scene, guid, std::move(unknown));
}

procedural_mesh_component* ensure_procedural_mesh_component(editor_scene_state& scene, ecs::entity entity)
{
    if (!scene.scene.alive(entity)) return nullptr;
    if (auto* existing = scene.scene.try_get<procedural_mesh_component>(entity)) return existing;

    std::optional<procedural_mesh_parameters> parameters = persisted_parameters(scene, entity);
    if (!parameters)
    {
        const auto* binding = find_asset_binding(scene, entity_guid_of(scene, entity));
        if (!binding || binding->source_kind != "primitive") return nullptr;
        const auto type = procedural_mesh_type_from_name(binding->subresource);
        if (!type) return nullptr;
        parameters = default_procedural_mesh_parameters(*type);
    }
    return &scene.scene.emplace<procedural_mesh_component>(entity, procedural_mesh_component{std::move(*parameters)});
}

bool regenerate_procedural_mesh(editor_scene_state& scene, render::renderer& renderer, ecs::entity entity)
{
    auto* procedural = ensure_procedural_mesh_component(scene, entity);
    auto* mesh_renderer = scene.scene.try_get<scene::mesh_renderer_component>(entity);
    if (!procedural || !mesh_renderer) return false;

    const auto mesh = make_procedural_mesh(procedural->parameters);
    const auto mesh_handle = renderer.create_mesh(mesh);
    if (!mesh_handle.valid()) return false;
    mesh_renderer->mesh = mesh_handle;

    const auto local_bounds = bounds_for_mesh(mesh);
    if (auto* bounds = scene.scene.try_get<scene::bounds_component>(entity))
    {
        bounds->local_bounds = local_bounds;
        bounds->dirty = true;
    }
    else
    {
        scene.scene.emplace<scene::bounds_component>(entity, local_bounds, local_bounds, true);
    }
    return true;
}

void synchronize_procedural_mesh_components(editor_scene_state& scene, render::renderer& renderer)
{
    for (const auto& binding : scene.asset_bindings)
    {
        if (binding.source_kind != "primitive") continue;
        const auto entity = find_entity_by_guid(scene, binding.entity);
        if (scene.scene.alive(entity)) (void)regenerate_procedural_mesh(scene, renderer, entity);
    }
}

} // namespace arc::editor
