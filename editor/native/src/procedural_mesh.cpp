#include <arc/editor/procedural_mesh.h>

#include <arc/editor/editor_state.h>
#include <arc/geometric/box.h>
#include <arc/math/constants.h>
#include <arc/render/primitives.h>

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cctype>
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

struct capsule_ring
{
    float center_y{};
    float normal_y{};
    float radial{1.0f};
};

std::string lowercase(std::string_view value)
{
    std::string result{value};
    std::transform(result.begin(), result.end(), result.begin(),
                   [](unsigned char character) { return static_cast<char>(std::tolower(character)); });
    return result;
}

float dimension(double value)
{
    if (!std::isfinite(value)) return minimum_dimension;
    return std::clamp(static_cast<float>(value), minimum_dimension, maximum_dimension);
}

std::uint32_t segment_count(double value, std::uint32_t minimum)
{
    if (!std::isfinite(value)) return minimum;
    const auto rounded = static_cast<long long>(std::llround(value));
    return static_cast<std::uint32_t>(std::clamp<long long>(rounded, minimum, maximum_segments));
}

render::mesh_vertex procedural_vertex(point3 position, point3 normal, float u, float v)
{
    return {.position = {position.x, position.y, position.z},
            .normal = {normal.x, normal.y, normal.z},
            .texcoord = {u, v},
            .color = {1.0f, 1.0f, 1.0f, 1.0f}};
}

point3 interpolate(point3 start, point3 end, float amount)
{
    return {start.x + (end.x - start.x) * amount, start.y + (end.y - start.y) * amount,
            start.z + (end.z - start.z) * amount};
}

void append_grid_face(render::mesh_data& mesh, point3 p00, point3 p10, point3 p01, point3 normal,
                      std::uint32_t segments_u, std::uint32_t segments_v)
{
    segments_u = std::max(segments_u, 1u);
    segments_v = std::max(segments_v, 1u);
    const auto base = static_cast<std::uint32_t>(mesh.vertices.size());
    const std::uint32_t stride = segments_u + 1u;
    for (std::uint32_t v_index = 0; v_index <= segments_v; ++v_index)
    {
        const float v = static_cast<float>(v_index) / static_cast<float>(segments_v);
        const point3 left = interpolate(p00, p01, v);
        const point3 right = interpolate(p10, interpolate(p10, p01, 1.0f), v);
        for (std::uint32_t u_index = 0; u_index <= segments_u; ++u_index)
        {
            const float u = static_cast<float>(u_index) / static_cast<float>(segments_u);
            mesh.vertices.push_back(procedural_vertex(interpolate(left, right, u), normal, u, v));
        }
    }

    for (std::uint32_t v_index = 0; v_index < segments_v; ++v_index)
    {
        for (std::uint32_t u_index = 0; u_index < segments_u; ++u_index)
        {
            const std::uint32_t a = base + v_index * stride + u_index;
            const std::uint32_t b = a + 1u;
            const std::uint32_t d = a + stride;
            const std::uint32_t c = d + 1u;
            mesh.indices.insert(mesh.indices.end(), {a, b, c, a, c, d});
        }
    }
}

render::mesh_data make_plane(const plane_mesh_parameters& parameters)
{
    const float half_width = parameters.width * 0.5f;
    const float half_depth = parameters.depth * 0.5f;
    render::mesh_data mesh;
    mesh.name = "Plane";

    const std::uint32_t stride = parameters.segments_x + 1u;
    for (std::uint32_t z = 0; z <= parameters.segments_z; ++z)
    {
        const float v = static_cast<float>(z) / static_cast<float>(parameters.segments_z);
        const float pz = -half_depth + parameters.depth * v;
        for (std::uint32_t x = 0; x <= parameters.segments_x; ++x)
        {
            const float u = static_cast<float>(x) / static_cast<float>(parameters.segments_x);
            const float px = -half_width + parameters.width * u;
            mesh.vertices.push_back(procedural_vertex({px, 0.0f, pz}, {0.0f, 1.0f, 0.0f}, u, v));
        }
    }
    for (std::uint32_t z = 0; z < parameters.segments_z; ++z)
    {
        for (std::uint32_t x = 0; x < parameters.segments_x; ++x)
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

render::mesh_data make_cube(const cube_mesh_parameters& parameters)
{
    const float x = parameters.width * 0.5f;
    const float y = parameters.height * 0.5f;
    const float z = parameters.depth * 0.5f;
    render::mesh_data mesh;
    mesh.name = "Cube";

    append_grid_face(mesh, {-x, -y, z}, {x, -y, z}, {-x, y, z}, {0.0f, 0.0f, 1.0f}, parameters.segments_x,
                     parameters.segments_y);
    append_grid_face(mesh, {x, -y, -z}, {-x, -y, -z}, {x, y, -z}, {0.0f, 0.0f, -1.0f}, parameters.segments_x,
                     parameters.segments_y);
    append_grid_face(mesh, {-x, -y, -z}, {-x, -y, z}, {-x, y, -z}, {-1.0f, 0.0f, 0.0f}, parameters.segments_z,
                     parameters.segments_y);
    append_grid_face(mesh, {x, -y, z}, {x, -y, -z}, {x, y, z}, {1.0f, 0.0f, 0.0f}, parameters.segments_z,
                     parameters.segments_y);
    append_grid_face(mesh, {-x, y, z}, {x, y, z}, {-x, y, -z}, {0.0f, 1.0f, 0.0f}, parameters.segments_x,
                     parameters.segments_z);
    append_grid_face(mesh, {-x, -y, -z}, {x, -y, -z}, {-x, -y, z}, {0.0f, -1.0f, 0.0f}, parameters.segments_x,
                     parameters.segments_z);
    return mesh;
}

render::mesh_data make_cylinder(const cylinder_mesh_parameters& parameters)
{
    render::mesh_data mesh;
    mesh.name = "Cylinder";
    const float half_height = parameters.height * 0.5f;
    const std::uint32_t stride = parameters.radial_segments + 1u;

    for (std::uint32_t row = 0; row <= parameters.height_segments; ++row)
    {
        const float v = static_cast<float>(row) / static_cast<float>(parameters.height_segments);
        const float y = -half_height + parameters.height * v;
        for (std::uint32_t segment = 0; segment <= parameters.radial_segments; ++segment)
        {
            const float u = static_cast<float>(segment) / static_cast<float>(parameters.radial_segments);
            const float theta = u * math::tau<float>;
            const float nx = std::cos(theta);
            const float nz = std::sin(theta);
            mesh.vertices.push_back(
                procedural_vertex({nx * parameters.radius, y, nz * parameters.radius}, {nx, 0.0f, nz}, u, v));
        }
    }
    for (std::uint32_t row = 0; row < parameters.height_segments; ++row)
    {
        for (std::uint32_t segment = 0; segment < parameters.radial_segments; ++segment)
        {
            const std::uint32_t a = row * stride + segment;
            const std::uint32_t b = a + 1u;
            const std::uint32_t d = a + stride;
            const std::uint32_t c = d + 1u;
            mesh.indices.insert(mesh.indices.end(), {a, d, c, a, c, b});
        }
    }

    const auto append_cap = [&](float y, float normal_y, bool top)
    {
        const std::uint32_t center = static_cast<std::uint32_t>(mesh.vertices.size());
        mesh.vertices.push_back(procedural_vertex({0.0f, y, 0.0f}, {0.0f, normal_y, 0.0f}, 0.5f, 0.5f));
        const std::uint32_t rim = static_cast<std::uint32_t>(mesh.vertices.size());
        for (std::uint32_t segment = 0; segment <= parameters.radial_segments; ++segment)
        {
            const float amount = static_cast<float>(segment) / static_cast<float>(parameters.radial_segments);
            const float theta = amount * math::tau<float>;
            const float x = std::cos(theta);
            const float z = std::sin(theta);
            mesh.vertices.push_back(procedural_vertex({x * parameters.radius, y, z * parameters.radius},
                                                      {0.0f, normal_y, 0.0f}, x * 0.5f + 0.5f,
                                                      z * 0.5f + 0.5f));
        }
        for (std::uint32_t segment = 0; segment < parameters.radial_segments; ++segment)
        {
            if (top)
                mesh.indices.insert(mesh.indices.end(), {center, rim + segment + 1u, rim + segment});
            else
                mesh.indices.insert(mesh.indices.end(), {center, rim + segment, rim + segment + 1u});
        }
    };
    append_cap(half_height, 1.0f, true);
    append_cap(-half_height, -1.0f, false);
    return mesh;
}

render::mesh_data make_cone(const cone_mesh_parameters& parameters)
{
    render::mesh_data mesh;
    mesh.name = "Cone";
    const float half_height = parameters.height * 0.5f;
    const float normal_y = parameters.radius / parameters.height;
    const std::uint32_t stride = parameters.radial_segments + 1u;

    for (std::uint32_t row = 0; row <= parameters.height_segments; ++row)
    {
        const float v = static_cast<float>(row) / static_cast<float>(parameters.height_segments);
        const float y = -half_height + parameters.height * v;
        const float row_radius = parameters.radius * (1.0f - v);
        for (std::uint32_t segment = 0; segment <= parameters.radial_segments; ++segment)
        {
            const float u = static_cast<float>(segment) / static_cast<float>(parameters.radial_segments);
            const float theta = u * math::tau<float>;
            const float x = std::cos(theta);
            const float z = std::sin(theta);
            const auto normal = math::normalize(math::vector3f{x, normal_y, z});
            mesh.vertices.push_back(procedural_vertex({x * row_radius, y, z * row_radius},
                                                      {normal[0], normal[1], normal[2]}, u, v));
        }
    }
    for (std::uint32_t row = 0; row < parameters.height_segments; ++row)
    {
        const bool apex_band = row + 1u == parameters.height_segments;
        for (std::uint32_t segment = 0; segment < parameters.radial_segments; ++segment)
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

    const std::uint32_t center = static_cast<std::uint32_t>(mesh.vertices.size());
    mesh.vertices.push_back(procedural_vertex({0.0f, -half_height, 0.0f}, {0.0f, -1.0f, 0.0f}, 0.5f, 0.5f));
    const std::uint32_t rim = static_cast<std::uint32_t>(mesh.vertices.size());
    for (std::uint32_t segment = 0; segment <= parameters.radial_segments; ++segment)
    {
        const float amount = static_cast<float>(segment) / static_cast<float>(parameters.radial_segments);
        const float theta = amount * math::tau<float>;
        const float x = std::cos(theta);
        const float z = std::sin(theta);
        mesh.vertices.push_back(procedural_vertex({x * parameters.radius, -half_height, z * parameters.radius},
                                                  {0.0f, -1.0f, 0.0f}, x * 0.5f + 0.5f,
                                                  z * 0.5f + 0.5f));
    }
    for (std::uint32_t segment = 0; segment < parameters.radial_segments; ++segment)
        mesh.indices.insert(mesh.indices.end(), {center, rim + segment, rim + segment + 1u});
    return mesh;
}

render::mesh_data make_capsule(const capsule_mesh_parameters& parameters)
{
    render::mesh_data mesh;
    mesh.name = "Capsule";
    const float half_height = parameters.height * 0.5f;
    std::vector<capsule_ring> rings;
    rings.reserve(parameters.hemisphere_rings * 2u + parameters.height_segments + 1u);

    for (std::uint32_t ring = 0; ring <= parameters.hemisphere_rings; ++ring)
    {
        const float amount = static_cast<float>(ring) / static_cast<float>(parameters.hemisphere_rings);
        const float latitude = math::pi<float> * 0.5f * (1.0f - amount);
        rings.push_back({half_height, std::sin(latitude), std::cos(latitude)});
    }
    for (std::uint32_t row = 1; row < parameters.height_segments; ++row)
    {
        const float amount = static_cast<float>(row) / static_cast<float>(parameters.height_segments);
        rings.push_back({half_height - parameters.height * amount, 0.0f, 1.0f});
    }
    for (std::uint32_t ring = 0; ring <= parameters.hemisphere_rings; ++ring)
    {
        const float amount = static_cast<float>(ring) / static_cast<float>(parameters.hemisphere_rings);
        const float latitude = -math::pi<float> * 0.5f * amount;
        rings.push_back({-half_height, std::sin(latitude), std::cos(latitude)});
    }

    const std::uint32_t stride = parameters.radial_segments + 1u;
    for (std::uint32_t ring_index = 0; ring_index < rings.size(); ++ring_index)
    {
        const auto& ring = rings[ring_index];
        const float v = rings.size() > 1u ? static_cast<float>(ring_index) / static_cast<float>(rings.size() - 1u) : 0.0f;
        for (std::uint32_t segment = 0; segment <= parameters.radial_segments; ++segment)
        {
            const float u = static_cast<float>(segment) / static_cast<float>(parameters.radial_segments);
            const float theta = u * math::tau<float>;
            const float nx = std::cos(theta) * ring.radial;
            const float nz = std::sin(theta) * ring.radial;
            mesh.vertices.push_back(procedural_vertex(
                {nx * parameters.radius, ring.center_y + ring.normal_y * parameters.radius, nz * parameters.radius},
                {nx, ring.normal_y, nz}, u, v));
        }
    }
    for (std::uint32_t ring = 0; ring + 1u < rings.size(); ++ring)
    {
        for (std::uint32_t segment = 0; segment < parameters.radial_segments; ++segment)
        {
            const std::uint32_t a = ring * stride + segment;
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
    for (const auto& vertex : mesh.vertices)
    {
        for (std::size_t axis = 0; axis < 3; ++axis)
        {
            minimum[axis] = std::min(minimum[axis], vertex.position[axis]);
            maximum[axis] = std::max(maximum[axis], vertex.position[axis]);
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
    const auto type_name = serialized.value("type", std::string{});
    const auto type = procedural_mesh_type_from_token(type_name);
    if (!type) return std::nullopt;

    auto component = procedural_mesh_component{default_procedural_mesh_parameters(*type)};
    const auto& values = serialized.contains("parameters") && serialized["parameters"].is_object()
                             ? serialized["parameters"]
                             : serialized;
    const auto apply = [&](std::string_view name)
    {
        const auto found = values.find(std::string{name});
        if (found != values.end() && found->is_number())
            (void)set_procedural_mesh_parameter(component, name, found->get<double>());
    };
    apply("width");
    apply("height");
    apply("depth");
    apply("radius");
    apply("segmentsX");
    apply("segmentsY");
    apply("segmentsZ");
    apply("segments");
    apply("rings");
    apply("radialSegments");
    apply("hemisphereRings");
    apply("heightSegments");
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

std::optional<procedural_mesh_parameters> persisted_parameters(const editor_scene_state& scene, ecs::entity entity)
{
    const auto guid = entity_guid_of(scene, entity);
    if (!guid.valid()) return std::nullopt;
    const auto unknown = unknown_components_for(scene, guid);
    const auto found = unknown.find(std::string{procedural_component_name});
    if (found == unknown.end()) return std::nullopt;
    return deserialize_parameters(*found);
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
    return procedural_mesh_type_from_token(lowercase(name));
}

editor_primitive_type procedural_mesh_type(const procedural_mesh_parameters& parameters) noexcept
{
    return std::visit(
        [](const auto& value) noexcept
        {
            using type = std::decay_t<decltype(value)>;
            if constexpr (std::is_same_v<type, plane_mesh_parameters>) return editor_primitive_type::plane;
            if constexpr (std::is_same_v<type, cube_mesh_parameters>) return editor_primitive_type::cube;
            if constexpr (std::is_same_v<type, sphere_mesh_parameters>) return editor_primitive_type::sphere;
            if constexpr (std::is_same_v<type, cylinder_mesh_parameters>) return editor_primitive_type::cylinder;
            if constexpr (std::is_same_v<type, cone_mesh_parameters>) return editor_primitive_type::cone;
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
                return render::make_uv_sphere_mesh(value.radius, value.segments, value.rings);
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
                    parameters.width = dimension(value);
                else if (parameter == "depth")
                    parameters.depth = dimension(value);
                else if (parameter == "segmentsX")
                    parameters.segments_x = segment_count(value, 1u);
                else if (parameter == "segmentsZ")
                    parameters.segments_z = segment_count(value, 1u);
                else
                    return false;
                return true;
            }
            else if constexpr (std::is_same_v<type, cube_mesh_parameters>)
            {
                if (parameter == "width")
                    parameters.width = dimension(value);
                else if (parameter == "height")
                    parameters.height = dimension(value);
                else if (parameter == "depth")
                    parameters.depth = dimension(value);
                else if (parameter == "segmentsX")
                    parameters.segments_x = segment_count(value, 1u);
                else if (parameter == "segmentsY")
                    parameters.segments_y = segment_count(value, 1u);
                else if (parameter == "segmentsZ")
                    parameters.segments_z = segment_count(value, 1u);
                else
                    return false;
                return true;
            }
            else if constexpr (std::is_same_v<type, sphere_mesh_parameters>)
            {
                if (parameter == "radius")
                    parameters.radius = dimension(value);
                else if (parameter == "segments")
                    parameters.segments = segment_count(value, 3u);
                else if (parameter == "rings")
                    parameters.rings = segment_count(value, 2u);
                else
                    return false;
                return true;
            }
            else if constexpr (std::is_same_v<type, cylinder_mesh_parameters> ||
                               std::is_same_v<type, cone_mesh_parameters>)
            {
                if (parameter == "radius")
                    parameters.radius = dimension(value);
                else if (parameter == "height")
                    parameters.height = dimension(value);
                else if (parameter == "radialSegments")
                    parameters.radial_segments = segment_count(value, 3u);
                else if (parameter == "heightSegments")
                    parameters.height_segments = segment_count(value, 1u);
                else
                    return false;
                return true;
            }
            else
            {
                if (parameter == "radius")
                    parameters.radius = dimension(value);
                else if (parameter == "height")
                    parameters.height = dimension(value);
                else if (parameter == "radialSegments")
                    parameters.radial_segments = segment_count(value, 3u);
                else if (parameter == "hemisphereRings")
                    parameters.hemisphere_rings = segment_count(value, 2u);
                else if (parameter == "heightSegments")
                    parameters.height_segments = segment_count(value, 1u);
                else
                    return false;
                return true;
            }
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
