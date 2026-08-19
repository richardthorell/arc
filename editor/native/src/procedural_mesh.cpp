#include <arc/editor/procedural_mesh.h>

#include <arc/editor/editor_state.h>
#include <arc/geometric/box.h>
#include <arc/render/primitives.h>

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cctype>
#include <cmath>
#include <limits>
#include <string>
#include <type_traits>

namespace arc::editor
{
namespace
{
using json = nlohmann::json;

constexpr std::string_view procedural_component_name = "ProceduralMesh";
constexpr float minimum_dimension = 0.001f;
constexpr float maximum_dimension = 100000.0f;
constexpr std::uint32_t maximum_segments = 512u;

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
            if constexpr (std::is_same_v<type, plane_mesh_parameters> || std::is_same_v<type, cube_mesh_parameters>)
                return {{"size", value.size}};
            else if constexpr (std::is_same_v<type, sphere_mesh_parameters>)
                return {{"radius", value.radius}, {"segments", value.segments}, {"rings", value.rings}};
            else if constexpr (std::is_same_v<type, cylinder_mesh_parameters> || std::is_same_v<type, cone_mesh_parameters>)
                return {{"radius", value.radius}, {"height", value.height}, {"radialSegments", value.radial_segments}};
            else
                return {{"radius", value.radius},
                        {"height", value.height},
                        {"radialSegments", value.radial_segments},
                        {"hemisphereRings", value.hemisphere_rings}};
        },
        parameters);
}

std::optional<procedural_mesh_parameters> deserialize_parameters(const json& serialized)
{
    if (!serialized.is_object()) return std::nullopt;
    const auto type_name = serialized.value("type", std::string{});
    const auto type = procedural_mesh_type_from_token(type_name);
    if (!type) return std::nullopt;

    auto result = default_procedural_mesh_parameters(*type);
    const auto& values = serialized.contains("parameters") && serialized["parameters"].is_object()
                             ? serialized["parameters"]
                             : serialized;
    auto component = procedural_mesh_component{result};
    const auto apply = [&](std::string_view name)
    {
        const auto found = values.find(std::string{name});
        if (found != values.end() && found->is_number())
            (void)set_procedural_mesh_parameter(component, name, found->get<double>());
    };
    apply("size");
    apply("radius");
    apply("height");
    apply("segments");
    apply("rings");
    apply("radialSegments");
    apply("hemisphereRings");
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
    const auto token = lowercase(name);
    return procedural_mesh_type_from_token(token);
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
                return render::make_plane_mesh(value.size);
            else if constexpr (std::is_same_v<type, cube_mesh_parameters>)
                return render::make_cube_mesh(value.size);
            else if constexpr (std::is_same_v<type, sphere_mesh_parameters>)
                return render::make_uv_sphere_mesh(value.radius, value.segments, value.rings);
            else if constexpr (std::is_same_v<type, cylinder_mesh_parameters>)
                return render::make_cylinder_mesh(value.radius, value.height, value.radial_segments);
            else if constexpr (std::is_same_v<type, cone_mesh_parameters>)
                return render::make_cone_mesh(value.radius, value.height, value.radial_segments);
            else
                return render::make_capsule_mesh(value.radius, value.height, value.radial_segments,
                                                 value.hemisphere_rings);
        },
        parameters);
}

bool set_procedural_mesh_parameter(procedural_mesh_component& component, std::string_view parameter, double value)
{
    return std::visit(
        [&](auto& parameters)
        {
            using type = std::decay_t<decltype(parameters)>;
            if constexpr (std::is_same_v<type, plane_mesh_parameters> || std::is_same_v<type, cube_mesh_parameters>)
            {
                if (parameter != "size") return false;
                parameters.size = dimension(value);
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
                    parameters.hemisphere_rings = segment_count(value, 1u);
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
