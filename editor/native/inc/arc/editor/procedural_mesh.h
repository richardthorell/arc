#pragma once

#include <arc/ecs/entity.h>
#include <arc/render/render.h>

#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <variant>

namespace arc::editor
{

struct editor_scene_state;
enum class editor_primitive_type : std::uint8_t;

struct plane_mesh_parameters
{
    float width{4.0f};
    float depth{4.0f};
    std::uint32_t segments_x{1};
    std::uint32_t segments_z{1};

    friend constexpr bool operator==(const plane_mesh_parameters&, const plane_mesh_parameters&) noexcept = default;
};

struct cube_mesh_parameters
{
    float width{1.0f};
    float height{1.0f};
    float depth{1.0f};
    std::uint32_t segments_x{1};
    std::uint32_t segments_y{1};
    std::uint32_t segments_z{1};

    friend constexpr bool operator==(const cube_mesh_parameters&, const cube_mesh_parameters&) noexcept = default;
};

struct sphere_mesh_parameters
{
    float radius{0.5f};
    std::uint32_t segments{32};
    std::uint32_t rings{16};

    friend constexpr bool operator==(const sphere_mesh_parameters&, const sphere_mesh_parameters&) noexcept = default;
};

struct cylinder_mesh_parameters
{
    float radius{0.5f};
    float height{1.0f};
    std::uint32_t radial_segments{32};
    std::uint32_t height_segments{1};

    friend constexpr bool operator==(const cylinder_mesh_parameters&, const cylinder_mesh_parameters&) noexcept = default;
};

struct cone_mesh_parameters
{
    float radius{0.5f};
    float height{1.0f};
    std::uint32_t radial_segments{32};
    std::uint32_t height_segments{1};

    friend constexpr bool operator==(const cone_mesh_parameters&, const cone_mesh_parameters&) noexcept = default;
};

struct capsule_mesh_parameters
{
    float radius{0.5f};
    float height{1.0f};
    std::uint32_t radial_segments{32};
    std::uint32_t hemisphere_rings{8};
    std::uint32_t height_segments{1};

    friend constexpr bool operator==(const capsule_mesh_parameters&, const capsule_mesh_parameters&) noexcept = default;
};

using procedural_mesh_parameters =
    std::variant<plane_mesh_parameters, cube_mesh_parameters, sphere_mesh_parameters, cylinder_mesh_parameters,
                 cone_mesh_parameters, capsule_mesh_parameters>;

struct procedural_mesh_component
{
    procedural_mesh_parameters parameters{cube_mesh_parameters{}};

    friend bool operator==(const procedural_mesh_component&, const procedural_mesh_component&) noexcept = default;
};

std::optional<editor_primitive_type> procedural_mesh_type_from_token(std::string_view token) noexcept;
std::optional<editor_primitive_type> procedural_mesh_type_from_name(std::string_view name) noexcept;
editor_primitive_type procedural_mesh_type(const procedural_mesh_parameters& parameters) noexcept;
const char* procedural_mesh_token(editor_primitive_type type) noexcept;
procedural_mesh_parameters default_procedural_mesh_parameters(editor_primitive_type type);
render::mesh_data make_procedural_mesh(const procedural_mesh_parameters& parameters);

bool set_procedural_mesh_parameter(procedural_mesh_component& component, std::string_view parameter, double value);
std::string procedural_mesh_snapshot_json(const procedural_mesh_component& component);

void persist_procedural_mesh_component(editor_scene_state& scene, ecs::entity entity);
void clear_procedural_mesh_component(editor_scene_state& scene, ecs::entity entity);
procedural_mesh_component* ensure_procedural_mesh_component(editor_scene_state& scene, ecs::entity entity);
bool regenerate_procedural_mesh(editor_scene_state& scene, render::renderer& renderer, ecs::entity entity);
void synchronize_procedural_mesh_components(editor_scene_state& scene, render::renderer& renderer);

} // namespace arc::editor
