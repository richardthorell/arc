#pragma once

#include <arc/scene/components.h>
#include <arc/render/mesh.h>

#include <cstdint>

namespace arc::scene
{

inline constexpr std::uint32_t default_terrain_sample_resolution = 257;
inline constexpr std::uint32_t default_terrain_chunk_quads = 128;

enum class terrain_brush_tool : std::uint8_t { sculpt, smooth, flatten, paint };

struct terrain_brush_settings
{
    terrain_brush_tool tool{ terrain_brush_tool::sculpt };
    float radius{ 6.0f };
    float strength{ 0.25f };
    float falloff{ 1.0f };
    std::uint32_t active_layer{};
    bool invert{};
    float flatten_height{};
};

struct terrain_dirty_region
{
    std::uint32_t min_x{};
    std::uint32_t min_z{};
    std::uint32_t max_x{};
    std::uint32_t max_z{};
    bool valid{};
};

struct terrain_raycast_hit
{
    math::vector3f position{};
    math::vector3f normal{ 0.0f, 1.0f, 0.0f };
    float distance{};
    bool hit{};
};

bool terrain_heightfield_valid(const terrain_component& terrain) noexcept;
void generate_terrain_heightfield(terrain_component& terrain);
float sample_terrain_height(const terrain_component& terrain, float local_x, float local_z) noexcept;
math::vector3f sample_terrain_normal(const terrain_component& terrain, float local_x, float local_z) noexcept;
render::mesh_data make_terrain_chunk_mesh(
    const terrain_component& terrain,
    std::uint32_t chunk_x,
    std::uint32_t chunk_z);
terrain_dirty_region apply_terrain_brush(
    terrain_component& terrain,
    const math::vector3f& local_center,
    const terrain_brush_settings& settings,
    float delta_seconds = 1.0f / 60.0f);
terrain_raycast_hit raycast_terrain(
    const terrain_component& terrain,
    const math::vector3f& local_origin,
    const math::vector3f& local_direction) noexcept;

}
