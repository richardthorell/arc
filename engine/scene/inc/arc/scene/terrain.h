#pragma once

#include <arc/ecs/identity.h>
#include <arc/render/terrain.h>
#include <arc/scene/components.h>

#include <cstdint>
#include <span>
#include <unordered_map>

namespace arc::render
{
class renderer;
}

namespace arc::scene
{

inline constexpr std::uint32_t default_terrain_sample_resolution = 257;
inline constexpr std::uint32_t default_terrain_chunk_quads = 128;

enum class terrain_brush_tool : std::uint8_t
{
    sculpt,
    smooth,
    flatten,
    paint
};

struct terrain_brush_settings
{
    terrain_brush_tool tool{terrain_brush_tool::sculpt};
    float radius{6.0f};
    float strength{0.25f};
    float falloff{1.0f};
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
    bool heights_changed{};
    bool weights_changed{};
};

/** @brief Nonserialized renderer proxy state for one terrain entity. */
struct terrain_render_proxy
{
    render::terrain_handle handle{};
    std::uint64_t synchronized_revision{};
    render::material_handle material{};
};

/** @brief Per-world terrain resource cache keyed by persistent entity identity. */
class terrain_render_proxy_cache
{
public:
    [[nodiscard]] terrain_render_proxy* find(ecs::entity_guid guid) noexcept;
    [[nodiscard]] const terrain_render_proxy* find(ecs::entity_guid guid) const noexcept;
    bool synchronize(ecs::entity_guid guid, const terrain_component& terrain, render::renderer& renderer,
                     const terrain_dirty_region* dirty_region = nullptr);
    void release_missing(std::span<const ecs::entity_guid> active, render::renderer& renderer);
    bool erase(ecs::entity_guid guid, render::renderer& renderer);
    void clear(render::renderer& renderer);

private:
    std::unordered_map<ecs::entity_guid, terrain_render_proxy, ecs::entity_guid_hash> proxies_;
};

struct terrain_raycast_hit
{
    math::vector3f position{};
    math::vector3f normal{0.0f, 1.0f, 0.0f};
    float distance{};
    bool hit{};
};

bool terrain_heightfield_valid(const terrain_component& terrain) noexcept;
void generate_terrain_heightfield(terrain_component& terrain);
float sample_terrain_height(const terrain_component& terrain, float local_x, float local_z) noexcept;
math::vector3f sample_terrain_normal(const terrain_component& terrain, float local_x, float local_z) noexcept;
terrain_dirty_region apply_terrain_brush(terrain_component& terrain, const math::vector3f& local_center,
                                         const terrain_brush_settings& settings, float delta_seconds = 1.0f / 60.0f);
terrain_raycast_hit raycast_terrain(const terrain_component& terrain, const math::vector3f& local_origin,
                                    const math::vector3f& local_direction) noexcept;

} // namespace arc::scene
