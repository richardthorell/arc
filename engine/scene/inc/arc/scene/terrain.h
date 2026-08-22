#pragma once

#include <arc/ecs/identity.h>
#include <arc/render/terrain.h>
#include <arc/scene/components.h>

#include <array>
#include <cstdint>
#include <optional>
#include <span>
#include <string>
#include <unordered_map>
#include <vector>

namespace arc::render
{
class renderer;
}

namespace arc::scene
{

/** @brief Default sample resolution used by newly authored terrain. */
inline constexpr std::uint32_t default_terrain_sample_resolution = 257;
/** @brief Legacy serialized chunk size retained for scene compatibility. */
inline constexpr std::uint32_t default_terrain_chunk_quads = 128;
/** @brief Resolutions supported by the terrain authoring and rendering pipeline. */
inline constexpr std::array<std::uint32_t, 5> supported_terrain_resolutions{257u, 513u, 1025u, 2049u, 4097u};

/** @brief Initial sample source used while staging a new terrain. */
enum class terrain_initial_source : std::uint8_t
{
    flat,
    procedural,
    heightmap
};

/** @brief Portable heightmap formats supported by editor import and export. */
enum class terrain_heightmap_format : std::uint8_t
{
    png16,
    raw_r16_le
};

/** @brief Validated settings used to stage a new terrain resource. */
struct terrain_creation_descriptor
{
    float size{180.0f};
    float minimum_elevation{};
    float maximum_elevation{48.0f};
    std::uint32_t sample_resolution{257u};
    std::uint32_t patch_quads{32u};
    terrain_initial_source source{terrain_initial_source::flat};
    std::uint64_t procedural_seed{1u};
};

/** @brief Parameters passed to a registered deterministic terrain generator. */
struct terrain_generation_descriptor
{
    std::string generator_id{"arc.terrain.flat.v1"};
    std::uint64_t seed{1u};
    float minimum_elevation{};
    float maximum_elevation{48.0f};
};

/** @brief Physical-size and resolution changes for an existing terrain. */
struct terrain_resample_descriptor
{
    std::uint32_t sample_resolution{257u};
    float physical_size{180.0f};
};

/** @brief Linear unsigned 16-bit height samples decoded independently of texture import. */
struct terrain_heightmap
{
    std::uint32_t width{};
    std::uint32_t height{};
    std::uint8_t bit_depth{16u};
    std::vector<std::uint16_t> samples;
    std::optional<float> encoded_minimum_elevation;
    std::optional<float> encoded_maximum_elevation;
};

/** @brief Mapping and orientation policy applied while importing a heightmap. */
struct terrain_heightmap_import_settings
{
    std::uint32_t target_resolution{257u};
    float physical_size{180.0f};
    float minimum_elevation{};
    float maximum_elevation{48.0f};
    bool flip_x{};
    bool flip_z{};
    bool normalize_source_range{true};
};

/** @brief Quantization and file-format policy applied while exporting a heightmap. */
struct terrain_heightmap_export_settings
{
    terrain_heightmap_format format{terrain_heightmap_format::png16};
    float minimum_elevation{};
    float maximum_elevation{48.0f};
};

/** @brief Conservative working-set estimate for an authoring operation. */
struct terrain_memory_estimate
{
    std::uint64_t cpu_bytes{};
    std::uint64_t gpu_bytes{};
    std::uint64_t staging_bytes{};
    std::uint64_t history_bytes{};
};

/** @brief Nonthrowing result returned by terrain authoring operations. */
struct [[nodiscard]] terrain_authoring_result
{
    bool succeeded{};
    std::string message;
};

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

[[nodiscard]] bool terrain_heightfield_valid(const terrain_component& terrain) noexcept;
[[nodiscard]] bool terrain_resolution_supported(std::uint32_t resolution) noexcept;
[[nodiscard]] terrain_memory_estimate estimate_terrain_memory(std::uint32_t resolution) noexcept;
[[nodiscard]] terrain_authoring_result
validate_terrain_creation(const terrain_creation_descriptor& descriptor) noexcept;
[[nodiscard]] terrain_authoring_result generate_terrain(terrain_component& terrain,
                                                        const terrain_generation_descriptor& descriptor);
void generate_terrain_heightfield(terrain_component& terrain);
[[nodiscard]] terrain_authoring_result resample_terrain(terrain_component& terrain,
                                                        const terrain_resample_descriptor& descriptor);
[[nodiscard]] terrain_authoring_result import_terrain_heightmap(terrain_component& terrain,
                                                                const terrain_heightmap& heightmap,
                                                                const terrain_heightmap_import_settings& settings);
[[nodiscard]] terrain_heightmap export_terrain_heightmap(const terrain_component& terrain,
                                                         const terrain_heightmap_export_settings& settings);
[[nodiscard]] float sample_terrain_height(const terrain_component& terrain, float local_x, float local_z) noexcept;
[[nodiscard]] math::vector3f sample_terrain_normal(const terrain_component& terrain, float local_x,
                                                   float local_z) noexcept;
terrain_dirty_region apply_terrain_brush(terrain_component& terrain, const math::vector3f& local_center,
                                         const terrain_brush_settings& settings, float delta_seconds = 1.0f / 60.0f);
[[nodiscard]] terrain_raycast_hit raycast_terrain(const terrain_component& terrain, const math::vector3f& local_origin,
                                                  const math::vector3f& local_direction) noexcept;
[[nodiscard]] terrain_raycast_hit raycast_terrain(const terrain_component& terrain,
                                                  const render::terrain_hierarchy& hierarchy,
                                                  const math::vector3f& local_origin,
                                                  const math::vector3f& local_direction) noexcept;

} // namespace arc::scene
