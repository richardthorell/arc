#pragma once

#include <arc/assets/assets.h>
#include <arc/ecs/world.h>
#include <arc/render/lighting_scene.h>
#include <arc/render/renderer.h>

#include <filesystem>
#include <string>
#include <vector>

namespace arc::scene
{

struct scene_irradiance_probe_bake_request
{
    ecs::entity volume_entity{};
    std::filesystem::path output_path;
    render::irradiance_probe_bake_options options{};
};

struct [[nodiscard]] scene_irradiance_probe_bake_result
{
    bool succeeded{};
    assets::asset_reference asset{.expected_type = assets::asset_types::irradiance_probe_volume};
    render::irradiance_probe_bake_statistics statistics{};
    std::uint32_t static_meshes{};
    std::uint32_t static_triangles{};
    std::uint32_t static_directional_lights{};
    std::vector<std::string> diagnostics;
    std::string error;
};

/**
 * @brief Gather bake-eligible static scene data, bake the selected volume, write its .arcprobe asset,
 *        and refresh the component binding.
 *
 * Only static/stationary renderers/lights participate. Movable objects and lights are intentionally
 * excluded so runtime motion cannot invalidate baked lighting semantics.
 */
[[nodiscard]] scene_irradiance_probe_bake_result
bake_scene_irradiance_probe_volume(ecs::world& world, render::renderer& renderer, assets::asset_manager& assets,
                                   const scene_irradiance_probe_bake_request& request);

} // namespace arc::scene
