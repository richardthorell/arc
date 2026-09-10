#pragma once

#include <arc/assets/package_artifact_reader.h>
#include <arc/render/virtual_geometry_streaming_io.h>
#include <arc/scene/terrain.h>
#include <arc/scene/terrain_artifacts.h>

#include <cstdint>
#include <vector>

namespace arc::scene
{

/** @brief Result of synchronizing M2.2 terrain region artifacts with generic virtual-geometry streaming. */
struct terrain_streaming_bind_result
{
    bool succeeded{};
    std::uint32_t bound_regions{};
    std::uint32_t skipped_regions{};
};

/**
 * @brief Lifetime owner for terrain-to-generic-virtual-geometry source registrations.
 *
 * Terrain contributes stable region/artifact mapping only. Page request priority, residency, fallback, IO scheduling,
 * decompression and GPU publication remain renderer-owned generic virtual-geometry behavior.
 */
class terrain_virtual_geometry_streaming_binding
{
public:
    [[nodiscard]] terrain_streaming_bind_result
    synchronize(const terrain_cooked_manifest& manifest, const assets::package_artifact_reader& package,
                const terrain_render_proxy& proxy, const render::renderer& renderer,
                render::filesystem_virtual_geometry_artifact_source& source);

    void clear(render::filesystem_virtual_geometry_artifact_source& source) noexcept;

private:
    std::vector<render::virtual_mesh_handle> resources_;
};

} // namespace arc::scene
