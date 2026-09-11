#include <arc/editor/terrain_rebuild.h>
#include <arc/editor/terrain_heightmap_io.h>
#include <arc/scene/terrain_render_regions.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <utility>

namespace arc::editor
{
namespace
{

scene::terrain_evaluation_result evaluate_region(const scene::terrain_asset& asset, scene::terrain_region_id id,
                                                 float size, std::uint32_t quads, float height_scale,
                                                 const std::filesystem::path& source_path)
{
    scene::terrain_evaluation_result result;
    result.region = id;
    result.build_snapshot = scene::make_terrain_build_region_snapshot(asset, id);
    const auto fail = [&](std::string message)
    {
        result.diagnostics.push_back({scene::terrain_evaluation_diagnostic_severity::error,
                                      scene::terrain_evaluation_diagnostic_code::missing_source_data,
                                      {},
                                      std::move(message)});
        return result;
    };
    if (quads == 0u || size <= 0.0f) return fail("Terrain source has no sample grid");
    const auto& transform = asset.source.transform;
    if ((transform.scale[0] != 1.0f || transform.scale[1] != 1.0f || transform.scale[2] != 1.0f) ||
        transform.translation[0] != 0.0f || transform.translation[2] != 0.0f ||
        std::abs(transform.rotation.x()) > 1.0e-6f || std::abs(transform.rotation.y()) > 1.0e-6f ||
        std::abs(transform.rotation.z()) > 1.0e-6f)
        return fail("Terrain region source transforms require an unscaled, axis-aligned sample grid");

    const auto bounds = result.build_snapshot.authoring_bounds;
    const double spacing = static_cast<double>(size) / quads;
    const double half = static_cast<double>(size) * 0.5;
    const auto index = [&](double value)
    {
        return static_cast<std::uint32_t>(
            std::clamp(std::round((value + half) / spacing), 0.0, static_cast<double>(quads)));
    };
    const auto x0 = index(bounds.min_x), x1 = index(bounds.max_x);
    const auto z0 = index(bounds.min_z), z1 = index(bounds.max_z);
    if (x0 >= x1 || z0 >= z1) return fail("Terrain region is outside the source grid");
    // One quad of evaluated halo supplies identical triangle normals at shared region vertices.
    const auto hx0 = x0 > 0u ? x0 - 1u : 0u, hx1 = std::min(x1 + 1u, quads);
    const auto hz0 = z0 > 0u ? z0 - 1u : 0u, hz1 = std::min(z1 + 1u, quads);
    const auto width = hx1 - hx0 + 1u, depth = hz1 - hz0 + 1u;
    std::vector<float> heights(static_cast<std::size_t>(width) * depth);
    std::vector<std::array<std::uint8_t, 4>> weights(heights.size(), {255u, 0u, 0u, 0u});
    if (asset.source.kind == scene::terrain_source_kind::heightfield)
    {
        scene::terrain_heightmap decoded;
        const auto loaded = load_terrain_heightmap(source_path, quads + 1u, quads + 1u, decoded);
        if (!loaded.succeeded) return fail(loaded.message);
        if (decoded.width != quads + 1u || decoded.height != quads + 1u)
            return fail("Terrain source resolution must match the authoring grid");
        const auto minimum = decoded.encoded_minimum_elevation.value_or(0.0f);
        const auto maximum = decoded.encoded_maximum_elevation.value_or(height_scale);
        for (std::uint32_t z = hz0; z <= hz1; ++z)
            for (std::uint32_t x = hx0; x <= hx1; ++x)
                heights[static_cast<std::size_t>(z - hz0) * width + x - hx0] =
                    minimum +
                    (maximum - minimum) *
                        (static_cast<float>(decoded.samples[static_cast<std::size_t>(z) * decoded.width + x]) /
                         65535.0f);
    }
    const double origin_x = (bounds.min_x + bounds.max_x) * 0.5;
    const double origin_z = (bounds.min_z + bounds.max_z) * 0.5;
    scene::terrain_evaluation_request request;
    request.region = id;
    request.heightfield_source = scene::terrain_heightfield_source_view{width,
                                                                        depth,
                                                                        static_cast<float>((hx1 - hx0) * spacing),
                                                                        static_cast<float>((hz1 - hz0) * spacing),
                                                                        heights,
                                                                        weights,
                                                                        asset.authoring_revision};
    request.source_bounds =
        scene::terrain_world_bounds{-half + hx0 * spacing - origin_x, 0.0, -half + hz0 * spacing - origin_z,
                                    -half + hx1 * spacing - origin_x, 0.0, -half + hz1 * spacing - origin_z};
    result = scene::make_default_terrain_evaluator().evaluate(asset, request);
    if (!result.succeeded) return result;
    auto padded = scene::build_terrain_render_regions(result.surface.view(), std::numeric_limits<double>::max());
    if (padded.size() != 1u) return fail("Terrain normal halo could not be compiled");
    const auto& source = std::get<scene::terrain_evaluated_heightfield>(result.surface.geometry);
    scene::terrain_evaluated_heightfield cropped;
    cropped.sample_width = x1 - x0 + 1u;
    cropped.sample_height = z1 - z0 + 1u;
    cropped.width = static_cast<float>((x1 - x0) * spacing);
    cropped.depth = static_cast<float>((z1 - z0) * spacing);
    float minimum = std::numeric_limits<float>::max(), maximum = std::numeric_limits<float>::lowest();
    for (std::uint32_t z = z0; z <= z1; ++z)
        for (std::uint32_t x = x0; x <= x1; ++x)
        {
            const auto sample = static_cast<std::size_t>(z - hz0) * width + x - hx0;
            const auto height = source.heights[sample] + static_cast<float>(asset.coordinates.origin_y);
            cropped.heights.push_back(height);
            cropped.material_weights.push_back(source.material_weights[sample]);
            result.vertex_normals.push_back(padded.front().vertex_normals[sample]);
            minimum = std::min(minimum, height);
            maximum = std::max(maximum, height);
        }
    result.surface.geometry = std::move(cropped);
    result.surface.local_bounds = {-half + x0 * spacing, minimum, -half + z0 * spacing,
                                   -half + x1 * spacing, maximum, -half + z1 * spacing};
    result.content_fingerprint = scene::terrain_surface_fingerprint(result.surface.view());
    return result;
}

} // namespace

terrain_rebuild_session::terrain_rebuild_session(scene::terrain_asset asset, const scene::terrain_component& terrain,
                                                 std::filesystem::path source_path)
    : size_(terrain.size), spacing_(static_cast<double>(terrain.size) / std::max(terrain.subdivisions, 1u))
{
    evaluate_ = [size = terrain.size, quads = terrain.subdivisions, scale = terrain.height_scale,
                 path = std::move(source_path)](const scene::terrain_asset& snapshot, scene::terrain_region_id id)
    { return evaluate_region(snapshot, id, size, quads, scale, path); };
    update(std::move(asset), true);
}

void terrain_rebuild_session::update(scene::terrain_asset asset, bool invalidate_all)
{
    const double half = static_cast<double>(size_) * 0.5;
    // Exclude the maximum boundary: it is a shared vertex, not an additional strip of quads.
    const scene::terrain_world_bounds extent{
        -half, 0.0, -half, std::nextafter(half, -half), 0.0, std::nextafter(half, -half)};
    const auto regions = scene::terrain_regions_overlapping(asset.coordinates, asset.partition, extent);
    std::vector<scene::terrain_region_id> dirty;
    for (const auto id : regions)
    {
        auto& next = scene::ensure_terrain_region(asset, id);
        const auto previous = std::find_if(asset_.regions.begin(), asset_.regions.end(),
                                           [id](const auto& region) { return region.id == id; });
        if (!invalidate_all && previous != asset_.regions.end() && previous->dirty_revision == next.dirty_revision)
        {
            next.compiled_revision = previous->compiled_revision;
            next.dirty_domains = previous->dirty_domains;
        }
        if (invalidate_all || (next.dirty_domains & scene::terrain_domain::geometry) != scene::terrain_domain::none)
            dirty.push_back(id);
    }
    // Seam positions and normals read adjacent samples. Rebuild only overlapping dependency halos.
    for (const auto id : regions)
    {
        auto& next = scene::ensure_terrain_region(asset, id);
        const auto halo =
            scene::expand_terrain_bounds(next.authoring_bounds, std::max(asset.partition.dependency_halo, spacing_));
        for (const auto changed : dirty)
        {
            const auto bounds = scene::terrain_region_bounds(asset.coordinates, asset.partition, changed);
            if (halo.min_x <= bounds.max_x && halo.max_x >= bounds.min_x && halo.min_z <= bounds.max_z &&
                halo.max_z >= bounds.min_z)
            {
                next.dirty_revision = asset.authoring_revision;
                next.dirty_domains |= scene::terrain_domain::geometry;
                break;
            }
        }
    }
    std::erase_if(asset.regions, [&](const auto& region)
                  { return std::find(regions.begin(), regions.end(), region.id) == regions.end(); });
    asset_ = std::move(asset);
    ready_.reset();
    failed_revision_ = 0u;
    error_.clear();
}

bool terrain_rebuild_session::pump(jobs::job_system& jobs, scene::terrain_render_proxy_cache& proxies,
                                   ecs::entity_guid guid, scene::terrain_component& terrain, render::renderer& renderer)
{
    if (!ready_) ready_ = queue_.take_ready(asset_);
    if (ready_)
    {
        if (ready_->stale)
            ready_.reset();
        else if (!ready_->succeeded)
        {
            error_ = "Terrain region build failed";
            for (const auto& region : ready_->regions)
                if (!region.evaluation.diagnostics.empty()) error_ = region.evaluation.diagnostics.front().message;
            failed_revision_ = asset_.authoring_revision;
            ready_.reset();
        }
        else if (proxies.publish(guid, *ready_, terrain, renderer))
        {
            // Keep the picking/sculpt cache in agreement with the asset-owned result, without serializing products.
            for (const auto& region : ready_->regions)
            {
                const auto& surface = region.evaluation.surface;
                const auto& grid = std::get<scene::terrain_evaluated_heightfield>(surface.geometry);
                const auto x0 =
                    static_cast<std::uint32_t>(std::llround((surface.local_bounds.min_x + size_ * 0.5) / spacing_));
                const auto z0 =
                    static_cast<std::uint32_t>(std::llround((surface.local_bounds.min_z + size_ * 0.5) / spacing_));
                for (std::uint32_t z = 0; z < grid.sample_height; ++z)
                    for (std::uint32_t x = 0; x < grid.sample_width; ++x)
                        terrain.heights[static_cast<std::size_t>(z0 + z) * (terrain.subdivisions + 1u) + x0 + x] =
                            grid.heights[static_cast<std::size_t>(z) * grid.sample_width + x];
                (void)scene::mark_terrain_region_compiled(
                    asset_, region.evaluation.region, scene::terrain_domain::geometry | scene::terrain_domain::topology,
                    region.evaluation.build_snapshot.target_dirty_revision);
            }
            terrain.asset_authoring_revision = asset_.authoring_revision;
            ready_.reset();
            error_.clear();
            return true;
        }
        else
            error_ = "Terrain region upload failed; retaining the visible generation and retrying";
    }
    if (!ready_ && failed_revision_ != asset_.authoring_revision) (void)queue_.schedule(jobs, asset_, evaluate_);
    return false;
}

} // namespace arc::editor
