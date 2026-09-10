#include <arc/scene/terrain_cooked_storage.h>

#include <arc/render/virtual_geometry_artifact.h>

#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <array>
#include <cstdint>
#include <string>
#include <vector>

namespace
{

arc::scene::terrain_evaluated_surface make_surface()
{
    arc::scene::terrain_evaluated_surface surface;
    surface.source_revision = 7u;
    surface.local_bounds = {0.0, 0.0, 0.0, 512.0, 4.0, 512.0};
    arc::scene::terrain_evaluated_heightfield heightfield;
    heightfield.sample_width = 5u;
    heightfield.sample_height = 5u;
    heightfield.width = 512.0f;
    heightfield.depth = 512.0f;
    heightfield.heights.resize(25u);
    heightfield.material_weights.resize(25u, std::array<std::uint8_t, 4>{255u, 0u, 0u, 0u});
    for (std::uint32_t z = 0; z < 5u; ++z)
        for (std::uint32_t x = 0; x < 5u; ++x)
            heightfield.heights[static_cast<std::size_t>(z) * 5u + x] = static_cast<float>((x + z) % 5u);
    surface.geometry = std::move(heightfield);
    return surface;
}

} // namespace

TEST_CASE("M2.2 cooks terrain into independently addressable region artifacts")
{
    using namespace arc;
    const auto terrain = assets::generate_asset_guid();
    const auto surface = make_surface();
    const auto cooked = scene::build_terrain_cooked_storage(terrain, surface.view(), 11u);
    REQUIRE(cooked);

    const auto& storage = cooked.value();
    REQUIRE(scene::validate_terrain_cooked_manifest(storage.manifest));
    REQUIRE(storage.manifest.regions.size() == 4u);
    REQUIRE(storage.artifacts.size() == 9u);

    for (const auto& region : storage.manifest.regions)
    {
        const auto* render = scene::find_terrain_artifact(storage.manifest, region.region,
                                                          scene::terrain_artifact_kind::render_geometry);
        const auto* fallback = scene::find_terrain_artifact(storage.manifest, region.region,
                                                            scene::terrain_artifact_kind::fallback_geometry);
        REQUIRE(render != nullptr);
        REQUIRE(fallback != nullptr);
        REQUIRE(render->generation != 0u);
        REQUIRE(render->payload_size != 0u);
        REQUIRE_FALSE(render->pages.empty());
        REQUIRE(std::ranges::any_of(render->pages, [](const auto& page) { return page.root; }));

        const auto artifact = std::ranges::find_if(storage.artifacts,
                                                   [&](const auto& value) { return value.name == render->storage_key; });
        REQUIRE(artifact != storage.artifacts.end());
        REQUIRE(artifact->schema == assets::artifact_schemas::virtual_geometry);
        REQUIRE(artifact->bytes.size() == render->payload_size);

        const auto inspected = render::inspect_virtual_geometry_artifact(artifact->bytes);
        REQUIRE(inspected);
        REQUIRE(inspected.value().meshes.size() == 1u);
        const auto& index = inspected.value().meshes.front();
        REQUIRE(index.metadata_offset == render->metadata_offset);
        REQUIRE(index.metadata_size == render->metadata_size);
        REQUIRE(index.pages.size() == render->pages.size());
        for (std::size_t page_index = 0; page_index < index.pages.size(); ++page_index)
        {
            CHECK(index.pages[page_index].offset == render->pages[page_index].offset);
            CHECK(index.pages[page_index].stored_size == render->pages[page_index].stored_size);
            CHECK(index.pages[page_index].content_hash == render->pages[page_index].content_hash);
        }
    }
}

TEST_CASE("terrain cooked manifest is lightweight and round trips without detailed geometry payloads")
{
    using namespace arc;
    const auto terrain = assets::generate_asset_guid();
    const auto surface = make_surface();
    const auto cooked = scene::build_terrain_cooked_storage(terrain, surface.view(), 11u);
    REQUIRE(cooked);

    const auto& artifacts = cooked.value().artifacts;
    const auto manifest_artifact =
        std::ranges::find_if(artifacts, [](const auto& value) { return value.schema == assets::artifact_schemas::terrain_manifest; });
    REQUIRE(manifest_artifact != artifacts.end());

    std::uint64_t detailed_bytes{};
    for (const auto& artifact : artifacts)
        if (artifact.schema == assets::artifact_schemas::virtual_geometry) detailed_bytes += artifact.bytes.size();
    REQUIRE(detailed_bytes > manifest_artifact->bytes.size());

    const auto decoded = scene::decode_terrain_cooked_manifest(manifest_artifact->bytes);
    REQUIRE(decoded);
    REQUIRE(decoded.value().terrain == terrain);
    REQUIRE(decoded.value().authoring_revision == 11u);
    REQUIRE(decoded.value().regions.size() == 4u);
    for (const auto& region : decoded.value().regions)
    {
        const auto* render = scene::find_terrain_artifact(decoded.value(), region.region,
                                                          scene::terrain_artifact_kind::render_geometry);
        REQUIRE(render != nullptr);
        REQUIRE(render->payload_size > 0u);
        REQUIRE_FALSE(render->pages.empty());
    }
}
