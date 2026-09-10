#include <arc/assets/cook.h>
#include <arc/assets/package_artifact_reader.h>
#include <arc/assets/terrain_types.h>
#include <arc/io/io.h>
#include <arc/jobs/jobs.h>
#include <arc/render/renderer.h>
#include <arc/render/virtual_geometry_streaming_io.h>
#include <arc/scene/terrain_cooked_storage.h>
#include <arc/scene/terrain_streaming.h>

#include <catch2/catch_test_macros.hpp>

#include <array>
#include <cstdint>
#include <filesystem>

namespace
{

arc::scene::terrain_evaluated_surface make_streaming_surface()
{
    arc::scene::terrain_evaluated_surface surface;
    surface.source_revision = 9u;
    surface.local_bounds = {0.0, 0.0, 0.0, 512.0, 6.0, 512.0};
    arc::scene::terrain_evaluated_heightfield heightfield;
    heightfield.sample_width = 5u;
    heightfield.sample_height = 5u;
    heightfield.width = 512.0f;
    heightfield.depth = 512.0f;
    heightfield.heights.resize(25u);
    heightfield.material_weights.resize(25u, std::array<std::uint8_t, 4>{255u, 0u, 0u, 0u});
    for (std::uint32_t z = 0; z < 5u; ++z)
        for (std::uint32_t x = 0; x < 5u; ++x)
            heightfield.heights[static_cast<std::size_t>(z) * 5u + x] = static_cast<float>((x * 3u + z * 5u) % 7u);
    surface.geometry = std::move(heightfield);
    return surface;
}

} // namespace

TEST_CASE("M2.3 binds cooked terrain regions to exact asynchronous package page ranges")
{
    using namespace arc;
    const auto terrain = assets::generate_asset_guid();
    const auto evaluated = make_streaming_surface();
    const auto cooked = scene::build_terrain_cooked_storage(terrain, evaluated.view(), 17u);
    REQUIRE(cooked);

    const auto root = std::filesystem::temp_directory_path() / ("arc-m2-3-terrain-" + assets::to_string(terrain));
    struct cleanup
    {
        std::filesystem::path root;
        ~cleanup()
        {
            std::error_code error;
            std::filesystem::remove_all(root, error);
        }
    } cleanup_guard{root};

    assets::derived_data_cache cache({.root = root / "cache"});
    assets::cook_manifest package_manifest;
    package_manifest.target = assets::windows_vulkan_cook_target();
    package_manifest.build_id = "m2-3-terrain-streaming";
    package_manifest.roots.push_back(terrain);
    package_manifest.dependency_closure.push_back(terrain);
    assets::cache_error cache_error;
    for (const auto& artifact : cooked.value().artifacts)
    {
        REQUIRE(cache.put_blob(artifact.hash, artifact.bytes, cache_error));
        package_manifest.artifacts.push_back({.asset = terrain,
                                              .type = assets::asset_types::terrain,
                                              .name = artifact.name,
                                              .schema = artifact.schema,
                                              .schema_version = artifact.schema_version,
                                              .hash = artifact.hash,
                                              .size = artifact.size,
                                              .chunk = "terrain"});
    }
    const auto package = assets::build_asset_packages(std::move(package_manifest), cache, root / "package");
    REQUIRE(package.succeeded());

    assets::package_artifact_reader reader;
    REQUIRE(reader.mount(package.manifest_path));
    REQUIRE(reader.bytes_read() == 0u);

    render::renderer renderer;
    scene::terrain_render_proxy proxy;
    proxy.regions.reserve(cooked.value().manifest.regions.size());
    for (const auto& manifest_region : cooked.value().manifest.regions)
    {
        const auto* artifact = scene::find_terrain_artifact(cooked.value().manifest, manifest_region.region,
                                                            scene::terrain_artifact_kind::render_geometry);
        REQUIRE(artifact != nullptr);
        render::virtual_mesh_data metadata;
        metadata.pages.reserve(artifact->pages.size());
        for (const auto& page : artifact->pages)
            metadata.pages.push_back({.uncompressed_size = page.decoded_size,
                                      .compressed_size = page.stored_size,
                                      .content_hash = page.content_hash,
                                      .root = page.root});
        const auto resource = renderer.create_virtual_mesh(std::move(metadata));
        REQUIRE(resource.valid());
        scene::terrain_render_region_proxy region;
        region.id = manifest_region.region;
        region.geometry.virtualized = resource;
        region.geometry.asset_generation = artifact->generation;
        proxy.regions.push_back(region);
    }

    jobs::job_system jobs({.worker_count = 1u, .io_worker_count = 1u, .enable_render_thread = false});
    io::async_file_service files(jobs);
    render::filesystem_virtual_geometry_artifact_source source(files);
    scene::terrain_virtual_geometry_streaming_binding binding;
    const auto bound = binding.synchronize(cooked.value().manifest, reader, proxy, renderer, source);
    REQUIRE(bound.succeeded);
    REQUIRE(bound.bound_regions == cooked.value().manifest.regions.size());
    REQUIRE(bound.skipped_regions == 0u);
    REQUIRE(reader.bytes_read() == 0u);

    const auto& first_region = cooked.value().manifest.regions.front();
    const auto* first_artifact = scene::find_terrain_artifact(cooked.value().manifest, first_region.region,
                                                              scene::terrain_artifact_kind::render_geometry);
    REQUIRE(first_artifact != nullptr);
    REQUIRE_FALSE(first_artifact->pages.empty());
    const auto resource = proxy.regions.front().geometry.virtualized;
    const render::virtual_geometry_page_load load{.resource = resource,
                                                  .resource_generation =
                                                      renderer.virtual_mesh_content_generation(resource),
                                                  .page_index = 0u,
                                                  .byte_size = first_artifact->pages.front().stored_size};
    auto future = source.read_page(load);
    const auto page = future.get();
    REQUIRE(page);
    REQUIRE(page.value().size() == first_artifact->pages.front().stored_size);
    REQUIRE(render::verify_virtual_geometry_artifact_page(page.value(),
                                                          {.offset = first_artifact->pages.front().offset,
                                                           .stored_size = first_artifact->pages.front().stored_size,
                                                           .decoded_size = first_artifact->pages.front().decoded_size,
                                                           .content_hash = first_artifact->pages.front().content_hash,
                                                           .root = first_artifact->pages.front().root}));

    binding.clear(source);
}
