#include <arc/assets/cook.h>
#include <arc/assets/package_artifact_reader.h>
#include <arc/assets/terrain_types.h>

#include <catch2/catch_test_macros.hpp>

#include <filesystem>
#include <span>
#include <string_view>
#include <vector>

TEST_CASE("named package artifacts mount metadata-first and support range reads")
{
    using namespace arc::assets;
    const auto root = std::filesystem::temp_directory_path() / ("arc-package-reader-" + to_string(generate_asset_guid()));
    struct cleanup
    {
        std::filesystem::path root;
        ~cleanup()
        {
            std::error_code error;
            std::filesystem::remove_all(root, error);
        }
    } cleanup_guard{root};

    derived_data_cache cache({.root = root / "cache"});
    const auto asset = generate_asset_guid();
    const auto make_bytes = [](std::string_view text)
    {
        const auto bytes = std::as_bytes(std::span(text.data(), text.size()));
        return std::vector<std::byte>(bytes.begin(), bytes.end());
    };
    const auto first_bytes = make_bytes("first-payload");
    const auto second_bytes = make_bytes("second-payload");
    const auto first_hash = hash_bytes(first_bytes);
    const auto second_hash = hash_bytes(second_bytes);
    cache_error error;
    REQUIRE(cache.put_blob(first_hash, first_bytes, error));
    REQUIRE(cache.put_blob(second_hash, second_bytes, error));

    cook_manifest manifest;
    manifest.target = windows_vulkan_cook_target();
    manifest.build_id = "named-artifacts";
    manifest.roots.push_back(asset);
    manifest.dependency_closure.push_back(asset);
    manifest.artifacts.push_back({.asset = asset,
                                  .type = asset_types::terrain,
                                  .name = "terrain/regions/0/0/virtual-geometry",
                                  .schema = artifact_schemas::virtual_geometry,
                                  .schema_version = 3u,
                                  .hash = first_hash,
                                  .size = first_bytes.size(),
                                  .chunk = "terrain-detail"});
    manifest.artifacts.push_back({.asset = asset,
                                  .type = asset_types::terrain,
                                  .name = "terrain/regions/1/0/virtual-geometry",
                                  .schema = artifact_schemas::virtual_geometry,
                                  .schema_version = 3u,
                                  .hash = second_hash,
                                  .size = second_bytes.size(),
                                  .chunk = "terrain-detail"});

    const auto package = build_asset_packages(manifest, cache, root / "package");
    REQUIRE(package.succeeded());

    package_artifact_reader reader;
    REQUIRE(reader.mount(package.manifest_path));
    REQUIRE(reader.bytes_read() == 0u);

    const cooked_artifact_address second{asset, artifact_schemas::virtual_geometry,
                                         "terrain/regions/1/0/virtual-geometry"};
    REQUIRE(reader.find(second) != nullptr);
    const auto range = reader.read_range(second, 1u, 4u);
    REQUIRE(range);
    REQUIRE(range.value() == std::vector<std::byte>(second_bytes.begin() + 1, second_bytes.begin() + 5));
    REQUIRE(reader.bytes_read() == 4u);

    const auto full = reader.read(second);
    REQUIRE(full);
    REQUIRE(full.value() == second_bytes);
    REQUIRE(reader.bytes_read() == 4u + second_bytes.size());
}
