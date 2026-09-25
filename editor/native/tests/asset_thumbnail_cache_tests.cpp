#include <arc/editor/arc_host.h>
#include <arc/editor/editor_state.h>
#include <arc/render/render.h>

#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <array>
#include <filesystem>
#include <fstream>
#include <memory>

TEST_CASE("asset thumbnail cache-only lookup never generates a missing preview")
{
    const auto root = std::filesystem::temp_directory_path() / "arc-editor-thumbnail-cache-test";
    std::filesystem::remove_all(root);
    std::filesystem::create_directories(root / "textures");
    const auto texture_path = root / "textures" / "preview.tga";

    std::array<unsigned char, 34> tga{};
    tga[2] = 2;
    tga[12] = 2;
    tga[14] = 2;
    tga[16] = 32;
    tga[17] = 0x20;
    const std::array<unsigned char, 16> pixels{0, 0, 255, 255, 0, 255, 0, 255, 255, 0, 0, 255, 255, 255, 255, 255};
    std::copy(pixels.begin(), pixels.end(), tga.begin() + 18);
    {
        std::ofstream output(texture_path, std::ios::binary);
        output.write(reinterpret_cast<const char*>(tga.data()), static_cast<std::streamsize>(tga.size()));
    }

    auto renderer = std::make_unique<arc::render::renderer>();
    arc::editor::arc_host_manager manager;
    auto host = manager.acquire(std::move(renderer));
    arc::editor::editor_asset_state assets;
    assets.root = root;
    REQUIRE(host->open_project({.name = "Thumbnail Cache Test", .root = root}, assets).succeeded);

    REQUIRE_FALSE(host->asset_thumbnail("textures/preview.tga", 0).has_value());

    const auto generated = host->asset_thumbnail("textures/preview.tga", 64);
    REQUIRE(generated.has_value());
    const auto cached = host->asset_thumbnail("textures/preview.tga", 0);
    REQUIRE(cached.has_value());
    CHECK(cached->data_url == generated->data_url);
    CHECK(cached->width == generated->width);
    CHECK(cached->height == generated->height);

    host.reset();
    std::error_code cleanup_error;
    std::filesystem::remove_all(root, cleanup_error);
}
