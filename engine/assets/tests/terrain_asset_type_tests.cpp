#include <arc/assets/assets.h>
#include <arc/assets/terrain_types.h>

#include <catch2/catch_test_macros.hpp>

TEST_CASE("terrain files classify as the unified terrain asset type")
{
    const auto classified = arc::assets::classify_asset_path("Content/Terrain/Island.terrain");
    REQUIRE(classified.has_value());
    REQUIRE(classified->first == arc::assets::asset_types::terrain);
    REQUIRE(classified->second == arc::assets::importer_ids::terrain);

    const auto uppercase = arc::assets::classify_asset_path("Content/Terrain/Island.TERRAIN");
    REQUIRE(uppercase.has_value());
    REQUIRE(uppercase->first == arc::assets::asset_types::terrain);
    REQUIRE(uppercase->second == arc::assets::importer_ids::terrain);
}
