#include <arc/water/water.h>

#include <catch2/catch_test_macros.hpp>

#include <span>
#include <string>
#include <vector>

TEST_CASE("Water presets round trip through the versioned asset format")
{
    arc::water::water_preset preset;
    preset.name = "North Atlantic";
    preset.settings.simulation.wind_speed = 17.0f;
    preset.settings.simulation.wind_direction = {0.82f, 0.57f};
    preset.settings.simulation.seed = 1337;
    preset.settings.appearance.roughness = 0.06f;

    const auto encoded = arc::water::write_water_asset_json(preset, false);
    REQUIRE(encoded.has_value());
    const auto decoded = arc::water::read_water_asset_json(encoded.value());
    REQUIRE(decoded.has_value());
    CHECK(decoded.value().name == preset.name);
    CHECK(decoded.value().body_type == arc::water::water_body_type::ocean);
    CHECK(decoded.value().settings.simulation.wind_speed == 17.0f);
    CHECK(decoded.value().settings.simulation.wind_direction[0] == 0.82f);
    CHECK(decoded.value().settings.simulation.seed == 1337);
    CHECK(decoded.value().settings.appearance.roughness == 0.06f);
}

TEST_CASE("Water preset codec rejects unsupported versions")
{
    const auto format =
        arc::water::read_water_asset_json(R"({"format":"arc.water-preset","formatVersion":99,"preset":{}})");
    REQUIRE_FALSE(format.has_value());
    CHECK(format.error().code == arc::water::water_asset_io_error_code::unsupported_format_version);

    const auto schema = arc::water::read_water_asset_json(
        R"({"format":"arc.water-preset","formatVersion":1,"preset":{"schemaVersion":99}})");
    REQUIRE_FALSE(schema.has_value());
    CHECK(schema.error().code == arc::water::water_asset_io_error_code::unsupported_schema_version);
}

TEST_CASE("Water importer materializes typed preset assets")
{
    arc::water::water_preset preset;
    const auto encoded = arc::water::write_water_asset_json(preset, false);
    REQUIRE(encoded.has_value());

    auto importer = arc::water::make_water_asset_importer();
    REQUIRE(importer != nullptr);
    CHECK(importer->descriptor().id == arc::assets::importer_ids::water_preset);
    CHECK(importer->descriptor().extensions == std::vector<std::string>{".arcwater"});

    const auto bytes = std::as_bytes(std::span<const char>{encoded.value().data(), encoded.value().size()});
    arc::assets::asset_import_context context;
    context.reference.guid = arc::assets::generate_asset_guid();
    context.reference.expected_type = arc::assets::asset_types::water_preset;
    context.source_path = "Engine/Water/Presets/Open Ocean.arcwater";
    context.source_bytes = bytes;

    const auto imported = importer->import(context);
    REQUIRE(imported.succeeded());
    const auto* payload = imported.payload.get<arc::water::water_preset>();
    REQUIRE(payload != nullptr);
    CHECK(payload->name == "Open Ocean");
}
