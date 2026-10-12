#include <arc/scene/irradiance_probe_asset.h>

#include <catch2/catch_test_macros.hpp>

#include <span>
#include <vector>

TEST_CASE("arcprobe importer publishes typed baked irradiance data")
{
    arc::render::irradiance_probe_volume_descriptor volume;
    volume.local_bounds = {arc::geometric::point3f{0.0f, 0.0f, 0.0f}, arc::geometric::point3f{1.0f, 1.0f, 1.0f}};
    volume.dimensions = {1u, 1u, 1u};
    volume.spacing = {1.0f, 1.0f, 1.0f};
    volume.probes.resize(1u);
    volume.probes[0].spherical_harmonics[0] = {1.0f, 0.5f, 0.25f};

    const auto encoded = arc::render::encode_irradiance_probe_volume(volume);
    REQUIRE_FALSE(encoded.empty());

    auto importer = arc::scene::make_irradiance_probe_volume_importer();
    REQUIRE(importer);
    REQUIRE(importer->descriptor().id == arc::assets::importer_ids::irradiance_probe_volume);
    REQUIRE(importer->descriptor().output_types.size() == 1u);
    REQUIRE(importer->descriptor().output_types.front() == arc::assets::asset_types::irradiance_probe_volume);
    REQUIRE(importer->descriptor().extensions == std::vector<std::string>{".arcprobe"});

    arc::assets::asset_import_context context;
    context.reference.expected_type = arc::assets::asset_types::irradiance_probe_volume;
    context.metadata.type = arc::assets::asset_types::irradiance_probe_volume;
    context.metadata.importer = arc::assets::importer_ids::irradiance_probe_volume;
    context.source_path = "Content/Lighting/test.arcprobe";
    context.source_bytes = encoded;

    const auto imported = importer->import(context);
    REQUIRE(imported.succeeded());
    REQUIRE(imported.payload.type() == arc::assets::asset_types::irradiance_probe_volume);
    const auto* decoded = imported.payload.get<arc::render::irradiance_probe_volume_descriptor>();
    REQUIRE(decoded != nullptr);
    REQUIRE(decoded->dimensions == std::array<std::uint32_t, 3>{1u, 1u, 1u});
    REQUIRE(decoded->probes.size() == 1u);
    REQUIRE(decoded->probes[0].spherical_harmonics[0][0] == 1.0f);
}

TEST_CASE("arcprobe importer rejects malformed cooked data")
{
    auto importer = arc::scene::make_irradiance_probe_volume_importer();
    const std::vector<std::byte> malformed{std::byte{0x01}, std::byte{0x02}};

    arc::assets::asset_import_context context;
    context.reference.expected_type = arc::assets::asset_types::irradiance_probe_volume;
    context.metadata.type = arc::assets::asset_types::irradiance_probe_volume;
    context.metadata.importer = arc::assets::importer_ids::irradiance_probe_volume;
    context.source_path = "Content/Lighting/bad.arcprobe";
    context.source_bytes = malformed;

    const auto imported = importer->import(context);
    REQUIRE_FALSE(imported.succeeded());
    REQUIRE(imported.error.code == arc::assets::asset_error_code::import_failed);
}
