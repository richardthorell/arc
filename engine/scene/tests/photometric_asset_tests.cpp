#include <arc/scene/photometric_asset.h>

#include <catch2/catch_test_macros.hpp>

#include <cstddef>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace
{

constexpr std::string_view test_ies = R"(IESNA:LM-63-2002
[TEST] ARC deterministic fixture
TILT=NONE
1 1000 2 3 2 1 1 0.1 0.1 0.1 1 0 10
0 45 90
0 180
10 20 40
5 10 20
)";

std::vector<std::byte> bytes_of(std::string_view text)
{
    const auto bytes = std::as_bytes(std::span(text.data(), text.size()));
    return {bytes.begin(), bytes.end()};
}

} // namespace

TEST_CASE("photometric artifacts round trip normalized LM-63 data")
{
    const auto parsed = arc::render::parse_ies_profile(test_ies);
    REQUIRE(parsed);

    const auto encoded = arc::scene::encode_photometric_profile_artifact(parsed.profile);
    REQUIRE_FALSE(encoded.empty());

    const auto decoded = arc::scene::decode_photometric_profile_artifact(encoded);
    REQUIRE(decoded);
    REQUIRE(decoded.profile.photometric_type == parsed.profile.photometric_type);
    REQUIRE(decoded.profile.vertical_angles_degrees == parsed.profile.vertical_angles_degrees);
    REQUIRE(decoded.profile.horizontal_angles_degrees == parsed.profile.horizontal_angles_degrees);
    REQUIRE(decoded.profile.normalized_candela == parsed.profile.normalized_candela);
    REQUIRE(decoded.profile.peak_candela == parsed.profile.peak_candela);
    REQUIRE(decoded.profile.declared_lumens == parsed.profile.declared_lumens);
}

TEST_CASE("photometric artifact decoder rejects damaged data")
{
    const auto parsed = arc::render::parse_ies_profile(test_ies);
    REQUIRE(parsed);
    auto encoded = arc::scene::encode_photometric_profile_artifact(parsed.profile);
    REQUIRE(encoded.size() > 8u);

    auto bad_magic = encoded;
    bad_magic[0] = std::byte{0};
    REQUIRE_FALSE(arc::scene::decode_photometric_profile_artifact(bad_magic));

    encoded.pop_back();
    REQUIRE_FALSE(arc::scene::decode_photometric_profile_artifact(encoded));
}

TEST_CASE("IES importer publishes a typed CPU payload and cooked artifact")
{
    auto importer = arc::scene::make_photometric_profile_importer();
    REQUIRE(importer);
    const auto& descriptor = importer->descriptor();
    REQUIRE(descriptor.id == arc::assets::importer_ids::photometric_profile);
    REQUIRE(descriptor.output_types.size() == 1u);
    REQUIRE(descriptor.output_types.front() == arc::assets::asset_types::photometric_profile);
    REQUIRE(descriptor.extensions == std::vector<std::string>{".ies"});

    const auto source = bytes_of(test_ies);
    arc::assets::asset_import_context context;
    context.reference.expected_type = arc::assets::asset_types::photometric_profile;
    context.metadata.type = arc::assets::asset_types::photometric_profile;
    context.metadata.importer = arc::assets::importer_ids::photometric_profile;
    context.source_path = "Content/Lighting/test.ies";
    context.source_bytes = source;

    const auto imported = importer->import(context);
    REQUIRE(imported.succeeded());
    const auto* profile = imported.payload.get<arc::render::photometric_profile>();
    REQUIRE(profile != nullptr);
    REQUIRE(profile->peak_candela == 80.0f);
    REQUIRE(imported.residency == arc::assets::asset_residency::cpu);
    REQUIRE(imported.artifacts.size() == 1u);
    REQUIRE(imported.artifacts.front().name == "photometric");
    REQUIRE(imported.artifacts.front().extension == ".arcies");

    const auto cooked = arc::scene::decode_photometric_profile_artifact(imported.artifacts.front().bytes);
    REQUIRE(cooked);
    REQUIRE(cooked.profile.normalized_candela == profile->normalized_candela);
}

TEST_CASE("IES importer reports malformed photometry as import failure")
{
    auto importer = arc::scene::make_photometric_profile_importer();
    const auto source = bytes_of("IESNA:LM-63-2002\nTILT=INCLUDE\n");

    arc::assets::asset_import_context context;
    context.reference.expected_type = arc::assets::asset_types::photometric_profile;
    context.metadata.type = arc::assets::asset_types::photometric_profile;
    context.metadata.importer = arc::assets::importer_ids::photometric_profile;
    context.source_path = "Content/Lighting/bad.ies";
    context.source_bytes = source;

    const auto imported = importer->import(context);
    REQUIRE_FALSE(imported.succeeded());
    REQUIRE(imported.error.code == arc::assets::asset_error_code::import_failed);
    REQUIRE(imported.error.message.find("TILT") != std::string::npos);
}
