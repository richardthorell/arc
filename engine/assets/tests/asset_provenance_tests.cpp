#include <arc/assets/provenance.h>

#include <catch2/catch_test_macros.hpp>

TEST_CASE("asset provenance keeps remote source identity atomic")
{
    using namespace arc::assets;

    asset_provenance provenance;
    CHECK(valid_asset_provenance(provenance));
    CHECK_FALSE(has_remote_source_identity(provenance));

    provenance.provider = "poly-haven";
    CHECK_FALSE(valid_asset_provenance(provenance));

    provenance.provider_asset_id = "wood-floor-01";
    CHECK(valid_asset_provenance(provenance));
    CHECK(has_remote_source_identity(provenance));
}

TEST_CASE("asset provenance records a reproducible import recipe independently of identity")
{
    using namespace arc::assets;

    const asset_provenance provenance{
        .provider = "poly-haven",
        .provider_asset_id = "wood-floor-01",
        .source_revision = "2026-09-26",
        .source_hash = "sha256:0123456789abcdef",
        .original_url = "https://example.invalid/assets/wood-floor-01",
        .license = "CC0-1.0",
        .variant = "2k/glb",
        .import_recipe = R"({"format":"glb","resolution":"2k"})",
    };

    CHECK(valid_asset_provenance(provenance));
    CHECK(provenance.provider == "poly-haven");
    CHECK(provenance.provider_asset_id == "wood-floor-01");
    CHECK(provenance.source_revision == "2026-09-26");
    CHECK(provenance.source_hash == "sha256:0123456789abcdef");
    CHECK(provenance.license == "CC0-1.0");
    CHECK(provenance.variant == "2k/glb");
    CHECK(provenance.import_recipe == R"({"format":"glb","resolution":"2k"})");
}

TEST_CASE("asset provenance rejects an empty import recipe")
{
    using namespace arc::assets;

    asset_provenance provenance;
    provenance.import_recipe.clear();
    CHECK_FALSE(valid_asset_provenance(provenance));
}
