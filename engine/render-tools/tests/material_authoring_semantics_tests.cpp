#include <arc/render_tools/material_asset.h>

#include <catch2/catch_test_macros.hpp>

#include <string_view>

namespace
{

void require_invalid_semantic(std::string_view source, std::string_view field)
{
    const auto result = arc::render::tools::parse_material_authoring_json(source);
    REQUIRE_FALSE(result);
    REQUIRE(result.error().code == arc::render::tools::material_asset_error_code::invalid_document);
    REQUIRE(result.error().message.find(field) != std::string::npos);
}

} // namespace

TEST_CASE("material authoring rejects unsupported current-schema semantics")
{
    require_invalid_semantic(
        R"({"version":4,"domain":"volume","graph":{"version":1,"nodes":[],"connections":[]}})", "domain");
    require_invalid_semantic(
        R"({"version":4,"shadingModel":"toon","graph":{"version":1,"nodes":[],"connections":[]}})",
        "shadingModel");
    require_invalid_semantic(
        R"({"version":4,"blendMode":"additive","graph":{"version":1,"nodes":[],"connections":[]}})",
        "blendMode");
}

TEST_CASE("material authoring rejects compatibility aliases in the current schema")
{
    require_invalid_semantic(
        R"({"version":4,"shadingModel":"custom_lit","graph":{"version":1,"nodes":[],"connections":[]}})",
        "shadingModel");
}
