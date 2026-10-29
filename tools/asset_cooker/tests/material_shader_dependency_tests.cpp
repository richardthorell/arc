#include "../material_shader_dependency.h"

#include <catch2/catch_test_macros.hpp>

#include <string_view>
#include <vector>

namespace
{
arc::assets::asset_snapshot shader(std::string_view path)
{
    arc::assets::asset_snapshot asset;
    asset.type = arc::assets::asset_types::shader;
    asset.source_path = path;
    return asset;
}
} // namespace

TEST_CASE("Material Shader dependency resolution uses the authored path")
{
    const std::vector dependencies{shader("assets/Shaders/other.slang"), shader("assets/Shaders/custom.slang")};
    const auto result = arc::tools::detail::find_material_shader_dependency(dependencies, "Shaders/custom.slang");
    REQUIRE_FALSE(result.ambiguous);
    REQUIRE(result.dependency == &dependencies[1]);

    const auto prefixed = arc::tools::detail::find_material_shader_dependency(dependencies, "assets/Shaders/custom.slang");
    REQUIRE_FALSE(prefixed.ambiguous);
    REQUIRE(prefixed.dependency == &dependencies[1]);
}

TEST_CASE("Material Shader missing dependency cannot fall back to an unrelated shader")
{
    auto unrelated = shader("assets/Shaders/other.slang");
    auto non_shader = shader("assets/Shaders/custom.slang");
    non_shader.type = arc::assets::asset_types::texture_2d;
    const std::vector dependencies{unrelated, non_shader};
    const auto result = arc::tools::detail::find_material_shader_dependency(dependencies, "Shaders/custom.slang");
    REQUIRE_FALSE(result.ambiguous);
    REQUIRE(result.dependency == nullptr);
}

TEST_CASE("Material Shader duplicate suffixes fail instead of depending on registration order")
{
    const std::vector dependencies{
        shader("project-a/assets/Shaders/custom.slang"),
        shader("project-b/assets/Shaders/custom.slang"),
    };
    const auto first = arc::tools::detail::find_material_shader_dependency(dependencies, "Shaders/custom.slang");
    REQUIRE(first.ambiguous);
    REQUIRE(first.dependency == nullptr);

    const std::vector reversed{dependencies[1], dependencies[0]};
    const auto second = arc::tools::detail::find_material_shader_dependency(reversed, "Shaders/custom.slang");
    REQUIRE(second.ambiguous);
    REQUIRE(second.dependency == nullptr);
}

TEST_CASE("Material Shader exact path wins over a suffix match")
{
    const std::vector dependencies{shader("other/assets/Shaders/custom.slang"), shader("Shaders/custom.slang")};
    const auto result = arc::tools::detail::find_material_shader_dependency(dependencies, "Shaders/custom.slang");
    REQUIRE_FALSE(result.ambiguous);
    REQUIRE(result.dependency == &dependencies[1]);
}

TEST_CASE("Material Shader duplicate exact paths are ambiguous")
{
    const std::vector dependencies{shader("Shaders/custom.slang"), shader("Shaders/custom.slang")};
    const auto result = arc::tools::detail::find_material_shader_dependency(dependencies, "Shaders/custom.slang");
    REQUIRE(result.ambiguous);
    REQUIRE(result.dependency == nullptr);
}
