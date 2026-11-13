#include "../material_shader_dependency.h"
#include "../material_texture_manifest.h"

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

    const auto prefixed =
        arc::tools::detail::find_material_shader_dependency(dependencies, "assets/Shaders/custom.slang");
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

TEST_CASE("Material texture bindings map compiler-assigned slots to exact registered asset GUIDs")
{
    arc::render::tools::material_graph_descriptor graph;
    arc::render::tools::material_texture_binding albedo_binding;
    albedo_binding.node_id = "albedo";
    albedo_binding.slot = 0;
    albedo_binding.type = arc::render::shader_parameter_type::texture_2d;
    albedo_binding.dimension_slot = 0;
    albedo_binding.default_source = "Textures/albedo.png";
    graph.textures.push_back(albedo_binding);

    arc::render::tools::material_texture_binding normal_binding;
    normal_binding.node_id = "normal";
    normal_binding.slot = 1;
    normal_binding.type = arc::render::shader_parameter_type::texture_2d;
    normal_binding.dimension_slot = 1;
    normal_binding.default_source = "Textures/normal.png";
    graph.textures.push_back(normal_binding);

    arc::render::tools::material_texture_binding unbound;
    unbound.node_id = "unbound";
    unbound.slot = 2;
    unbound.type = arc::render::shader_parameter_type::texture_2d;
    unbound.dimension_slot = 2;
    graph.textures.push_back(unbound);
    auto albedo = shader("project/assets/Textures/albedo.png");
    albedo.type = arc::assets::asset_types::texture_2d;
    albedo.guid = {1, 2};
    auto normal = shader("project/assets/Textures/normal.png");
    normal.type = arc::assets::asset_types::texture_2d;
    normal.guid = {3, 4};
    const std::vector dependencies{normal, albedo};
    const auto manifest = arc::tools::detail::make_material_texture_manifest({9, 10}, graph, dependencies);
    REQUIRE(manifest);
    REQUIRE(manifest.value().textures.size() == 3);
    CHECK(manifest.value().textures[0].asset == albedo.guid);
    CHECK(manifest.value().textures[1].asset == normal.guid);
    CHECK_FALSE(manifest.value().textures[2].asset.valid());

    const std::vector ambiguous{albedo, albedo, normal};
    CHECK_FALSE(arc::tools::detail::make_material_texture_manifest({9, 10}, graph, ambiguous));
    const std::vector missing{normal};
    CHECK_FALSE(arc::tools::detail::make_material_texture_manifest({9, 10}, graph, missing));
}
