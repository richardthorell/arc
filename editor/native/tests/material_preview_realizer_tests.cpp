#include <arc/editor/material_preview.h>
#include <arc/editor/material_preview_realizer.h>

#include <catch2/catch_test_macros.hpp>

#include <cmath>
#include <cstddef>
#include <filesystem>
#include <fstream>
#include <string>

namespace
{

constexpr float epsilon = 1.0e-5f;

bool near(float lhs, float rhs)
{
    return std::abs(lhs - rhs) <= epsilon;
}

const std::string red_material_source = R"({
  "version": 4,
  "name": "Preview Red",
  "domain": "surface",
  "blendMode": "opaque",
  "shadingModel": "standard",
  "doubleSided": false,
  "graph": {
    "version": 1,
    "nodes": [
      {"id":"output","type":"output","position":[0,0],"values":{}},
      {"id":"base","type":"vector3","position":[0,0],"values":{"value":[1.0,0.0,0.0]},"parameter":{"exposed":true,"name":"Base Color"}}
    ],
    "connections": [
      {"id":"base-output","from":{"nodeId":"base","pin":"value"},"to":{"nodeId":"output","pin":"baseColor"}}
    ]
  }
})";

} // namespace

TEST_CASE("material preview realizes authored base color through native Material IR")
{
    const auto result = arc::editor::realize_material_preview_descriptor(red_material_source, "Preview Red");
    REQUIRE(result.succeeded);
    CHECK(near(result.material.base_color[0], 1.0f));
    CHECK(near(result.material.base_color[1], 0.0f));
    CHECK(near(result.material.base_color[2], 0.0f));
    CHECK(near(result.material.roughness, 0.6f));
    if (result.material.runtime_program)
    {
        REQUIRE_FALSE(result.material.runtime_program->passes.empty());
        CHECK(result.material.runtime_program->passes.front().pass == arc::render::material_pass::gbuffer);
        CHECK_FALSE(result.material.runtime_program->passes.front().compiled.bytecode.empty());
        CHECK_FALSE(result.material.runtime_program->parameters.empty());
        CHECK_FALSE(result.material.runtime_program->parameter_defaults.empty());
    }
}

TEST_CASE("material thumbnail renderer uses graph-realized surface values")
{
    const auto source_path = std::filesystem::temp_directory_path() / "arc-preview-red-thumbnail.arcmat";
    {
        std::ofstream output(source_path, std::ios::binary | std::ios::trunc);
        REQUIRE(output.good());
        output << red_material_source;
    }

    arc::editor::material_asset asset = arc::editor::make_default_material_asset("Preview Red");
    asset.path = source_path;
    asset.graph_reserved = true;
    const auto preview = arc::editor::render_material_preview(asset, source_path.parent_path(), 64u);
    std::filesystem::remove(source_path);

    REQUIRE(preview.succeeded());
    REQUIRE(preview.texture.has_pixels());
    const std::size_t center = (32u * 64u + 32u) * 4u;
    const auto red = std::to_integer<unsigned char>(preview.texture.pixels[center]);
    const auto green = std::to_integer<unsigned char>(preview.texture.pixels[center + 1u]);
    const auto blue = std::to_integer<unsigned char>(preview.texture.pixels[center + 2u]);
    CHECK(red > green + 20u);
    CHECK(red > blue + 20u);
}

TEST_CASE("material preview evaluates static Material IR math instead of projecting authored fields")
{
    const std::string source = R"({
  "version": 4,
  "name": "Preview Math",
  "domain": "surface",
  "blendMode": "opaque",
  "shadingModel": "standard",
  "doubleSided": false,
  "graph": {
    "version": 1,
    "nodes": [
      {"id":"output","type":"output","position":[0,0],"values":{}},
      {"id":"color","type":"vector3","position":[0,0],"values":{"value":[0.2,0.4,0.8]}},
      {"id":"scale","type":"constant","position":[0,0],"values":{"value":0.5}},
      {"id":"multiply","type":"multiply","position":[0,0],"values":{}}
    ],
    "connections": [
      {"id":"color-a","from":{"nodeId":"color","pin":"value"},"to":{"nodeId":"multiply","pin":"a"}},
      {"id":"scale-b","from":{"nodeId":"scale","pin":"value"},"to":{"nodeId":"multiply","pin":"b"}},
      {"id":"result-output","from":{"nodeId":"multiply","pin":"result"},"to":{"nodeId":"output","pin":"baseColor"}}
    ]
  }
})";

    const auto result = arc::editor::realize_material_preview_descriptor(source, "Preview Math");
    REQUIRE(result.succeeded);
    CHECK(near(result.material.base_color[0], 0.1f));
    CHECK(near(result.material.base_color[1], 0.2f));
    CHECK(near(result.material.base_color[2], 0.4f));
}

TEST_CASE("material preview keeps ABI defaults for dynamic graph outputs")
{
    const std::string source = R"({
  "version": 4,
  "name": "Preview Dynamic",
  "domain": "surface",
  "blendMode": "opaque",
  "shadingModel": "standard",
  "doubleSided": false,
  "graph": {
    "version": 1,
    "nodes": [
      {"id":"output","type":"output","position":[0,0],"values":{}},
      {"id":"time","type":"time","position":[0,0],"values":{}}
    ],
    "connections": [
      {"id":"time-roughness","from":{"nodeId":"time","pin":"seconds"},"to":{"nodeId":"output","pin":"roughness"}}
    ]
  }
})";

    const auto result = arc::editor::realize_material_preview_descriptor(source, "Preview Dynamic");
    REQUIRE(result.succeeded);
    CHECK(near(result.material.roughness, 0.6f));
    CHECK_FALSE(result.diagnostics.empty());
}
