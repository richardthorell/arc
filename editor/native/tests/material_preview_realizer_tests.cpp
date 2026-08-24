#include <arc/editor/material_preview_realizer.h>

#include <catch2/catch_test_macros.hpp>

#include <cmath>
#include <string>

namespace
{

constexpr float epsilon = 1.0e-5f;

bool near(float lhs, float rhs)
{
    return std::abs(lhs - rhs) <= epsilon;
}

} // namespace

TEST_CASE("material preview realizes authored base color through native Material IR")
{
    const std::string source = R"({
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

    const auto result = arc::editor::realize_material_preview_descriptor(source, "Preview Red");
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
