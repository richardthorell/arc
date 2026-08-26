#include <arc/render_tools/material_graph.h>

#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <string_view>

TEST_CASE("material color authoring nodes lower to vector Material IR")
{
    constexpr std::string_view graph = R"({
      "version":1,
      "nodes":[
        {"id":"material-output","type":"output","values":{}},
        {"id":"base-color","type":"colorRgb","values":{"value":[0.2,0.4,0.8]},
         "parameter":{"exposed":true,"name":"Base Color"}},
        {"id":"overlay-color","type":"colorRgba","values":{"value":[0.1,0.2,0.3,0.5]}}
      ],
      "connections":[
        {"id":"1","from":{"nodeId":"base-color","pin":"value"},
         "to":{"nodeId":"material-output","pin":"baseColor"}}
      ]
    })";

    const auto result = arc::render::tools::compile_material_graph_json(graph);
    REQUIRE(result);

    const auto& compilation = result.value();
    const auto rgb = std::ranges::find(compilation.ir.nodes, "base-color", &arc::render::tools::material_ir_node::id);
    REQUIRE(rgb != compilation.ir.nodes.end());
    REQUIRE(rgb->kind == arc::render::tools::material_ir_node_kind::vector3);
    REQUIRE(rgb->literal.components == 3);
    REQUIRE(rgb->literal.values[0] == 0.2f);
    REQUIRE(rgb->literal.values[1] == 0.4f);
    REQUIRE(rgb->literal.values[2] == 0.8f);

    const auto rgba =
        std::ranges::find(compilation.ir.nodes, "overlay-color", &arc::render::tools::material_ir_node::id);
    REQUIRE(rgba != compilation.ir.nodes.end());
    REQUIRE(rgba->kind == arc::render::tools::material_ir_node_kind::vector4);
    REQUIRE(rgba->literal.components == 4);
    REQUIRE(rgba->literal.values[3] == 0.5f);

    REQUIRE(compilation.descriptor.parameters.size() == 1);
    REQUIRE(compilation.descriptor.parameters.front().name == "Base Color");
    REQUIRE(compilation.descriptor.parameters.front().type == arc::render::shader_parameter_type::float3);
}
