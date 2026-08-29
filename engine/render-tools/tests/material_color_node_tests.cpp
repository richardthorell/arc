#include <arc/render_tools/material_graph.h>

#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <string>
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

TEST_CASE("material Color codegen resolves each output pin independently")
{
    constexpr std::string_view graph = R"({
      "version":1,
      "nodes":[
        {"id":"material-output","type":"output","values":{}},
        {"id":"color","type":"colorRgba","values":{"value":[0.1,0.2,0.3,0.5]}},
        {"id":"rgba-length","type":"length","values":{}}
      ],
      "connections":[
        {"id":"1","from":{"nodeId":"color","pin":"rgb"},
         "to":{"nodeId":"material-output","pin":"baseColor"}},
        {"id":"2","from":{"nodeId":"color","pin":"r"},
         "to":{"nodeId":"material-output","pin":"metallic"}},
        {"id":"3","from":{"nodeId":"color","pin":"g"},
         "to":{"nodeId":"material-output","pin":"roughness"}},
        {"id":"4","from":{"nodeId":"color","pin":"b"},
         "to":{"nodeId":"material-output","pin":"ao"}},
        {"id":"5","from":{"nodeId":"color","pin":"a"},
         "to":{"nodeId":"material-output","pin":"opacity"}},
        {"id":"6","from":{"nodeId":"color","pin":"rgba"},
         "to":{"nodeId":"rgba-length","pin":"value"}},
        {"id":"7","from":{"nodeId":"rgba-length","pin":"result"},
         "to":{"nodeId":"material-output","pin":"clearCoat"}}
      ]
    })";

    const auto compilation = arc::render::tools::compile_material_graph_json(graph);
    REQUIRE(compilation);
    const auto generated = arc::render::tools::generate_material_slang(compilation.value());
    REQUIRE(generated);

    const auto& source = generated.value().source;
    const auto require_assignment = [&source](std::string_view type, std::string_view pin, std::string_view suffix)
    {
        const auto prefix = std::string(type) + " arc_node_color_" + std::string(pin) + " = ";
        const auto assignment = source.find(prefix);
        REQUIRE(assignment != std::string::npos);
        REQUIRE(source.find(std::string(suffix) + ';', assignment) != std::string::npos);
    };

    require_assignment("float3", "rgb", ".rgb");
    require_assignment("float", "r", ".r");
    require_assignment("float", "g", ".g");
    require_assignment("float", "b", ".b");
    require_assignment("float", "a", ".a");

    const auto rgba_assignment = source.find("float4 arc_node_color_rgba = ");
    REQUIRE(rgba_assignment != std::string::npos);
    REQUIRE(source.find("float arc_node_rgba_length_result = length(arc_node_color_rgba);", rgba_assignment) !=
            std::string::npos);
}

TEST_CASE("exposed material Color RGB output swizzles the parameter value")
{
    constexpr std::string_view graph = R"({
      "version":1,
      "nodes":[
        {"id":"material-output","type":"output","values":{}},
        {"id":"color","type":"colorRgba","values":{"value":[0.1,0.2,0.3,0.5]},
         "parameter":{"exposed":true,"name":"Tint"}}
      ],
      "connections":[
        {"id":"1","from":{"nodeId":"color","pin":"rgb"},
         "to":{"nodeId":"material-output","pin":"baseColor"}}
      ]
    })";

    const auto compilation = arc::render::tools::compile_material_graph_json(graph);
    REQUIRE(compilation);
    const auto generated = arc::render::tools::generate_material_slang(compilation.value());
    REQUIRE(generated);

    const auto& source = generated.value().source;
    const auto assignment = source.find("float3 arc_node_color_rgb = arcMaterialParameters.arc_param_");
    REQUIRE(assignment != std::string::npos);
    REQUIRE(source.find(".rgb;", assignment) != std::string::npos);
}
