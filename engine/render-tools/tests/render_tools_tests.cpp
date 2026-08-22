#include <arc/render_tools/render_tools.h>

#include <catch2/catch_test_macros.hpp>

TEST_CASE("native material graph lowering is deterministic and preserves stable parameters")
{
    constexpr std::string_view graph = R"({
      "version":1,
      "nodes":[
        {"id":"base-color-stable","type":"vector3","values":{"value":[0.2,0.4,0.8]},
         "parameter":{"exposed":true,"name":"Base Color"}},
        {"id":"material-output","type":"output","values":{}}
      ],
      "connections":[
        {"id":"edge-1","from":{"nodeId":"base-color-stable","pin":"value"},
         "to":{"nodeId":"material-output","pin":"baseColor"}}
      ]
    })";

    const auto first = arc::render::tools::lower_material_graph_json(graph);
    const auto second = arc::render::tools::lower_material_graph_json(graph);
    REQUIRE(first);
    REQUIRE(second);
    REQUIRE(first.value().source == second.value().source);
    REQUIRE(first.value().parameters.size() == 1);
    REQUIRE(first.value().parameters.front().id == arc::render::make_shader_parameter_id("base-color-stable"));
    REQUIRE(first.value().generated_line_nodes.size() == 1);
    const auto field = "arc_param_" +
                       std::to_string(arc::render::make_shader_parameter_id("base-color-stable").representation());
    REQUIRE(first.value().source.find("arcMaterialParameters." + field) != std::string::npos);
    const auto mapped_line = first.value().generated_line_nodes.begin()->first;
    std::size_t cursor{};
    for (std::uint32_t line = 1; line < mapped_line; ++line)
        cursor = first.value().source.find('\n', cursor) + 1;
    REQUIRE(first.value().source.substr(cursor).starts_with("    auto arc_node_base_color_stable_value"));
}

TEST_CASE("native material graph lowering rejects cycles")
{
    constexpr std::string_view graph = R"({
      "version":1,
      "nodes":[
        {"id":"a","type":"add","values":{}},
        {"id":"b","type":"multiply","values":{}},
        {"id":"material-output","type":"output","values":{}}
      ],
      "connections":[
        {"id":"1","from":{"nodeId":"a","pin":"result"},"to":{"nodeId":"b","pin":"a"}},
        {"id":"2","from":{"nodeId":"b","pin":"result"},"to":{"nodeId":"a","pin":"a"}},
        {"id":"3","from":{"nodeId":"a","pin":"result"},"to":{"nodeId":"material-output","pin":"baseColor"}}
      ]
    })";
    const auto result = arc::render::tools::lower_material_graph_json(graph);
    REQUIRE_FALSE(result);
    REQUIRE(result.error().code == arc::render::shader_compile_error_code::validation_failed);
}
