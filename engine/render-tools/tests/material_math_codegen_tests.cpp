#include <arc/render_tools/material_graph.h>

#include <catch2/catch_test_macros.hpp>

#include <string>
#include <string_view>

namespace
{
bool contains(std::string_view text, std::string_view expected)
{
    return text.find(expected) != std::string_view::npos;
}
} // namespace

TEST_CASE("material math nodes preserve float widths and broadcast scalars")
{
    constexpr std::string_view graph = R"({
      "version":1,
      "nodes":[
        {"id":"material-output","type":"output","values":{}},
        {"id":"scalar","type":"constant","values":{"value":0.25}},
        {"id":"v2","type":"vector2","values":{"value":[1.0,2.0]}},
        {"id":"v3","type":"vector3","values":{"value":[1.0,2.0,3.0]}},
        {"id":"v4","type":"vector4","values":{"value":[1.0,2.0,3.0,4.0]}},
        {"id":"add2","type":"add","values":{}},
        {"id":"add3","type":"add","values":{}},
        {"id":"add4","type":"add","values":{}},
        {"id":"len2","type":"length","values":{}},
        {"id":"len4","type":"length","values":{}}
      ],
      "connections":[
        {"id":"1","from":{"nodeId":"v2","pin":"value"},"to":{"nodeId":"add2","pin":"a"}},
        {"id":"2","from":{"nodeId":"scalar","pin":"value"},"to":{"nodeId":"add2","pin":"b"}},
        {"id":"3","from":{"nodeId":"add2","pin":"result"},"to":{"nodeId":"len2","pin":"value"}},
        {"id":"4","from":{"nodeId":"len2","pin":"result"},"to":{"nodeId":"material-output","pin":"metallic"}},
        {"id":"5","from":{"nodeId":"v3","pin":"value"},"to":{"nodeId":"add3","pin":"a"}},
        {"id":"6","from":{"nodeId":"scalar","pin":"value"},"to":{"nodeId":"add3","pin":"b"}},
        {"id":"7","from":{"nodeId":"add3","pin":"result"},"to":{"nodeId":"material-output","pin":"baseColor"}},
        {"id":"8","from":{"nodeId":"v4","pin":"value"},"to":{"nodeId":"add4","pin":"a"}},
        {"id":"9","from":{"nodeId":"scalar","pin":"value"},"to":{"nodeId":"add4","pin":"b"}},
        {"id":"10","from":{"nodeId":"add4","pin":"result"},"to":{"nodeId":"len4","pin":"value"}},
        {"id":"11","from":{"nodeId":"len4","pin":"result"},"to":{"nodeId":"material-output","pin":"roughness"}}
      ]
    })";

    const auto compilation = arc::render::tools::compile_material_graph_json(graph);
    REQUIRE(compilation);
    const auto generated = arc::render::tools::generate_material_slang(compilation.value());
    REQUIRE(generated);

    const auto& source = generated.value().source;
    REQUIRE(contains(source, "float2 arc_node_add2_result"));
    REQUIRE(contains(source, "float3 arc_node_add3_result"));
    REQUIRE(contains(source, "float4 arc_node_add4_result"));
    REQUIRE(contains(source, "float2(arc_node_scalar_value)"));
    REQUIRE(contains(source, "float3(arc_node_scalar_value)"));
    REQUIRE(contains(source, "float4(arc_node_scalar_value)"));
}

TEST_CASE("material math codegen covers the complete core math catalog")
{
    constexpr std::string_view graph = R"({
      "version":1,
      "nodes":[
        {"id":"material-output","type":"output","values":{}},
        {"id":"v","type":"vector3","values":{"value":[0.25,0.5,0.75]}},
        {"id":"s","type":"constant","values":{"value":0.5}},
        {"id":"abs","type":"abs","values":{}},
        {"id":"ceil","type":"ceil","values":{}},
        {"id":"floor","type":"floor","values":{}},
        {"id":"round","type":"round","values":{}},
        {"id":"truncate","type":"truncate","values":{}},
        {"id":"frac","type":"frac","values":{}},
        {"id":"fmod","type":"fmod","values":{}},
        {"id":"min","type":"min","values":{}},
        {"id":"max","type":"max","values":{}},
        {"id":"oneMinus","type":"oneMinus","values":{}},
        {"id":"power","type":"power","values":{}},
        {"id":"sqrt","type":"squareRoot","values":{}},
        {"id":"log","type":"logarithm","values":{}},
        {"id":"log2","type":"log2","values":{}},
        {"id":"log10","type":"log10","values":{}},
        {"id":"sin","type":"sine","values":{}},
        {"id":"cos","type":"cosine","values":{}},
        {"id":"asin","type":"arcsine","values":{}},
        {"id":"acos","type":"arccosine","values":{}},
        {"id":"atan","type":"arctangent","values":{}},
        {"id":"atan2","type":"arctangent2","values":{}},
        {"id":"smooth","type":"smoothStep","values":{}},
        {"id":"step","type":"step","values":{}},
        {"id":"sign","type":"sign","values":{}},
        {"id":"if","type":"if","values":{}},
        {"id":"distance","type":"distance","values":{}},
        {"id":"length","type":"length","values":{}},
        {"id":"sum","type":"add","values":{}}
      ],
      "connections":[
        {"id":"1","from":{"nodeId":"v","pin":"value"},"to":{"nodeId":"abs","pin":"value"}},
        {"id":"2","from":{"nodeId":"abs","pin":"result"},"to":{"nodeId":"ceil","pin":"value"}},
        {"id":"3","from":{"nodeId":"ceil","pin":"result"},"to":{"nodeId":"floor","pin":"value"}},
        {"id":"4","from":{"nodeId":"floor","pin":"result"},"to":{"nodeId":"round","pin":"value"}},
        {"id":"5","from":{"nodeId":"round","pin":"result"},"to":{"nodeId":"truncate","pin":"value"}},
        {"id":"6","from":{"nodeId":"truncate","pin":"result"},"to":{"nodeId":"frac","pin":"value"}},
        {"id":"7","from":{"nodeId":"frac","pin":"result"},"to":{"nodeId":"fmod","pin":"a"}},
        {"id":"8","from":{"nodeId":"s","pin":"value"},"to":{"nodeId":"fmod","pin":"b"}},
        {"id":"9","from":{"nodeId":"fmod","pin":"result"},"to":{"nodeId":"min","pin":"a"}},
        {"id":"10","from":{"nodeId":"s","pin":"value"},"to":{"nodeId":"min","pin":"b"}},
        {"id":"11","from":{"nodeId":"min","pin":"result"},"to":{"nodeId":"max","pin":"a"}},
        {"id":"12","from":{"nodeId":"s","pin":"value"},"to":{"nodeId":"max","pin":"b"}},
        {"id":"13","from":{"nodeId":"max","pin":"result"},"to":{"nodeId":"oneMinus","pin":"value"}},
        {"id":"14","from":{"nodeId":"oneMinus","pin":"result"},"to":{"nodeId":"power","pin":"base"}},
        {"id":"15","from":{"nodeId":"s","pin":"value"},"to":{"nodeId":"power","pin":"exponent"}},
        {"id":"16","from":{"nodeId":"power","pin":"result"},"to":{"nodeId":"sqrt","pin":"value"}},
        {"id":"17","from":{"nodeId":"sqrt","pin":"result"},"to":{"nodeId":"log","pin":"value"}},
        {"id":"18","from":{"nodeId":"log","pin":"result"},"to":{"nodeId":"log2","pin":"value"}},
        {"id":"19","from":{"nodeId":"log2","pin":"result"},"to":{"nodeId":"log10","pin":"value"}},
        {"id":"20","from":{"nodeId":"log10","pin":"result"},"to":{"nodeId":"sin","pin":"value"}},
        {"id":"21","from":{"nodeId":"sin","pin":"result"},"to":{"nodeId":"cos","pin":"value"}},
        {"id":"22","from":{"nodeId":"cos","pin":"result"},"to":{"nodeId":"asin","pin":"value"}},
        {"id":"23","from":{"nodeId":"asin","pin":"result"},"to":{"nodeId":"acos","pin":"value"}},
        {"id":"24","from":{"nodeId":"acos","pin":"result"},"to":{"nodeId":"atan","pin":"value"}},
        {"id":"25","from":{"nodeId":"atan","pin":"result"},"to":{"nodeId":"atan2","pin":"y"}},
        {"id":"26","from":{"nodeId":"s","pin":"value"},"to":{"nodeId":"atan2","pin":"x"}},
        {"id":"27","from":{"nodeId":"s","pin":"value"},"to":{"nodeId":"smooth","pin":"min"}},
        {"id":"28","from":{"nodeId":"atan2","pin":"result"},"to":{"nodeId":"smooth","pin":"max"}},
        {"id":"29","from":{"nodeId":"v","pin":"value"},"to":{"nodeId":"smooth","pin":"value"}},
        {"id":"30","from":{"nodeId":"s","pin":"value"},"to":{"nodeId":"step","pin":"edge"}},
        {"id":"31","from":{"nodeId":"smooth","pin":"result"},"to":{"nodeId":"step","pin":"value"}},
        {"id":"32","from":{"nodeId":"step","pin":"result"},"to":{"nodeId":"sign","pin":"value"}},
        {"id":"33","from":{"nodeId":"s","pin":"value"},"to":{"nodeId":"if","pin":"a"}},
        {"id":"34","from":{"nodeId":"s","pin":"value"},"to":{"nodeId":"if","pin":"b"}},
        {"id":"35","from":{"nodeId":"sign","pin":"result"},"to":{"nodeId":"if","pin":"greater"}},
        {"id":"36","from":{"nodeId":"smooth","pin":"result"},"to":{"nodeId":"if","pin":"equal"}},
        {"id":"37","from":{"nodeId":"v","pin":"value"},"to":{"nodeId":"if","pin":"less"}},
        {"id":"38","from":{"nodeId":"if","pin":"result"},"to":{"nodeId":"material-output","pin":"baseColor"}},
        {"id":"39","from":{"nodeId":"if","pin":"result"},"to":{"nodeId":"distance","pin":"a"}},
        {"id":"40","from":{"nodeId":"v","pin":"value"},"to":{"nodeId":"distance","pin":"b"}},
        {"id":"41","from":{"nodeId":"if","pin":"result"},"to":{"nodeId":"length","pin":"value"}},
        {"id":"42","from":{"nodeId":"distance","pin":"result"},"to":{"nodeId":"sum","pin":"a"}},
        {"id":"43","from":{"nodeId":"length","pin":"result"},"to":{"nodeId":"sum","pin":"b"}},
        {"id":"44","from":{"nodeId":"sum","pin":"result"},"to":{"nodeId":"material-output","pin":"roughness"}}
      ]
    })";

    const auto compilation = arc::render::tools::compile_material_graph_json(graph);
    REQUIRE(compilation);
    const auto generated = arc::render::tools::generate_material_slang(compilation.value());
    REQUIRE(generated);

    const auto& source = generated.value().source;
    for (const auto intrinsic :
         {"abs(",  "ceil(",  "floor(",      "round(", "trunc(", "frac(",     "fmod(",  "min(",  "max(",
          "pow(",  "sqrt(",  "log(",        "log2(",  "log10(", "sin(",      "cos(",   "asin(", "acos(",
          "atan(", "atan2(", "smoothstep(", "step(",  "sign(",  "distance(", "length("})
        REQUIRE(contains(source, intrinsic));
    REQUIRE(contains(source, "arc_node_oneMinus_result"));
    REQUIRE(contains(source, "arc_node_if_result"));
}

TEST_CASE("material math rejects incompatible vector widths")
{
    constexpr std::string_view graph = R"({
      "version":1,
      "nodes":[
        {"id":"material-output","type":"output","values":{}},
        {"id":"v2","type":"vector2","values":{"value":[1.0,2.0]}},
        {"id":"v3","type":"vector3","values":{"value":[1.0,2.0,3.0]}},
        {"id":"add","type":"add","values":{}},
        {"id":"length","type":"length","values":{}}
      ],
      "connections":[
        {"id":"1","from":{"nodeId":"v2","pin":"value"},"to":{"nodeId":"add","pin":"a"}},
        {"id":"2","from":{"nodeId":"v3","pin":"value"},"to":{"nodeId":"add","pin":"b"}},
        {"id":"3","from":{"nodeId":"add","pin":"result"},"to":{"nodeId":"length","pin":"value"}},
        {"id":"4","from":{"nodeId":"length","pin":"result"},"to":{"nodeId":"material-output","pin":"roughness"}}
      ]
    })";

    const auto compilation = arc::render::tools::compile_material_graph_json(graph);
    REQUIRE(compilation);
    const auto generated = arc::render::tools::generate_material_slang(compilation.value());
    REQUIRE_FALSE(generated);
    REQUIRE(contains(generated.error().message, "incompatible vector widths"));
}