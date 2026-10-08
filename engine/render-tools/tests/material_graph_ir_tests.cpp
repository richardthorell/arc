#include <arc/render_tools/material_graph.h>

#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <array>
#include <cstring>
#include <string_view>

namespace
{
const arc::render::tools::material_surface_output_binding*
find_output(const arc::render::tools::material_graph_descriptor& descriptor,
            arc::render::tools::material_surface_output output)
{
    for (const auto& binding : descriptor.outputs)
        if (binding.output == output) return &binding;
    return nullptr;
}

const arc::render::shader_parameter_descriptor*
find_parameter(const arc::render::tools::material_graph_descriptor& descriptor, std::string_view name)
{
    for (const auto& parameter : descriptor.parameters)
        if (parameter.name == name) return &parameter;
    return nullptr;
}
} // namespace

TEST_CASE("native material graph compiler emits deterministic backend-neutral IR and descriptor")
{
    constexpr std::string_view graph = R"({
      "version":1,
      "nodes":[
        {"id":"material-output","type":"output","values":{}},
        {"id":"surface-normal","type":"normalMap","values":{"strength":0.75}},
        {"id":"clock","type":"time","values":{}},
        {"id":"tint","type":"vector3","values":{"value":[0.2,0.4,0.8]},
         "parameter":{"exposed":true,"name":"Tint"}},
        {"id":"albedo-texture","type":"textureSample","values":{},
         "parameter":{"exposed":true,"name":"Albedo"}},
        {"id":"uv0","type":"texCoord","values":{}},
        {"id":"tinted-albedo","type":"multiply","values":{}}
      ],
      "connections":[
        {"id":"6","from":{"nodeId":"surface-normal","pin":"result"},
         "to":{"nodeId":"material-output","pin":"normal"}},
        {"id":"3","from":{"nodeId":"tinted-albedo","pin":"result"},
         "to":{"nodeId":"material-output","pin":"baseColor"}},
        {"id":"5","from":{"nodeId":"albedo-texture","pin":"rgb"},
         "to":{"nodeId":"surface-normal","pin":"texture"}},
        {"id":"1","from":{"nodeId":"uv0","pin":"uv"},
         "to":{"nodeId":"albedo-texture","pin":"uv"}},
        {"id":"4","from":{"nodeId":"clock","pin":"time"},
         "to":{"nodeId":"material-output","pin":"metallic"}},
        {"id":"2","from":{"nodeId":"albedo-texture","pin":"rgb"},
         "to":{"nodeId":"tinted-albedo","pin":"a"}},
        {"id":"7","from":{"nodeId":"tint","pin":"value"},
         "to":{"nodeId":"tinted-albedo","pin":"b"}}
      ]
    })";

    const auto first = arc::render::tools::compile_material_graph_json(graph);
    const auto second = arc::render::tools::compile_material_graph_json(graph);
    REQUIRE(first);
    REQUIRE(second);

    const auto& compilation = first.value();
    REQUIRE(compilation.ir.version == arc::render::tools::material_ir_version);
    REQUIRE(compilation.ir.output_node_id == "material-output");
    REQUIRE(compilation.ir.nodes.size() == 7);
    REQUIRE(compilation.ir.connections.size() == 7);
    REQUIRE(compilation.ir.nodes.front().id == "albedo-texture");
    REQUIRE(compilation.ir.nodes.back().id == "uv0");
    REQUIRE(compilation.ir.nodes == second.value().ir.nodes);
    REQUIRE(compilation.ir.connections == second.value().ir.connections);

    const auto& descriptor = compilation.descriptor;
    REQUIRE(descriptor.material_abi == arc::render::material_abi_version);
    REQUIRE(descriptor.requirements.uses_time);
    REQUIRE(descriptor.requirements.uses_uv0);
    REQUIRE(descriptor.requirements.uses_texture_sampling);
    REQUIRE(descriptor.requirements.uses_normal_mapping);
    REQUIRE(descriptor.textures.size() == 1);
    REQUIRE(descriptor.textures.front().node_id == "albedo-texture");
    REQUIRE(descriptor.textures.front().slot == 0);
    REQUIRE(descriptor.textures.front().parameter_name == "Albedo");

    REQUIRE(descriptor.parameters.size() == 2);
    const auto* tint = find_parameter(descriptor, "Tint");
    REQUIRE(tint != nullptr);
    REQUIRE(tint->id == arc::render::make_shader_parameter_id("tint"));
    REQUIRE(tint->type == arc::render::shader_parameter_type::float3);
    REQUIRE(tint->size == 12);
    REQUIRE(tint->default_value.size() == 12);
    std::array<float, 3> tint_default{};
    std::memcpy(tint_default.data(), tint->default_value.data(), tint->default_value.size());
    constexpr std::array<float, 3> expected_tint{0.2f, 0.4f, 0.8f};
    REQUIRE(tint_default == expected_tint);

    const auto* base_color = find_output(descriptor, arc::render::tools::material_surface_output::base_color);
    REQUIRE(base_color != nullptr);
    REQUIRE(base_color->connected);
    REQUIRE(base_color->source_node == "tinted-albedo");
    REQUIRE(base_color->source_pin == "result");

    const auto* roughness = find_output(descriptor, arc::render::tools::material_surface_output::roughness);
    REQUIRE(roughness != nullptr);
    REQUIRE_FALSE(roughness->connected);
}

TEST_CASE("native material graph compiler preserves valid Scalar authoring ranges")
{
    constexpr std::string_view graph = R"({
      "version":1,
      "nodes":[
        {"id":"out","type":"output","values":{}},
        {"id":"roughness","type":"constant","values":{"value":0.45,"min":0.0,"max":1.0},
         "parameter":{"exposed":true,"name":"Roughness"}}
      ],
      "connections":[
        {"id":"1","from":{"nodeId":"roughness","pin":"value"},"to":{"nodeId":"out","pin":"roughness"}}
      ]
    })";

    const auto result = arc::render::tools::compile_material_graph_json(graph);
    REQUIRE(result);
    const auto& nodes = result.value().ir.nodes;
    const auto found =
        std::find_if(nodes.begin(), nodes.end(), [](const auto& node) { return node.id == "roughness"; });
    REQUIRE(found != nodes.end());
    CHECK(found->has_range);
    CHECK(found->minimum == 0.0f);
    CHECK(found->maximum == 1.0f);
    CHECK(found->literal.values[0] == 0.45f);
    const auto* parameter = find_parameter(result.value().descriptor, "Roughness");
    REQUIRE(parameter != nullptr);
    CHECK(parameter->has_range);
    CHECK(parameter->minimum == 0.0f);
    CHECK(parameter->maximum == 1.0f);
}

TEST_CASE("native material graph compiler rejects invalid Scalar authoring ranges")
{
    SECTION("requires both bounds")
    {
        constexpr std::string_view graph = R"({
          "version":1,
          "nodes":[
            {"id":"out","type":"output","values":{}},
            {"id":"scalar","type":"constant","values":{"value":0.5,"min":0.0}}
          ],
          "connections":[
            {"id":"1","from":{"nodeId":"scalar","pin":"value"},"to":{"nodeId":"out","pin":"roughness"}}
          ]
        })";
        const auto result = arc::render::tools::compile_material_graph_json(graph);
        REQUIRE_FALSE(result);
        CHECK(result.error().message.find("requires both min and max") != std::string::npos);
    }

    SECTION("rejects inverted bounds")
    {
        constexpr std::string_view graph = R"({
          "version":1,
          "nodes":[
            {"id":"out","type":"output","values":{}},
            {"id":"scalar","type":"constant","values":{"value":0.5,"min":1.0,"max":0.0}}
          ],
          "connections":[
            {"id":"1","from":{"nodeId":"scalar","pin":"value"},"to":{"nodeId":"out","pin":"roughness"}}
          ]
        })";
        const auto result = arc::render::tools::compile_material_graph_json(graph);
        REQUIRE_FALSE(result);
        CHECK(result.error().message.find("range is invalid") != std::string::npos);
    }

    SECTION("rejects values outside the authored range")
    {
        constexpr std::string_view graph = R"({
          "version":1,
          "nodes":[
            {"id":"out","type":"output","values":{}},
            {"id":"scalar","type":"constant","values":{"value":1.5,"min":0.0,"max":1.0}}
          ],
          "connections":[
            {"id":"1","from":{"nodeId":"scalar","pin":"value"},"to":{"nodeId":"out","pin":"roughness"}}
          ]
        })";
        const auto result = arc::render::tools::compile_material_graph_json(graph);
        REQUIRE_FALSE(result);
        CHECK(result.error().message.find("outside its authored range") != std::string::npos);
    }
}

TEST_CASE("Material Output exposes normalized semantic ranges")
{
    constexpr std::string_view graph = R"({
      "version":1,
      "nodes":[{"id":"out","type":"output","values":{}}],
      "connections":[]
    })";

    const auto result = arc::render::tools::compile_material_graph_json(graph);
    REQUIRE(result);

    const auto* roughness = find_output(result.value().descriptor, arc::render::tools::material_surface_output::roughness);
    REQUIRE(roughness != nullptr);
    CHECK(roughness->has_expected_range);
    CHECK(roughness->minimum == 0.0f);
    CHECK(roughness->maximum == 1.0f);

    const auto* ior =
        find_output(result.value().descriptor, arc::render::tools::material_surface_output::index_of_refraction);
    REQUIRE(ior != nullptr);
    CHECK_FALSE(ior->has_expected_range);
}

TEST_CASE("Material Output warns for directly incompatible Scalar domains without changing shader math")
{
    SECTION("authored range extends outside the output semantic")
    {
        constexpr std::string_view graph = R"({
          "version":1,
          "nodes":[
            {"id":"out","type":"output","values":{}},
            {"id":"roughness","type":"constant","values":{"value":0.5,"min":0.0,"max":5.0}}
          ],
          "connections":[
            {"id":"1","from":{"nodeId":"roughness","pin":"value"},"to":{"nodeId":"out","pin":"roughness"}}
          ]
        })";

        const auto result = arc::render::tools::compile_material_graph_json(graph);
        REQUIRE(result);
        REQUIRE(result.value().diagnostics.size() == 1);
        CHECK(result.value().diagnostics.front().severity == arc::render::shader_diagnostic_severity::warning);
        CHECK(result.value().diagnostics.front().code == "material.output-range");
        CHECK(result.value().diagnostics.front().location.graph_node_id == "roughness");
    }

    SECTION("unrestricted literal is visibly outside the semantic")
    {
        constexpr std::string_view graph = R"({
          "version":1,
          "nodes":[
            {"id":"out","type":"output","values":{}},
            {"id":"metallic","type":"constant","values":{"value":2.0}}
          ],
          "connections":[
            {"id":"1","from":{"nodeId":"metallic","pin":"value"},"to":{"nodeId":"out","pin":"metallic"}}
          ]
        })";

        const auto result = arc::render::tools::compile_material_graph_json(graph);
        REQUIRE(result);
        REQUIRE(result.value().diagnostics.size() == 1);
        CHECK(result.value().diagnostics.front().code == "material.output-range");
    }

    SECTION("compatible authored range is clean")
    {
        constexpr std::string_view graph = R"({
          "version":1,
          "nodes":[
            {"id":"out","type":"output","values":{}},
            {"id":"roughness","type":"constant","values":{"value":0.5,"min":0.0,"max":1.0}}
          ],
          "connections":[
            {"id":"1","from":{"nodeId":"roughness","pin":"value"},"to":{"nodeId":"out","pin":"roughness"}}
          ]
        })";

        const auto result = arc::render::tools::compile_material_graph_json(graph);
        REQUIRE(result);
        CHECK(result.value().diagnostics.empty());
    }
}

TEST_CASE("material graph texture semantics propagate to descriptor bindings")
{
    constexpr std::string_view graph = R"({
      "version":1,
      "nodes":[
        {"id":"out","type":"output","values":{}},
        {"id":"normal-texture","type":"textureSample2D","values":{"semantic":"normal"},
         "parameter":{"exposed":true,"name":"Normal Texture"}},
        {"id":"normal-map","type":"normalMap","values":{"strength":1.0}}
      ],
      "connections":[
        {"id":"1","from":{"nodeId":"normal-texture","pin":"rgb"},
         "to":{"nodeId":"normal-map","pin":"texture"}},
        {"id":"2","from":{"nodeId":"normal-map","pin":"normal"},
         "to":{"nodeId":"out","pin":"normal"}}
      ]
    })";

    const auto result = arc::render::tools::compile_material_graph_json(graph);
    REQUIRE(result);
    REQUIRE(result.value().descriptor.textures.size() == 1);
    const auto& texture = result.value().descriptor.textures.front();
    REQUIRE(texture.parameter_name == "Normal Texture");
    REQUIRE(texture.semantic == arc::render::texture_semantic::normal);
    REQUIRE(result.value().descriptor.requirements.uses_normal_mapping);
}

TEST_CASE("material graph rejects unknown texture semantics")
{
    constexpr std::string_view graph = R"({
      "version":1,
      "nodes":[
        {"id":"out","type":"output","values":{}},
        {"id":"texture","type":"textureSample2D","values":{"semantic":"not-a-semantic"}}
      ],
      "connections":[
        {"id":"1","from":{"nodeId":"texture","pin":"rgb"},"to":{"nodeId":"out","pin":"baseColor"}}
      ]
    })";

    const auto result = arc::render::tools::compile_material_graph_json(graph);
    REQUIRE_FALSE(result);
    REQUIRE(result.error().message.find("unsupported material texture semantic") != std::string::npos);
}

TEST_CASE("material descriptor excludes unreachable nodes and assigns texture slots by stable node ID")
{
    constexpr std::string_view graph = R"({
      "version":1,
      "nodes":[
        {"id":"z-texture","type":"textureSample","values":{}},
        {"id":"unused-time","type":"time","values":{},
         "parameter":{"exposed":true,"name":"Unused"}},
        {"id":"material-output","type":"output","values":{}},
        {"id":"a-texture","type":"textureSample","values":{}},
        {"id":"unused-texture","type":"textureSample","values":{}}
      ],
      "connections":[
        {"id":"1","from":{"nodeId":"z-texture","pin":"rgb"},
         "to":{"nodeId":"material-output","pin":"baseColor"}},
        {"id":"2","from":{"nodeId":"a-texture","pin":"rgb"},
         "to":{"nodeId":"material-output","pin":"emissive"}}
      ]
    })";

    const auto result = arc::render::tools::compile_material_graph_json(graph);
    REQUIRE(result);
    const auto& descriptor = result.value().descriptor;
    REQUIRE(descriptor.textures.size() == 2);
    REQUIRE(descriptor.textures[0].node_id == "a-texture");
    REQUIRE(descriptor.textures[0].slot == 0);
    REQUIRE(descriptor.textures[1].node_id == "z-texture");
    REQUIRE(descriptor.textures[1].slot == 1);
    REQUIRE(descriptor.parameters.empty());
    REQUIRE_FALSE(descriptor.requirements.uses_time);
    REQUIRE(descriptor.requirements.uses_texture_sampling);
    REQUIRE(descriptor.requirements.uses_uv0);
}

TEST_CASE("native material graph compiler rejects structurally ambiguous graphs")
{
    SECTION("duplicate node ID")
    {
        constexpr std::string_view graph = R"({
          "version":1,
          "nodes":[{"id":"same","type":"constant","values":{}},
                   {"id":"same","type":"output","values":{}}],
          "connections":[]
        })";
        const auto result = arc::render::tools::compile_material_graph_json(graph);
        REQUIRE_FALSE(result);
        REQUIRE(result.error().code == arc::render::shader_compile_error_code::validation_failed);
    }

    SECTION("multiple output nodes")
    {
        constexpr std::string_view graph = R"({
          "version":1,
          "nodes":[{"id":"out-a","type":"output","values":{}},
                   {"id":"out-b","type":"output","values":{}}],
          "connections":[]
        })";
        const auto result = arc::render::tools::compile_material_graph_json(graph);
        REQUIRE_FALSE(result);
        REQUIRE(result.error().message == "material graph contains multiple output nodes");
    }

    SECTION("invalid connection")
    {
        constexpr std::string_view graph = R"({
          "version":1,
          "nodes":[{"id":"material-output","type":"output","values":{}}],
          "connections":[{"id":"1","from":{"nodeId":"missing","pin":"value"},
                          "to":{"nodeId":"material-output","pin":"baseColor"}}]
        })";
        const auto result = arc::render::tools::compile_material_graph_json(graph);
        REQUIRE_FALSE(result);
        REQUIRE(result.error().code == arc::render::shader_compile_error_code::validation_failed);
    }

    SECTION("cycle")
    {
        constexpr std::string_view graph = R"({
          "version":1,
          "nodes":[{"id":"a","type":"add","values":{}},
                   {"id":"b","type":"multiply","values":{}},
                   {"id":"material-output","type":"output","values":{}}],
          "connections":[
            {"id":"1","from":{"nodeId":"a","pin":"result"},"to":{"nodeId":"b","pin":"a"}},
            {"id":"2","from":{"nodeId":"b","pin":"result"},"to":{"nodeId":"a","pin":"a"}},
            {"id":"3","from":{"nodeId":"a","pin":"result"},
             "to":{"nodeId":"material-output","pin":"baseColor"}}
          ]
        })";
        const auto result = arc::render::tools::compile_material_graph_json(graph);
        REQUIRE_FALSE(result);
        REQUIRE(result.error().message == "material graph contains a cycle");
    }
}

TEST_CASE("material graph texture samples preserve typed dimensions")
{
    constexpr std::string_view graph = R"({
      "version":1,
      "nodes":[
        {"id":"out","type":"output","values":{}},
        {"id":"tex2d","type":"textureSample","values":{"dimension":"2d"},"parameter":{"exposed":true,"name":"Albedo"}},
        {"id":"cube","type":"textureSample","values":{"dimension":"cube"},"parameter":{"exposed":true,"name":"Environment"}},
        {"id":"volume","type":"textureSample","values":{"dimension":"3d"},"parameter":{"exposed":true,"name":"Volume"}},
        {"id":"direction","type":"vector3","values":{"value":[0,0,1]}},
        {"id":"uvw","type":"vector3","values":{"value":[0.5,0.5,0.5]}}
      ],
      "connections":[
        {"id":"1","from":{"nodeId":"tex2d","pin":"rgb"},"to":{"nodeId":"out","pin":"baseColor"}},
        {"id":"2","from":{"nodeId":"direction","pin":"value"},"to":{"nodeId":"cube","pin":"uv"}},
        {"id":"3","from":{"nodeId":"cube","pin":"rgb"},"to":{"nodeId":"out","pin":"emissive"}},
        {"id":"4","from":{"nodeId":"uvw","pin":"value"},"to":{"nodeId":"volume","pin":"uv"}},
        {"id":"5","from":{"nodeId":"volume","pin":"r"},"to":{"nodeId":"out","pin":"roughness"}}
      ]
    })";

    const auto result = arc::render::tools::compile_material_graph_json(graph);
    REQUIRE(result);
    const auto& descriptor = result.value().descriptor;
    REQUIRE(descriptor.textures.size() == 3);
    REQUIRE(descriptor.textures[0].type == arc::render::shader_parameter_type::texture_cube);
    REQUIRE(descriptor.textures[0].dimension_slot == 0);
    REQUIRE(descriptor.textures[1].type == arc::render::shader_parameter_type::texture_2d);
    REQUIRE(descriptor.textures[1].dimension_slot == 0);
    REQUIRE(descriptor.textures[2].type == arc::render::shader_parameter_type::texture_3d);
    REQUIRE(descriptor.textures[2].dimension_slot == 0);
    REQUIRE(descriptor.requirements.uses_uv0);

    const auto* albedo = find_parameter(descriptor, "Albedo");
    const auto* environment = find_parameter(descriptor, "Environment");
    const auto* volume = find_parameter(descriptor, "Volume");
    REQUIRE(albedo != nullptr);
    REQUIRE(environment != nullptr);
    REQUIRE(volume != nullptr);
    REQUIRE(albedo->type == arc::render::shader_parameter_type::texture_2d);
    REQUIRE(environment->type == arc::render::shader_parameter_type::texture_cube);
    REQUIRE(volume->type == arc::render::shader_parameter_type::texture_3d);
}

TEST_CASE("cube and 3D material texture samples require explicit vec3 coordinates")
{
    constexpr std::string_view graph = R"({
      "version":1,
      "nodes":[
        {"id":"out","type":"output","values":{}},
        {"id":"cube","type":"textureSample","values":{"dimension":"cube"}}
      ],
      "connections":[
        {"id":"1","from":{"nodeId":"cube","pin":"rgb"},"to":{"nodeId":"out","pin":"baseColor"}}
      ]
    })";
    const auto result = arc::render::tools::compile_material_graph_json(graph);
    REQUIRE_FALSE(result);
    REQUIRE(result.error().message.find("requires a vec3 coordinate input") != std::string::npos);
}

TEST_CASE("native material graph compiler accepts explicit typed texture sample nodes")

{

    constexpr std::string_view graph =
        R"({
      "version":1,
      "nodes":[
        {"id":"sample-2d","type":"textureSample2D","values":{}},
        {"id":"sample-cube","type":"textureSampleCube","values":{}},
        {"id":"sample-3d","type":"textureSample3D","values":{}},
        {"id":"direction","type":"vector3","values":{"value":[0,0,1]}},
        {"id":"coords","type":"vector3","values":{"value":[0.5,0.5,0.5]}},
        {"id":"material-output","type":"output","values":{}}
      ],
      "connections":[
        {"id":"1","from":{"nodeId":"sample-2d","pin":"rgb"},"to":{"nodeId":"material-output","pin":"baseColor"}},
        {"id":"2","from":{"nodeId":"direction","pin":"value"},"to":{"nodeId":"sample-cube","pin":"uv"}},
        {"id":"3","from":{"nodeId":"sample-cube","pin":"r"},"to":{"nodeId":"material-output","pin":"metallic"}},
        {"id":"4","from":{"nodeId":"coords","pin":"value"},"to":{"nodeId":"sample-3d","pin":"uv"}},
        {"id":"5","from":{"nodeId":"sample-3d","pin":"r"},"to":{"nodeId":"material-output","pin":"roughness"}}
      ]
    })";

    const auto result = arc::render::tools::compile_material_graph_json(graph);

    REQUIRE(result);

    REQUIRE(result.value().descriptor.textures.size() == 3);

    REQUIRE(result.value().descriptor.textures[0].type == arc::render::shader_parameter_type::texture_2d);

    REQUIRE(result.value().descriptor.textures[1].type == arc::render::shader_parameter_type::texture_3d);

    REQUIRE(result.value().descriptor.textures[2].type == arc::render::shader_parameter_type::texture_cube);
}
