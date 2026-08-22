#include <arc/render_tools/material_graph.h>

#include <catch2/catch_test_macros.hpp>

#include <array>
#include <cstring>
#include <string_view>

namespace
{
const arc::render::tools::material_surface_output_binding* find_output(
    const arc::render::tools::material_graph_descriptor& descriptor,
    arc::render::tools::material_surface_output output)
{
    for (const auto& binding : descriptor.outputs)
        if (binding.output == output) return &binding;
    return nullptr;
}

const arc::render::shader_parameter_descriptor* find_parameter(
    const arc::render::tools::material_graph_descriptor& descriptor,
    std::string_view name)
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
