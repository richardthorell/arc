#include <arc/render_tools/render_tools.h>

#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <string>
#include <string_view>

TEST_CASE("Material IR codegen deterministically implements the full material ABI")
{
    constexpr std::string_view graph = R"({
      "version":1,
      "nodes":[
        {"id":"material-output","type":"output","values":{}},
        {"id":"z-texture","type":"textureSample","values":{}},
        {"id":"a-texture","type":"textureSample","values":{}},
        {"id":"clock","type":"time","values":{}},
        {"id":"tint","type":"vector3","values":{"value":[0.2,0.4,0.8]},
         "parameter":{"exposed":true,"name":"Tint"}},
        {"id":"tinted","type":"multiply","values":{}},
        {"id":"clear-coat","type":"constant","values":{"value":0.35}},
        {"id":"transmission","type":"constant","values":{"value":0.2}}
      ],
      "connections":[
        {"id":"1","from":{"nodeId":"z-texture","pin":"rgb"},
         "to":{"nodeId":"tinted","pin":"a"}},
        {"id":"2","from":{"nodeId":"tint","pin":"value"},
         "to":{"nodeId":"tinted","pin":"b"}},
        {"id":"3","from":{"nodeId":"tinted","pin":"result"},
         "to":{"nodeId":"material-output","pin":"baseColor"}},
        {"id":"4","from":{"nodeId":"a-texture","pin":"rgb"},
         "to":{"nodeId":"material-output","pin":"emissive"}},
        {"id":"5","from":{"nodeId":"clock","pin":"time"},
         "to":{"nodeId":"material-output","pin":"metallic"}},
        {"id":"6","from":{"nodeId":"clear-coat","pin":"value"},
         "to":{"nodeId":"material-output","pin":"clearCoat"}},
        {"id":"7","from":{"nodeId":"transmission","pin":"value"},
         "to":{"nodeId":"material-output","pin":"transmission"}}
      ]
    })";

    const auto compilation = arc::render::tools::compile_material_graph_json(graph);
    REQUIRE(compilation);
    REQUIRE(compilation.value().descriptor.outputs.size() == 24);
    const auto first = arc::render::tools::generate_material_slang(compilation.value());
    const auto second = arc::render::tools::generate_material_slang(compilation.value());
    REQUIRE(first);
    REQUIRE(second);
    REQUIRE(first.value().source == second.value().source);
    REQUIRE(first.value().generated_line_nodes == second.value().generated_line_nodes);

    const auto& source = first.value().source;
    REQUIRE(source.find("ARC_MATERIAL_ABI_VERSION = 1") != std::string::npos);
    REQUIRE(source.find("ArcSurfaceData arc_evaluate_material(ArcSurfaceInput input)") != std::string::npos);
    REQUIRE(source.find("surface.clearCoatNormalWS = surface.normalWS") != std::string::npos);
    REQUIRE(source.find("surface.clearCoatRoughness = 0.1") != std::string::npos);
    REQUIRE(source.find("surface.ambientOcclusion") != std::string::npos);
    REQUIRE(source.find("surface.emissiveRadiance") != std::string::npos);
    REQUIRE(source.find("surface.clearCoat = arc_node_clear_coat_value") != std::string::npos);
    REQUIRE(source.find("surface.transmission = arc_node_transmission_value") != std::string::npos);
    REQUIRE(source.find("arcFrame.timeSeconds") != std::string::npos);
    REQUIRE(source.find("float3 arc_node_a_texture_rgb = arcMaterialTextures[0].Sample") != std::string::npos);
    REQUIRE(source.find("float3 arc_node_z_texture_rgb = arcMaterialTextures[1].Sample") != std::string::npos);
    REQUIRE(source.find("float3 arc_node_tinted_result =") != std::string::npos);
    REQUIRE(source.find("float arc_node_clock_time = arcFrame.timeSeconds") != std::string::npos);
    REQUIRE(source.find("input.uv0") != std::string::npos);

    REQUIRE(first.value().parameters.size() == compilation.value().descriptor.parameters.size());
    REQUIRE(first.value().parameters.front().id == compilation.value().descriptor.parameters.front().id);
    REQUIRE(first.value().generated_line_nodes.size() == 7);
    REQUIRE(std::ranges::any_of(first.value().generated_line_nodes,
                                [](const auto& entry) { return entry.second == "a-texture"; }));
    REQUIRE(std::ranges::any_of(first.value().generated_line_nodes,
                                [](const auto& entry) { return entry.second == "z-texture"; }));
}

TEST_CASE("Material IR generated source compiles with the pinned Slang toolchain")
{
    arc::render::tools::slang_shader_compiler compiler;
    if (!compiler.available())
    {
        SUCCEED("Pinned slangc is optional for this unit test environment");
        return;
    }

    constexpr std::string_view graph = R"({
      "version":1,
      "nodes":[
        {"id":"base-color","type":"vector3","values":{"value":[0.25,0.5,0.75]},
         "parameter":{"exposed":true,"name":"Base Color"}},
        {"id":"roughness","type":"constant","values":{"value":0.35}},
        {"id":"sheen","type":"constant","values":{"value":0.15}},
        {"id":"material-output","type":"output","values":{}}
      ],
      "connections":[
        {"id":"1","from":{"nodeId":"base-color","pin":"value"},
         "to":{"nodeId":"material-output","pin":"baseColor"}},
        {"id":"2","from":{"nodeId":"roughness","pin":"value"},
         "to":{"nodeId":"material-output","pin":"roughness"}},
        {"id":"3","from":{"nodeId":"sheen","pin":"value"},
         "to":{"nodeId":"material-output","pin":"sheen"}}
      ]
    })";

    const auto compilation = arc::render::tools::compile_material_graph_json(graph);
    REQUIRE(compilation);
    const auto generated = arc::render::tools::generate_material_slang(compilation.value());
    REQUIRE(generated);

    arc::render::shader_compile_request request{.source_path = "material_codegen_test.generated.slang",
                                                .source_override = generated.value().source,
                                                .entry_point = "main",
                                                .profile = "spirv_1_5",
                                                .domain = arc::render::shader_domain::surface,
                                                .stage = arc::render::shader_stage::fragment,
                                                .target = arc::render::shader_target::spirv,
                                                .optimization = arc::render::shader_optimization::development,
                                                .generated_line_nodes = generated.value().generated_line_nodes};
    const auto result = compiler.compile(request);
    if (!result)
    {
        std::string failure = result.error().message;
        for (const auto& diagnostic : result.error().diagnostics)
        {
            failure += "\n" + diagnostic.location.path + ':' + std::to_string(diagnostic.location.line) + ':' +
                       std::to_string(diagnostic.location.column) + ' ' + diagnostic.message;
        }
        FAIL(failure);
    }
    REQUIRE_FALSE(result.value().bytecode.empty());
}

TEST_CASE("Material shader codegen rejects incompatible IR or ABI versions")
{
    arc::render::tools::material_graph_compilation compilation;
    compilation.ir.version = arc::render::tools::material_ir_version + 1;
    const auto bad_ir = arc::render::tools::generate_material_slang(compilation);
    REQUIRE_FALSE(bad_ir);
    REQUIRE(bad_ir.error().code == arc::render::shader_compile_error_code::validation_failed);

    compilation.ir.version = arc::render::tools::material_ir_version;
    compilation.descriptor.material_abi = arc::render::material_abi_version + 1;
    const auto bad_abi = arc::render::tools::generate_material_slang(compilation);
    REQUIRE_FALSE(bad_abi);
    REQUIRE(bad_abi.error().code == arc::render::shader_compile_error_code::validation_failed);
}
