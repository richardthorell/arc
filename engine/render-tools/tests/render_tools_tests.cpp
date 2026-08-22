#include <arc/render_tools/render_tools.h>

#include <catch2/catch_test_macros.hpp>

#include <cstdint>
#include <filesystem>
#include <fstream>

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
    const auto field =
        "arc_param_" + std::to_string(arc::render::make_shader_parameter_id("base-color-stable").representation());
    REQUIRE(first.value().source.find("arcMaterialParameters." + field) != std::string::npos);
    const auto mapped_line = first.value().generated_line_nodes.begin()->first;
    std::size_t cursor{};
    for (std::uint32_t line = 1; line < mapped_line; ++line)
        cursor = first.value().source.find('\n', cursor) + 1;
    REQUIRE(first.value().source.substr(cursor).starts_with("    float3 arc_node_base_color_stable_value"));
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

TEST_CASE("Slang compiler reports unavailable toolchains without throwing")
{
    const auto missing = std::filesystem::temp_directory_path() / "arc-definitely-missing-slangc";
    arc::render::tools::slang_shader_compiler compiler({.executable = missing, .require_pinned_version = false});

    REQUIRE_FALSE(compiler.available());
    REQUIRE(compiler.fingerprint() == "slang/unavailable");

    const auto result = compiler.compile({.source_path = "missing.slang",
                                          .entry_point = "main",
                                          .profile = "spirv_1_5",
                                          .domain = arc::render::shader_domain::surface,
                                          .stage = arc::render::shader_stage::fragment,
                                          .target = arc::render::shader_target::spirv});
    REQUIRE_FALSE(result);
    REQUIRE(result.error().code == arc::render::shader_compile_error_code::compiler_unavailable);
}

TEST_CASE("Slang compiler produces SPIR-V when the pinned toolchain is available")
{
    arc::render::tools::slang_shader_compiler compiler;
    if (!compiler.available())
    {
        SUCCEED("Pinned slangc is optional for this unit test environment");
        return;
    }

    const auto source = std::filesystem::temp_directory_path() / "arc_shader_compiler_test.slang";
    {
        std::ofstream file(source);
        file << "[shader(\"fragment\")] float4 main() : SV_Target { return float4(1, 0, 0, 1); }";
    }

    arc::render::shader_compile_request request{.source_path = source.string(),
                                                .entry_point = "main",
                                                .profile = "spirv_1_5",
                                                .domain = arc::render::shader_domain::surface,
                                                .stage = arc::render::shader_stage::fragment,
                                                .target = arc::render::shader_target::spirv,
                                                .optimization = arc::render::shader_optimization::development};

    auto result = compiler.compile(request);
    if (!result) INFO(result.error().message);
    REQUIRE(result);

    const auto& output = result.value();
    REQUIRE(output.bytecode.size() >= 4);
    REQUIRE(output.bytecode.size() % sizeof(std::uint32_t) == 0);
    REQUIRE(output.bytecode[0] == 0x03u);
    REQUIRE(output.bytecode[1] == 0x02u);
    REQUIRE(output.bytecode[2] == 0x23u);
    REQUIRE(output.bytecode[3] == 0x07u);
    REQUIRE(output.reflection.entry_points.size() == 1);
    REQUIRE(output.reflection.entry_points.front().name == "main");
    REQUIRE(output.reflection.entry_points.front().stage == arc::render::shader_stage::fragment);
    REQUIRE(compiler.fingerprint().starts_with("slang/"));

    std::filesystem::remove(source);
}
