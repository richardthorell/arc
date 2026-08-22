#include <arc/render_tools/render_tools.h>

#include <catch2/catch_test_macros.hpp>

#include <array>
#include <string>
#include <string_view>

namespace
{

constexpr std::string_view material_graph = R"({
  "version":1,
  "nodes":[
    {"id":"base","type":"vector3","values":{"value":[0.25,0.5,0.75]},
     "parameter":{"exposed":true,"name":"Base Color"}},
    {"id":"opacity","type":"constant","values":{"value":0.8}},
    {"id":"clip","type":"constant","values":{"value":0.5}},
    {"id":"material-output","type":"output","values":{}}
  ],
  "connections":[
    {"id":"1","from":{"nodeId":"base","pin":"value"},
     "to":{"nodeId":"material-output","pin":"baseColor"}},
    {"id":"2","from":{"nodeId":"opacity","pin":"value"},
     "to":{"nodeId":"material-output","pin":"opacity"}},
    {"id":"3","from":{"nodeId":"clip","pin":"value"},
     "to":{"nodeId":"material-output","pin":"alphaClip"}}
  ]
})";

constexpr std::string_view custom_material_shader = R"(
ArcSurfaceData arc_evaluate_material(ArcSurfaceInput input)
{
    ArcSurfaceData surface = arcDefaultSurface(input.normalWS);
    surface.baseColor = float3(0.12, 0.35, 0.8) * input.vertexColor.rgb;
    surface.roughness = 0.28;
    return surface;
}
)";

void require_compiles(arc::render::tools::slang_shader_compiler& compiler,
                      const arc::render::tools::material_pass_shader_source& generated, arc::render::material_pass pass)
{
    arc::render::shader_compile_request request{.source_path = "material_pass_test.generated.slang",
                                                .source_override = generated.source,
                                                .entry_point = generated.entry_point,
                                                .profile = "spirv_1_5",
                                                .library_version = "arc-material-pass/2",
                                                .domain = arc::render::shader_domain::surface,
                                                .stage = arc::render::shader_stage::fragment,
                                                .target = arc::render::shader_target::spirv,
                                                .optimization = arc::render::shader_optimization::development,
                                                .required_passes = {pass},
                                                .generated_line_nodes = generated.generated_line_nodes};
    const auto result = compiler.compile(request);
    if (!result)
    {
        std::string failure = result.error().message;
        for (const auto& diagnostic : result.error().diagnostics)
            failure += "\n" + diagnostic.location.path + ':' + std::to_string(diagnostic.location.line) + ':' +
                       std::to_string(diagnostic.location.column) + ' ' + diagnostic.message;
        FAIL(failure);
    }
    REQUIRE_FALSE(result.value().bytecode.empty());
    REQUIRE(result.value().reflection.passes.size() == 1);
    REQUIRE(result.value().reflection.passes.front().pass == pass);
}

} // namespace

TEST_CASE("Material IR composes deterministic engine-owned pass shaders")
{
    const auto compilation = arc::render::tools::compile_material_graph_json(material_graph);
    REQUIRE(compilation);

    arc::render::material_descriptor material;
    material.alpha_mode = arc::render::material_alpha_mode::masked;

    const auto first = arc::render::tools::generate_material_pass_slang(compilation.value(), material,
                                                                        arc::render::material_pass::gbuffer);
    const auto second = arc::render::tools::generate_material_pass_slang(compilation.value(), material,
                                                                         arc::render::material_pass::gbuffer);
    const auto shadow = arc::render::tools::generate_material_pass_slang(compilation.value(), material,
                                                                         arc::render::material_pass::shadow);

    REQUIRE(first);
    REQUIRE(second);
    REQUIRE(shadow);
    REQUIRE(first.value().source == second.value().source);
    REQUIRE(first.value().permutation == second.value().permutation);
    REQUIRE(first.value().permutation != shadow.value().permutation);
    REQUIRE(first.value().generated_line_nodes == second.value().generated_line_nodes);

    const auto& source = first.value().source;
    REQUIRE(source.find("arc_evaluate_material(arcMakeMaterialSurfaceInput(passInput))") != std::string::npos);
    REQUIRE(source.find("surface.opacity < surface.alphaCutoff") != std::string::npos);
    REQUIRE(source.find("SV_Target0") != std::string::npos);
    REQUIRE(source.find("SV_Target5") != std::string::npos);
    REQUIRE(source.find("output.motion = arcMaterialMotion(passInput)") != std::string::npos);
    REQUIRE(source.find("struct ArcCompilerInput") == std::string::npos);
    REQUIRE(first.value().generated_line_nodes.size() == 3);
}

TEST_CASE("opaque depth composition skips unnecessary material evaluation")
{
    const auto compilation = arc::render::tools::compile_material_graph_json(material_graph);
    REQUIRE(compilation);

    arc::render::material_descriptor material;
    const auto depth = arc::render::tools::generate_material_pass_slang(compilation.value(), material,
                                                                        arc::render::material_pass::depth);
    REQUIRE(depth);

    const auto main_position = depth.value().source.rfind("[shader(\"fragment\")] void main");
    REQUIRE(main_position != std::string::npos);
    REQUIRE(depth.value().source.substr(main_position).find("arc_evaluate_material") == std::string::npos);
}

TEST_CASE("handwritten Material Shaders use the same engine pass composer")
{
    const auto evaluator =
        arc::render::tools::make_custom_material_evaluator(custom_material_shader, "materials/custom_surface.slang");
    REQUIRE(evaluator);
    REQUIRE(evaluator.value().handwritten);
    REQUIRE(evaluator.value().source.find("struct ArcSurfaceData") != std::string::npos);
    REQUIRE(evaluator.value().source.find(custom_material_shader) != std::string::npos);

    arc::render::material_descriptor material;
    material.alpha_mode = arc::render::material_alpha_mode::masked;
    const auto gbuffer = arc::render::tools::generate_material_pass_slang(evaluator.value(), material,
                                                                          arc::render::material_pass::gbuffer);
    REQUIRE(gbuffer);
    REQUIRE(gbuffer.value().source.find("arc_evaluate_material(arcMakeMaterialSurfaceInput(passInput))") !=
            std::string::npos);
    REQUIRE(gbuffer.value().source.find("surface.opacity < surface.alphaCutoff") != std::string::npos);
}

TEST_CASE("handwritten Material Shaders cannot own render-pass entry points")
{
    const auto missing_evaluator = arc::render::tools::make_custom_material_evaluator("float helper() { return 1.0; }");
    REQUIRE_FALSE(missing_evaluator);

    const auto full_pass = arc::render::tools::make_custom_material_evaluator(
        "ArcSurfaceData arc_evaluate_material(ArcSurfaceInput input) { return arcDefaultSurface(input.normalWS); }\n"
        "[shader(\"fragment\")] float4 main() : SV_Target0 { return 1.0; }");
    REQUIRE_FALSE(full_pass);
}

TEST_CASE("compiled material pass shaders compile with pinned Slang")
{
    arc::render::tools::slang_shader_compiler compiler;
    if (!compiler.available())
    {
        SUCCEED("Pinned slangc is optional for this unit test environment");
        return;
    }

    const auto compilation = arc::render::tools::compile_material_graph_json(material_graph);
    REQUIRE(compilation);

    arc::render::material_descriptor material;
    material.alpha_mode = arc::render::material_alpha_mode::masked;
    constexpr std::array passes{arc::render::material_pass::depth,    arc::render::material_pass::shadow,
                                arc::render::material_pass::gbuffer,  arc::render::material_pass::forward,
                                arc::render::material_pass::motion,   arc::render::material_pass::object_id,
                                arc::render::material_pass::selection};

    for (const auto pass : passes)
    {
        const auto generated = arc::render::tools::generate_material_pass_slang(compilation.value(), material, pass);
        REQUIRE(generated);
        require_compiles(compiler, generated.value(), pass);
    }
}

TEST_CASE("handwritten Material Shader pass sources compile with pinned Slang")
{
    arc::render::tools::slang_shader_compiler compiler;
    if (!compiler.available())
    {
        SUCCEED("Pinned slangc is optional for this unit test environment");
        return;
    }

    const auto evaluator = arc::render::tools::make_custom_material_evaluator(custom_material_shader, "custom.slang");
    REQUIRE(evaluator);

    arc::render::material_descriptor material;
    material.alpha_mode = arc::render::material_alpha_mode::masked;
    constexpr std::array passes{arc::render::material_pass::depth,    arc::render::material_pass::shadow,
                                arc::render::material_pass::gbuffer,  arc::render::material_pass::forward,
                                arc::render::material_pass::motion,   arc::render::material_pass::object_id,
                                arc::render::material_pass::selection};
    for (const auto pass : passes)
    {
        const auto generated = arc::render::tools::generate_material_pass_slang(evaluator.value(), material, pass);
        REQUIRE(generated);
        require_compiles(compiler, generated.value(), pass);
    }
}
