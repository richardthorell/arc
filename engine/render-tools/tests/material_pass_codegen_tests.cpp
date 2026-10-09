#include <arc/render_tools/render_tools.h>

#include <catch2/catch_test_macros.hpp>

#include <array>
#include <algorithm>
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

constexpr std::string_view transmission_material_shader = R"(
ArcSurfaceData arc_evaluate_material(ArcSurfaceInput input)
{
    ArcSurfaceData surface = arcDefaultSurface(input.normalWS);
    surface.baseColor = float3(0.08, 0.22, 0.28);
    surface.roughness = 0.08;
    surface.opacity = 0.72;
    surface.transmission = 0.82;
    surface.thickness = 6.0;
    surface.indexOfRefraction = 1.333;
    surface.attenuationColor = float3(0.65, 0.88, 0.92);
    surface.attenuationDistance = 12.0;
    return surface;
}
)";

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
    if (pass == arc::render::material_pass::forward)
    {
        constexpr std::array<std::string_view, 6> names{"arcVirtualShadowAddresses",    "arcVirtualShadowViews",
                                                        "arcVirtualShadowPages",        "arcVirtualShadowStaticAtlas",
                                                        "arcVirtualShadowDynamicAtlas", "arcVirtualShadowSampler"};
        const auto& resources = result.value().reflection.resources;
        for (std::uint32_t index = 0; index < names.size(); ++index)
        {
            const auto found =
                std::ranges::find(resources, names[index], &arc::render::shader_resource_descriptor::name);
            if (!generated.virtual_shadow_sampling)
            {
                REQUIRE(found == resources.end());
                continue;
            }
            REQUIRE(found != resources.end());
            CHECK(found->set == 2u);
            CHECK(found->binding == 10u + index);
            CHECK_FALSE(found->writable);
            const auto expected = index < 3u   ? arc::render::shader_resource_kind::structured_buffer
                                  : index < 5u ? arc::render::shader_resource_kind::sampled_texture
                                               : arc::render::shader_resource_kind::sampler;
            CHECK(found->kind == expected);
        }
    }
}

} // namespace

TEST_CASE("forward shadow variants have distinct identities and conventional shaders omit VSM resources")
{
    using namespace arc::render;
    using namespace arc::render::tools;
    const auto evaluator = make_custom_material_evaluator(transmission_material_shader);
    REQUIRE(evaluator);
    material_descriptor material;
    material.alpha_mode = material_alpha_mode::blend;
    const auto conventional =
        generate_material_pass_slang(evaluator.value(), material, material_pass::forward, 0, false, false);
    const auto virtualized =
        generate_material_pass_slang(evaluator.value(), material, material_pass::forward, 0, false, true);
    REQUIRE(conventional);
    REQUIRE(virtualized);
    CHECK(conventional.value().permutation != virtualized.value().permutation);
    CHECK_FALSE(conventional.value().virtual_shadow_sampling);
    CHECK(virtualized.value().virtual_shadow_sampling);
    slang_shader_compiler compiler;
    if (compiler.available())
    {
        require_compiles(compiler, conventional.value(), material_pass::forward);
        require_compiles(compiler, virtualized.value(), material_pass::forward);
    }
}

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
    REQUIRE(source.find("saturate(surface.clearCoat)") != std::string::npos);
    REQUIRE(source.find("clamp(surface.clearCoatRoughness, 0.04, 1.0)") != std::string::npos);
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

TEST_CASE("canonical forward pass evaluates generic PBR transmission from the Material ABI")
{
    const auto evaluator =
        arc::render::tools::make_custom_material_evaluator(transmission_material_shader, "materials/glass.slang");
    REQUIRE(evaluator);

    arc::render::material_descriptor material;
    material.shading_model = arc::render::material_shading_model::transmission;
    material.render_path = arc::render::material_render_path::clustered_forward;
    material.deferred_compatible = false;
    material.alpha_mode = arc::render::material_alpha_mode::blend;

    const auto forward = arc::render::tools::generate_material_pass_slang(evaluator.value(), material,
                                                                          arc::render::material_pass::forward);
    REQUIRE(forward);

    const auto& source = forward.value().source;
    REQUIRE(source.find("StructuredBuffer<ArcForwardLightingData> arcForwardLighting : register(t0, space2)") !=
            std::string::npos);
    REQUIRE(source.find("arcForwardDirectionalShadowMap") != std::string::npos);
    REQUIRE(source.find("arcVirtualShadowAddresses : register(t10, space2)") != std::string::npos);
    REQUIRE(source.find("arcVirtualShadowSampler : register(s15, space2)") != std::string::npos);
    REQUIRE(source.find("arc_virtual_directional_shadow_visibility(light, worldPosition") != std::string::npos);
    REQUIRE(source.find("address_space.identityTopology.x != light.shadow_identity.w") != std::string::npos);
    REQUIRE(source.find("sampled += min(static_visibility, dynamic_visibility)") != std::string::npos);
    REQUIRE(source.find("float4 directionIntensity;\n    float4 colorFlags;\n"
                        "    uint4 shadowIdentity;\n    uint4 shadowRouting;\n    float4 shadowParameters;") !=
            std::string::npos);
    REQUIRE(source.find("lighting.directionalLights[index].shadowRouting.x != 0u") != std::string::npos);
    REQUIRE(source.find("float shadow = index == 0u") == std::string::npos);
    REQUIRE(source.find("arcForwardLocalShadowAtlas") != std::string::npos);
    REQUIRE(source.find("arcForwardSceneColor") != std::string::npos);
    REQUIRE(source.find("arcForwardShadows") != std::string::npos);
    REQUIRE(source.find("arcForwardScene") != std::string::npos);
    REQUIRE(source.find("arcForwardF0FromIor") != std::string::npos);
    REQUIRE(source.find("arcForwardFresnelSchlick") != std::string::npos);
    REQUIRE(source.find("arcForwardBeerLambert") != std::string::npos);
    REQUIRE(source.find("surface.thickness") != std::string::npos);
    REQUIRE(source.find("surface.attenuationColor") != std::string::npos);
    REQUIRE(source.find("surface.attenuationDistance") != std::string::npos);
    REQUIRE(source.find("surface.transmission") != std::string::npos);
    REQUIRE(source.find("arcEvaluateForwardSurface(surface, surfaceInput, passInput)") != std::string::npos);
    REQUIRE(source.find("water") == std::string::npos);
}

TEST_CASE("generic transmission material forward source compiles with pinned Slang")
{
    arc::render::tools::slang_shader_compiler compiler;
    if (!compiler.available())
    {
        SUCCEED("Pinned slangc is optional for this unit test environment");
        return;
    }

    const auto evaluator =
        arc::render::tools::make_custom_material_evaluator(transmission_material_shader, "materials/glass.slang");
    REQUIRE(evaluator);

    arc::render::material_descriptor material;
    material.shading_model = arc::render::material_shading_model::transmission;
    material.render_path = arc::render::material_render_path::clustered_forward;
    material.deferred_compatible = false;
    material.alpha_mode = arc::render::material_alpha_mode::blend;

    const auto forward = arc::render::tools::generate_material_pass_slang(evaluator.value(), material,
                                                                          arc::render::material_pass::forward);
    REQUIRE(forward);
    require_compiles(compiler, forward.value(), arc::render::material_pass::forward);
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
