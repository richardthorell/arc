#include <arc/render_tools/material_pass_codegen.h>

#include <sstream>
#include <string>
#include <string_view>
#include <utility>

namespace arc::render::tools
{
namespace
{

constexpr std::string_view standalone_harness_marker = "struct ArcCompilerInput\n";

void append_pass_input(std::ostringstream& source)
{
    source << "struct ArcMaterialPassInput\n"
              "{\n"
              "    float3 positionWS : TEXCOORD0;\n"
              "    float3 normalWS : TEXCOORD1;\n"
              "    float4 tangentWS : TEXCOORD2;\n"
              "    float2 uv0 : TEXCOORD3;\n"
              "    float2 uv1 : TEXCOORD4;\n"
              "    float4 vertexColor : COLOR0;\n"
              "    float3 viewWS : TEXCOORD5;\n"
              "    float4 clipPosition : TEXCOORD6;\n"
              "    float4 previousClipPosition : TEXCOORD7;\n"
              "    nointerpolation uint objectId : TEXCOORD8;\n"
              "};\n"
              "ArcSurfaceInput arcMakeMaterialSurfaceInput(ArcMaterialPassInput passInput)\n"
              "{\n"
              "    ArcSurfaceInput input;\n"
              "    input.positionWS = passInput.positionWS;\n"
              "    input.normalWS = passInput.normalWS;\n"
              "    input.tangentWS = passInput.tangentWS;\n"
              "    input.uv0 = passInput.uv0;\n"
              "    input.uv1 = passInput.uv1;\n"
              "    input.vertexColor = passInput.vertexColor;\n"
              "    input.viewWS = passInput.viewWS;\n"
              "    return input;\n"
              "}\n"
              "float2 arcMaterialMotion(ArcMaterialPassInput passInput)\n"
              "{\n"
              "    float2 currentNdc = passInput.clipPosition.xy / max(abs(passInput.clipPosition.w), 0.00001);\n"
              "    float2 previousNdc = passInput.previousClipPosition.xy / "
              "max(abs(passInput.previousClipPosition.w), 0.00001);\n"
              "    return (currentNdc - previousNdc) * 0.5;\n"
              "}\n";
}

void append_surface_evaluation(std::ostringstream& source, material_alpha_mode alpha_mode)
{
    source << "    ArcSurfaceData surface = arc_evaluate_material(arcMakeMaterialSurfaceInput(passInput));\n";
    if (alpha_mode == material_alpha_mode::masked)
        source << "    if (surface.opacity < surface.alphaCutoff) discard;\n";
}

void append_depth_or_shadow(std::ostringstream& source, const material_descriptor& material)
{
    source << "[shader(\"fragment\")] void main(ArcMaterialPassInput passInput)\n"
              "{\n";
    if (material.alpha_mode == material_alpha_mode::masked) append_surface_evaluation(source, material.alpha_mode);
    source << "}\n";
}

void append_gbuffer(std::ostringstream& source, const material_descriptor& material)
{
    source << "struct ArcMaterialGBufferOutput\n"
              "{\n"
              "    float4 albedo : SV_Target0;\n"
              "    float4 normalAo : SV_Target1;\n"
              "    float4 material : SV_Target2;\n"
              "    float4 emissive : SV_Target3;\n"
              "    float2 motion : SV_Target4;\n"
              "    uint objectId : SV_Target5;\n"
              "};\n"
              "[shader(\"fragment\")] ArcMaterialGBufferOutput main(ArcMaterialPassInput passInput)\n"
              "{\n";
    append_surface_evaluation(source, material.alpha_mode);
    source << "    ArcMaterialGBufferOutput output;\n"
              "    output.albedo = float4(surface.baseColor, surface.opacity);\n"
              "    output.normalAo = float4(normalize(surface.normalWS) * 0.5 + 0.5, surface.ambientOcclusion);\n"
              "    output.material = float4(saturate(surface.metallic), clamp(surface.roughness, 0.04, 1.0), 1.0, 0.0);\n"
              "    output.emissive = float4(surface.emissiveRadiance, 1.0);\n"
              "    output.motion = arcMaterialMotion(passInput);\n"
              "    output.objectId = passInput.objectId;\n"
              "    return output;\n"
              "}\n";
}

void append_forward(std::ostringstream& source, const material_descriptor& material)
{
    source << "[shader(\"fragment\")] float4 main(ArcMaterialPassInput passInput) : SV_Target0\n"
              "{\n";
    append_surface_evaluation(source, material.alpha_mode);
    source << "    return float4(surface.baseColor + surface.emissiveRadiance, surface.opacity);\n"
              "}\n";
}

void append_motion(std::ostringstream& source, const material_descriptor& material)
{
    source << "[shader(\"fragment\")] float2 main(ArcMaterialPassInput passInput) : SV_Target0\n"
              "{\n";
    if (material.alpha_mode == material_alpha_mode::masked) append_surface_evaluation(source, material.alpha_mode);
    source << "    return arcMaterialMotion(passInput);\n"
              "}\n";
}

void append_object_id(std::ostringstream& source)
{
    source << "[shader(\"fragment\")] uint main(ArcMaterialPassInput passInput) : SV_Target0\n"
              "{\n"
              "    return passInput.objectId;\n"
              "}\n";
}

void append_selection(std::ostringstream& source)
{
    source << "[shader(\"fragment\")] float4 main(ArcMaterialPassInput passInput) : SV_Target0\n"
              "{\n"
              "    return float4(1.0, 1.0, 1.0, 1.0);\n"
              "}\n";
}

} // namespace

material_pass_codegen_result generate_material_pass_slang(const material_graph_compilation& compilation,
                                                           const material_descriptor& material, material_pass pass,
                                                           std::uint8_t debug_view, bool wireframe)
{
    if (!material_supports_pass(material, pass))
        return material_pass_codegen_result::failure(
            {.code = shader_compile_error_code::validation_failed,
             .message = "material is not eligible for the requested render pass"});
    if (pass == material_pass::ray_hit)
        return material_pass_codegen_result::failure(
            {.code = shader_compile_error_code::validation_failed,
             .message = "ray-hit material composition is not implemented by material pass contract v1"});

    auto evaluator = generate_material_slang(compilation);
    if (!evaluator) return material_pass_codegen_result::failure(evaluator.error());

    auto evaluator_source = std::move(evaluator).value();
    const auto marker = evaluator_source.source.find(standalone_harness_marker);
    if (marker == std::string::npos)
        return material_pass_codegen_result::failure(
            {.code = shader_compile_error_code::validation_failed,
             .message = "generated material evaluator is missing the Stage 7 standalone harness boundary"});
    evaluator_source.source.resize(marker);

    std::ostringstream pass_source;
    pass_source << evaluator_source.source;
    pass_source << "// ARC engine material pass contract v" << material_pass_contract_version << "; codegen v"
                << material_pass_codegen_version << ".\n";
    append_pass_input(pass_source);

    switch (pass)
    {
        case material_pass::depth:
        case material_pass::shadow:
            append_depth_or_shadow(pass_source, material);
            break;
        case material_pass::gbuffer:
            append_gbuffer(pass_source, material);
            break;
        case material_pass::forward:
            append_forward(pass_source, material);
            break;
        case material_pass::motion:
            append_motion(pass_source, material);
            break;
        case material_pass::object_id:
            append_object_id(pass_source);
            break;
        case material_pass::selection:
            append_selection(pass_source);
            break;
        case material_pass::ray_hit:
            break;
    }

    const auto key = make_material_pass_permutation_key(material, pass, debug_view, wireframe);
    return material_pass_codegen_result::success({.pass = pass,
                                                  .permutation = make_material_pass_permutation_id(key),
                                                  .source = std::move(pass_source).str(),
                                                  .generated_line_nodes = std::move(evaluator_source.generated_line_nodes),
                                                  .parameters = std::move(evaluator_source.parameters),
                                                  .diagnostics = std::move(evaluator_source.diagnostics)});
}

} // namespace arc::render::tools
