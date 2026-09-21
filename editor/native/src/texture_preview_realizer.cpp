#include <arc/editor/texture_preview_realizer.h>

#include <arc/render_tools/render_tools.h>

#include <algorithm>
#include <memory>
#include <string_view>

namespace arc::editor
{
namespace
{
constexpr std::string_view preview_source = R"(
Texture2D<float4> arcMaterialTextures2D[1];
SamplerState arcMaterialSampler;

cbuffer arcMaterialParameters
{
    float4 texturePreviewChannels;
    float4 texturePreviewControls;
};

ArcSurfaceData arc_evaluate_material(ArcSurfaceInput input)
{
    uint width = 1;
    uint height = 1;
    arcMaterialTextures2D[0].GetDimensions(width, height);

    float2 uv = saturate(input.uv0);
    float4 sampled;
    if (texturePreviewControls.y > 0.5)
    {
        int2 pixel = int2(min(uint2(uv * float2(width, height)), uint2(width - 1, height - 1)));
        sampled = arcMaterialTextures2D[0].Load(int3(pixel, 0));
    }
    else
    {
        sampled = arcMaterialTextures2D[0].Sample(arcMaterialSampler, uv);
    }

    float3 color = sampled.rgb * texturePreviewChannels.rgb;
    const float rgbChannels = texturePreviewChannels.x + texturePreviewChannels.y + texturePreviewChannels.z;
    if (rgbChannels < 0.5 && texturePreviewChannels.w > 0.5)
        color = sampled.aaa;

    float alpha = texturePreviewChannels.w > 0.5 ? sampled.a : 1.0;
    if (texturePreviewControls.z > 0.5 && alpha < 0.9999)
    {
        uint2 pixel = uint2(uv * float2(width, height));
        float checker = ((pixel.x / 16u + pixel.y / 16u) & 1u) != 0u ? 0.22 : 0.12;
        color = lerp(float3(checker), color, alpha);
    }

    color *= exp2(texturePreviewControls.x);

    ArcSurfaceData surface = arcDefaultSurface(float3(0.0, 1.0, 0.0));
    surface.baseColor = float3(0.0);
    surface.emissiveRadiance = color;
    surface.opacity = 1.0;
    surface.roughness = 1.0;
    return surface;
}
)";

std::shared_ptr<render::material_runtime_program> compile_texture_preview_program(std::string& message)
{
    auto evaluator = render::tools::make_custom_material_evaluator(preview_source, "texture_preview_v1.slang");
    if (!evaluator)
    {
        message = "Texture preview evaluator generation failed: " + evaluator.error().message;
        return {};
    }

    render::material_descriptor descriptor;
    descriptor.name = "Texture Preview V1";
    descriptor.domain = render::material_domain::surface;
    descriptor.shading_model = render::material_shading_model::unlit;
    descriptor.alpha_mode = render::material_alpha_mode::opaque;
    descriptor.double_sided = true;
    descriptor.cast_shadows = false;

    auto generated =
        render::tools::generate_material_pass_slang(evaluator.value(), descriptor, render::material_pass::gbuffer);
    if (!generated)
    {
        message = "Texture preview shader generation failed: " + generated.error().message;
        return {};
    }

    render::tools::slang_shader_compiler compiler;
    if (!compiler.available())
    {
        message = "Pinned Slang compiler is unavailable for the native texture preview";
        return {};
    }

    render::shader_compile_request request{.source_path = "texture_preview_v1.generated.slang",
                                           .source_override = generated.value().source,
                                           .entry_point = generated.value().entry_point,
                                           .profile = "spirv_1_5",
                                           .library_version = "arc-texture-preview/1",
                                           .domain = render::shader_domain::surface,
                                           .stage = render::shader_stage::fragment,
                                           .target = render::shader_target::spirv,
                                           .optimization = render::shader_optimization::development,
                                           .required_passes = {render::material_pass::gbuffer}};
    auto compiled = compiler.compile(request);
    if (!compiled)
    {
        message = "Texture preview shader compilation failed: " + compiled.error().message;
        return {};
    }

    auto program = std::make_shared<render::material_runtime_program>();
    program->contract_version = render::material_pass_contract_version;
    program->material_abi = render::material_abi_version;
    program->generation = 1u;
    program->uses_texture_sampling = true;
    program->texture_bindings.push_back(
        {.slot = 0u, .type = render::shader_parameter_type::texture_2d, .dimension_slot = 0u});

    for (const auto& reflected : compiled.value().reflection.parameters)
    {
        if (reflected.name != "texturePreviewChannels" && reflected.name != "texturePreviewControls") continue;
        program->parameters.push_back(reflected);
        program->parameter_block_size =
            std::max(program->parameter_block_size, reflected.offset + reflected.size);
    }
    if (program->parameters.size() != 2u)
    {
        message = "Texture preview shader reflection is missing display parameters";
        return {};
    }

    program->parameter_defaults.assign(program->parameter_block_size, std::byte{});
    program->passes.push_back({.pass = render::material_pass::gbuffer,
                               .permutation = generated.value().permutation,
                               .compiled = std::move(compiled).value()});
    message = "Native Texture2D preview shader compiled";
    return program;
}

void set_parameter(render::material_descriptor& material, std::string_view name, const math::vector4f& value)
{
    const auto id = render::make_shader_parameter_id(name);
    const auto found = std::ranges::find(material.parameters, id, &render::material_parameter_override::id);
    if (found != material.parameters.end())
    {
        found->value = value;
        return;
    }
    material.parameters.push_back({.id = id, .name = std::string{name}, .value = value});
}
} // namespace

void apply_texture_preview_options(render::material_descriptor& material,
                                   const texture_preview_display_options& options)
{
    set_parameter(material, "texturePreviewChannels", options.channels);
    set_parameter(material, "texturePreviewControls",
                  {options.exposure_ev, options.nearest ? 1.0f : 0.0f, options.checkerboard ? 1.0f : 0.0f, 0.0f});
}

texture_preview_material_result
realize_texture_preview_material(render::texture_handle texture, const texture_preview_display_options& options)
{
    texture_preview_material_result result;
    if (!texture.valid())
    {
        result.message = "Texture preview requires a valid renderer texture";
        return result;
    }

    auto program = compile_texture_preview_program(result.message);
    if (!program) return result;

    result.material.name = "Native Texture2D Preview";
    result.material.domain = render::material_domain::surface;
    result.material.shading_model = render::material_shading_model::unlit;
    result.material.render_path = render::material_render_path::deferred;
    result.material.alpha_mode = render::material_alpha_mode::opaque;
    result.material.double_sided = true;
    result.material.cast_shadows = false;
    result.material.runtime_program = std::move(program);
    result.material.runtime_textures = {texture};
    apply_texture_preview_options(result.material, options);
    result.succeeded = true;
    result.message = "Native Texture2D preview material realized";
    return result;
}

} // namespace arc::editor
