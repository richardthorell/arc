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

constexpr std::string_view custom_material_abi =
    R"(// ARC Material ABI v2. Engine-owned declarations for handwritten Material Shaders.
static const uint ARC_MATERIAL_ABI_VERSION = 2;
float4 arcSampleTexture2D(Texture2D<float4> textureResource, SamplerState samplerResource, float2 uv,
                          uint textureMetadataIndex)
{
    return textureResource.Sample(samplerResource, uv);
}
float4 arcSampleTextureCube(TextureCube<float4> textureResource, SamplerState samplerResource, float3 direction,
                            uint textureMetadataIndex)
{
    return textureResource.Sample(samplerResource, direction);
}
float4 arcSampleTexture3D(Texture3D<float4> textureResource, SamplerState samplerResource, float3 coordinates,
                          uint textureMetadataIndex)
{
    return textureResource.Sample(samplerResource, coordinates);
}
struct ArcSurfaceInput
{
    float3 positionWS;
    float3 normalWS;
    float4 tangentWS;
    float2 uv0;
    float2 uv1;
    float4 vertexColor;
    float3 viewWS;
};
struct ArcSurfaceData
{
    float3 baseColor;
    float metallic;
    float roughness;
    float3 normalWS;
    float3 clearCoatNormalWS;
    float3 tangentWS;
    float ambientOcclusion;
    float3 emissiveRadiance;
    float opacity;
    float alphaCutoff;
    float indexOfRefraction;
    float clearCoat;
    float clearCoatRoughness;
    float sheen;
    float3 sheenColor;
    float sheenRoughness;
    float anisotropy;
    float anisotropyRotation;
    float transmission;
    float thickness;
    float3 attenuationColor;
    float attenuationDistance;
    float3 subsurfaceColor;
    float subsurface;
};
ArcSurfaceData arcDefaultSurface(float3 normalWS)
{
    ArcSurfaceData surface;
    surface.baseColor = float3(0.8);
    surface.metallic = 0.0;
    surface.roughness = 0.6;
    surface.normalWS = normalize(normalWS);
    surface.clearCoatNormalWS = surface.normalWS;
    surface.tangentWS = float3(1.0, 0.0, 0.0);
    surface.ambientOcclusion = 1.0;
    surface.emissiveRadiance = float3(0.0);
    surface.opacity = 1.0;
    surface.alphaCutoff = 0.5;
    surface.indexOfRefraction = 1.5;
    surface.clearCoat = 0.0;
    surface.clearCoatRoughness = 0.1;
    surface.sheen = 0.0;
    surface.sheenColor = float3(0.0);
    surface.sheenRoughness = 0.5;
    surface.anisotropy = 0.0;
    surface.anisotropyRotation = 0.0;
    surface.transmission = 0.0;
    surface.thickness = 0.0;
    surface.attenuationColor = float3(1.0);
    surface.attenuationDistance = 1.0;
    surface.subsurfaceColor = float3(1.0, 0.35, 0.2);
    surface.subsurface = 0.0;
    return surface;
}
)";

void append_pass_input(std::ostringstream& source)
{
    // Keep this interface in exact location order with assets/shaders/gbuffer.vert. Slang maps the
    // TEXCOORD semantic sequence to the Vulkan user-varying locations without backend annotations.
    source << "struct ArcMaterialPassInput\n"
              "{\n"
              "    float3 normalWS : TEXCOORD0;\n"
              "    float3 positionWS : TEXCOORD1;\n"
              "    float4 vertexColor : TEXCOORD2;\n"
              "    float2 uv0 : TEXCOORD3;\n"
              "    float4 tangentWS : TEXCOORD4;\n"
              "    float4 clipPosition : TEXCOORD5;\n"
              "    float4 previousClipPosition : TEXCOORD6;\n"
              "    nointerpolation uint objectId : TEXCOORD7;\n"
              "    float3 viewWS : TEXCOORD8;\n"
              "};\n"
              "ArcSurfaceInput arcMakeMaterialSurfaceInput(ArcMaterialPassInput passInput)\n"
              "{\n"
              "    ArcSurfaceInput input;\n"
              "    input.positionWS = passInput.positionWS;\n"
              "    input.normalWS = passInput.normalWS;\n"
              "    input.tangentWS = passInput.tangentWS;\n"
              "    input.uv0 = passInput.uv0;\n"
              "    input.uv1 = passInput.uv0;\n"
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
    source
        << "    ArcMaterialGBufferOutput output;\n"
           "    output.albedo = float4(surface.baseColor, surface.opacity);\n"
           "    output.normalAo = float4(normalize(surface.normalWS) * 0.5 + 0.5, surface.ambientOcclusion);\n"
           "    output.material = float4(saturate(surface.metallic), clamp(surface.roughness, 0.04, 1.0), "
           "saturate(surface.clearCoat), clamp(surface.clearCoatRoughness, 0.04, 1.0));\n"
           "    output.emissive = float4(surface.emissiveRadiance, 1.0);\n"
           "    output.motion = arcMaterialMotion(passInput);\n"
           "    output.objectId = passInput.objectId;\n"
           "    return output;\n"
           "}\n";
}

void append_forward_lighting_library(std::ostringstream& source)
{
    source << R"(struct ArcForwardDirectionalLight
{
    float4 directionIntensity;
    float4 colorFlags;
};

struct ArcForwardPointLight
{
    float4 positionRange;
    float4 colorIntensity;
    float4 objectIdShadow;
    float4 shadowParameters;
};

struct ArcForwardSpotLight
{
    float4 positionRange;
    float4 directionInnerAngle;
    float4 colorIntensity;
    float4 params;
    float4 objectIdShadow;
    float4 shadowParameters;
};

struct ArcForwardAreaLight
{
    float4 positionShape;
    float4 directionTwoSided;
    float4 tangentWidth;
    float4 colorIntensity;
    float4 dimensionsShadow;
};

struct ArcForwardLocalShadowFace
{
    float4x4 lightViewProjection;
    float4 atlasRect;
    float4 parameters;
};

struct ArcForwardLightingData
{
    ArcForwardDirectionalLight directionalLights[4];
    ArcForwardPointLight pointLights[64];
    ArcForwardSpotLight spotLights[64];
    ArcForwardAreaLight areaLights[32];
    ArcForwardLocalShadowFace localShadowFaces[144];
    float4 ambientColorIntensity;
    uint directionalCount;
    uint pointCount;
    uint spotCount;
    uint areaCount;
    uint skippedDirectionalCount;
    uint skippedPointCount;
    uint skippedSpotCount;
    uint skippedAreaCount;
    uint localShadowFaceCount;
    uint localShadowPadding0;
    uint localShadowPadding1;
    uint localShadowPadding2;
};

struct ArcForwardShadowData
{
    float4x4 lightViewProjection[4];
    float4 cascadeSplits;
    float4 params;
    float4 cascadeTexelSize;
    float4 cascadeBlendStarts;
    float4 configuration;
};

struct ArcForwardSceneData
{
    float4 cameraPositionViewportWidth;
    float4 fogColorDensity;
    float4 fogParamsViewportHeight;
};

StructuredBuffer<ArcForwardLightingData> arcForwardLighting : register(t0, space2);
Texture2DArray<float> arcForwardDirectionalShadowMap : register(t1, space2);
SamplerComparisonState arcForwardDirectionalShadowSampler : register(s2, space2);
Texture2D<float> arcForwardLocalShadowAtlas : register(t3, space2);
SamplerComparisonState arcForwardLocalShadowSampler : register(s4, space2);
Texture2D<float4> arcForwardSceneColor : register(t5, space2);
SamplerState arcForwardSceneColorSampler : register(s6, space2);
ConstantBuffer<ArcForwardShadowData> arcForwardShadows : register(b7, space2);
ConstantBuffer<ArcForwardSceneData> arcForwardScene : register(b8, space2);

static const float ARC_FORWARD_PI = 3.14159265358979323846;

float3 arcForwardF0FromIor(float ior)
{
    float ratio = (ior - 1.0) / max(ior + 1.0, 1e-4);
    return float3(ratio * ratio);
}

float3 arcForwardFresnelSchlick(float cosTheta, float3 f0)
{
    float factor = pow(saturate(1.0 - cosTheta), 5.0);
    return f0 + (1.0 - f0) * factor;
}

float arcForwardD_GGX(float nDotH, float roughness)
{
    float alpha = max(roughness * roughness, 0.002);
    float alpha2 = alpha * alpha;
    float denominator = nDotH * nDotH * (alpha2 - 1.0) + 1.0;
    return alpha2 / max(ARC_FORWARD_PI * denominator * denominator, 1e-6);
}

float arcForwardV_SmithGGXCorrelated(float nDotV, float nDotL, float roughness)
{
    float alpha2 = pow(max(roughness, 0.02), 4.0);
    float gv = nDotL * sqrt(max(nDotV * nDotV * (1.0 - alpha2) + alpha2, 1e-6));
    float gl = nDotV * sqrt(max(nDotL * nDotL * (1.0 - alpha2) + alpha2, 1e-6));
    return 0.5 / max(gv + gl, 1e-5);
}

float arcForwardBurleyDiffuse(float nDotV, float nDotL, float lDotH, float roughness)
{
    float energyBias = lerp(0.0, 0.5, roughness);
    float energyFactor = lerp(1.0, 1.0 / 1.51, roughness);
    float fd90 = energyBias + 2.0 * lDotH * lDotH * roughness;
    float lightScatter = 1.0 + (fd90 - 1.0) * pow(1.0 - nDotL, 5.0);
    float viewScatter = 1.0 + (fd90 - 1.0) * pow(1.0 - nDotV, 5.0);
    return lightScatter * viewScatter * energyFactor / ARC_FORWARD_PI;
}

float3 arcForwardBeerLambert(float3 attenuationColor, float distance, float attenuationDistance)
{
    if (attenuationDistance <= 0.0) return float3(1.0);
    float3 coefficient = -log(max(attenuationColor, float3(1e-4))) / attenuationDistance;
    return exp(-coefficient * max(distance, 0.0));
}

float3 arcForwardEvaluateLight(ArcSurfaceData surface, float3 viewWS, float3 lightWS, float3 radiance,
                               float visibility)
{
    float3 normalWS = normalize(surface.normalWS);
    float3 halfWS = normalize(viewWS + lightWS);
    float nDotV = saturate(dot(normalWS, viewWS));
    float nDotL = saturate(dot(normalWS, lightWS));
    float nDotH = saturate(dot(normalWS, halfWS));
    float lDotH = saturate(dot(lightWS, halfWS));
    if (nDotL <= 0.0 || visibility <= 0.0) return float3(0.0);

    float roughness = clamp(surface.roughness, 0.04, 1.0);
    float metallic = saturate(surface.metallic);
    float transmission = saturate(surface.transmission) * (1.0 - metallic);
    float3 dielectricF0 = arcForwardF0FromIor(max(surface.indexOfRefraction, 1.0001));
    float3 f0 = lerp(dielectricF0, surface.baseColor, metallic);
    float distribution = arcForwardD_GGX(nDotH, roughness);
    float visibilityTerm = arcForwardV_SmithGGXCorrelated(nDotV, nDotL, roughness);
    float3 specular = distribution * visibilityTerm * arcForwardFresnelSchlick(lDotH, f0);
    float diffuseWeight = (1.0 - metallic) * (1.0 - transmission);
    float3 diffuse = surface.baseColor * diffuseWeight *
                     arcForwardBurleyDiffuse(nDotV, nDotL, lDotH, roughness);

    float clearCoat = saturate(surface.clearCoat);
    float coatRoughness = clamp(surface.clearCoatRoughness, 0.04, 1.0);
    float coatDistribution = arcForwardD_GGX(nDotH, coatRoughness);
    float coatVisibility = arcForwardV_SmithGGXCorrelated(nDotV, nDotL, coatRoughness);
    float3 coatFresnel = arcForwardFresnelSchlick(lDotH, float3(0.04));
    float3 coat = clearCoat * coatDistribution * coatVisibility * coatFresnel;
    float3 baseEnergy = 1.0 - clearCoat * coatFresnel;
    return ((diffuse + specular) * baseEnergy + coat) * radiance * nDotL * visibility;
}

uint arcForwardPointShadowFace(float3 direction)
{
    float3 absoluteDirection = abs(direction);
    if (absoluteDirection.x >= absoluteDirection.y && absoluteDirection.x >= absoluteDirection.z)
        return direction.x >= 0.0 ? 0u : 1u;
    if (absoluteDirection.y >= absoluteDirection.z)
        return direction.y >= 0.0 ? 2u : 3u;
    return direction.z >= 0.0 ? 4u : 5u;
}

float arcForwardSampleLocalShadowFace(uint faceIndex, float3 worldPosition)
{
    ArcForwardLightingData lighting = arcForwardLighting[0];
    if (faceIndex >= min(lighting.localShadowFaceCount, 144u)) return 1.0;
    ArcForwardLocalShadowFace face = lighting.localShadowFaces[faceIndex];
    float4 clip = mul(face.lightViewProjection, float4(worldPosition, 1.0));
    if (clip.w <= 0.0) return 1.0;
    float3 projected = clip.xyz / clip.w;
    float2 uv = projected.xy * 0.5 + 0.5;
    if (projected.z <= 0.0 || projected.z >= 1.0 || any(uv < 0.0) || any(uv > 1.0)) return 1.0;
    float2 atlasUv = face.atlasRect.xy + uv * face.atlasRect.zw;
    float comparison = projected.z - face.parameters.y;
    float texel = face.parameters.x;
    float result = 0.0;
    [unroll]
    for (int y = -1; y <= 1; ++y)
        [unroll]
        for (int x = -1; x <= 1; ++x)
            result += arcForwardLocalShadowAtlas.SampleCmpLevelZero(
                arcForwardLocalShadowSampler, atlasUv + float2(x, y) * texel, comparison);
    return result / 9.0;
}

float arcForwardPointShadow(ArcForwardPointLight light, float3 worldPosition)
{
    int firstFace = int(light.shadowParameters.x + 0.5);
    if (firstFace < 0 || light.shadowParameters.y < 5.5) return 1.0;
    uint face = arcForwardPointShadowFace(worldPosition - light.positionRange.xyz);
    float sampled = arcForwardSampleLocalShadowFace(uint(firstFace) + face, worldPosition);
    return lerp(1.0, sampled, saturate(light.shadowParameters.z));
}

float arcForwardSpotShadow(ArcForwardSpotLight light, float3 worldPosition)
{
    int face = int(light.shadowParameters.x + 0.5);
    if (face < 0) return 1.0;
    float sampled = arcForwardSampleLocalShadowFace(uint(face), worldPosition);
    return lerp(1.0, sampled, saturate(light.shadowParameters.z));
}

int arcForwardShadowCascade(float cameraDistance)
{
    int cascadeCount = clamp(int(arcForwardShadows.configuration.x + 0.5), 0, 4);
    [unroll]
    for (int cascade = 0; cascade < 4; ++cascade)
        if (cascade < cascadeCount && cameraDistance <= arcForwardShadows.cascadeSplits[cascade]) return cascade;
    return -1;
}

)";
    source << R"(float arcForwardSampleDirectionalCascade(int cascade, float3 worldPosition, float3 surfaceNormal,
                                         float3 lightDirection)
{
    float4 lightClip = mul(arcForwardShadows.lightViewProjection[cascade], float4(worldPosition, 1.0));
    float3 projected = lightClip.xyz / max(abs(lightClip.w), 1e-6);
    float2 uv = projected.xy * 0.5 + 0.5;
    if (any(uv < 0.0) || any(uv > 1.0) || projected.z < 0.0 || projected.z > 1.0) return 1.0;
    float normalBias = arcForwardShadows.params.z *
                       saturate(1.0 - dot(normalize(surfaceNormal), normalize(lightDirection)));
    float compareDepth = projected.z - arcForwardShadows.params.y - normalBias;
    return min(
        arcForwardDirectionalShadowMap.SampleCmpLevelZero(
            arcForwardDirectionalShadowSampler, float3(uv, float(cascade)), compareDepth),
        arcForwardDirectionalShadowMap.SampleCmpLevelZero(
            arcForwardDirectionalShadowSampler, float3(uv, float(cascade + 4)), compareDepth));
}

float arcForwardDirectionalShadow(float3 worldPosition, float3 surfaceNormal, float3 lightDirection)
{
    if (arcForwardShadows.params.x <= 0.0 || arcForwardShadows.configuration.x < 0.5) return 1.0;
    float3 cameraPosition = arcForwardScene.cameraPositionViewportWidth.xyz;
    float3 cameraForward = normalize(arcForwardShadows.configuration.yzw);
    float cameraDepth = max(dot(worldPosition - cameraPosition, cameraForward), 0.0);
    int cascade = arcForwardShadowCascade(cameraDepth);
    if (cascade < 0) return 1.0;
    float visibility = arcForwardSampleDirectionalCascade(cascade, worldPosition, surfaceNormal, lightDirection);
    int cascadeCount = clamp(int(arcForwardShadows.configuration.x + 0.5), 0, 4);
    if (cascade + 1 < cascadeCount)
    {
        float blendStart = arcForwardShadows.cascadeBlendStarts[cascade];
        float blendEnd = arcForwardShadows.cascadeSplits[cascade];
        float blend = smoothstep(blendStart, max(blendEnd, blendStart + 1e-5), cameraDepth);
        if (blend > 0.0)
            visibility = lerp(
                visibility,
                arcForwardSampleDirectionalCascade(cascade + 1, worldPosition, surfaceNormal, lightDirection),
                blend);
    }
    return lerp(1.0 - arcForwardShadows.params.x, 1.0, visibility);
}

float3 arcEvaluateForwardSurface(ArcSurfaceData surface, ArcSurfaceInput input, ArcMaterialPassInput passInput)
{
    ArcForwardLightingData lighting = arcForwardLighting[0];
    float3 viewWS = normalize(input.viewWS);
    float3 direct = float3(0.0);

    [loop]
    for (uint index = 0u; index < min(lighting.directionalCount, 4u); ++index)
    {
        float3 lightWS = normalize(-lighting.directionalLights[index].directionIntensity.xyz);
        float3 radiance = lighting.directionalLights[index].colorFlags.rgb *
                          lighting.directionalLights[index].directionIntensity.w;
        float shadow = index == 0u
                           ? arcForwardDirectionalShadow(input.positionWS, surface.normalWS, lightWS)
                           : 1.0;
        direct += arcForwardEvaluateLight(surface, viewWS, lightWS, radiance, shadow);
    }

    [loop]
    for (uint index = 0u; index < min(lighting.pointCount, 64u); ++index)
    {
        ArcForwardPointLight light = lighting.pointLights[index];
        float3 toLight = light.positionRange.xyz - input.positionWS;
        float distanceSquared = max(dot(toLight, toLight), 1e-4);
        float distanceToLight = sqrt(distanceSquared);
        float normalizedRange = saturate(distanceToLight / max(light.positionRange.w, 1e-4));
        float cutoff = 1.0 - pow(normalizedRange, 4.0);
        float3 radiance = light.colorIntensity.rgb * light.colorIntensity.w * cutoff * cutoff / distanceSquared;
        direct += arcForwardEvaluateLight(surface, viewWS, toLight / distanceToLight, radiance,
                                          arcForwardPointShadow(light, input.positionWS));
    }

    [loop]
    for (uint index = 0u; index < min(lighting.spotCount, 64u); ++index)
    {
        ArcForwardSpotLight light = lighting.spotLights[index];
        float3 toLight = light.positionRange.xyz - input.positionWS;
        float distanceSquared = max(dot(toLight, toLight), 1e-4);
        float distanceToLight = sqrt(distanceSquared);
        float3 lightWS = toLight / distanceToLight;
        float normalizedRange = saturate(distanceToLight / max(light.positionRange.w, 1e-4));
        float cutoff = 1.0 - pow(normalizedRange, 4.0);
        float cone = smoothstep(cos(light.params.x), cos(light.directionInnerAngle.w),
                                dot(-lightWS, normalize(light.directionInnerAngle.xyz)));
        float3 radiance = light.colorIntensity.rgb * light.colorIntensity.w *
                          cutoff * cutoff * cone / distanceSquared;
        direct += arcForwardEvaluateLight(surface, viewWS, lightWS, radiance,
                                          arcForwardSpotShadow(light, input.positionWS));
    }

    [loop]
    for (uint index = 0u; index < min(lighting.areaCount, 32u); ++index)
    {
        ArcForwardAreaLight light = lighting.areaLights[index];
        float3 toLight = light.positionShape.xyz - input.positionWS;
        float distanceSquared = max(dot(toLight, toLight), 1e-4);
        float distanceToLight = sqrt(distanceSquared);
        float3 lightWS = toLight / distanceToLight;
        float facing = dot(normalize(light.directionTwoSided.xyz), -lightWS);
        facing = light.directionTwoSided.w > 0.5 ? abs(facing) : max(facing, 0.0);
        float width = max(light.tangentWidth.w, 1e-4);
        float height = max(light.dimensionsShadow.y, 1e-4);
        float area = light.positionShape.w > 0.5 ? ARC_FORWARD_PI * width * height * 0.25 : width * height;
        float3 radiance = light.colorIntensity.rgb * light.colorIntensity.w *
                          min(area * facing / distanceSquared, 2.0 * ARC_FORWARD_PI);
        direct += arcForwardEvaluateLight(surface, viewWS, lightWS, radiance, 1.0);
    }

    float roughness = clamp(surface.roughness, 0.04, 1.0);
    float metallic = saturate(surface.metallic);
    float transmission = saturate(surface.transmission) * (1.0 - metallic);
    float nDotV = saturate(dot(normalize(surface.normalWS), viewWS));
    float3 dielectricF0 = arcForwardF0FromIor(max(surface.indexOfRefraction, 1.0001));
    float3 f0 = lerp(dielectricF0, surface.baseColor, metallic);
    float3 fresnel = arcForwardFresnelSchlick(nDotV, f0);
    float3 ambientRadiance = lighting.ambientColorIntensity.rgb * lighting.ambientColorIntensity.w;
    float diffuseWeight = (1.0 - metallic) * (1.0 - transmission);
    float3 ambient = surface.baseColor * ambientRadiance * diffuseWeight * surface.ambientOcclusion;
    float3 reflection = ambientRadiance * fresnel * lerp(1.0, 0.35, roughness);
    float clearCoat = saturate(surface.clearCoat);
    float coatRoughness = clamp(surface.clearCoatRoughness, 0.04, 1.0);
    float3 coatFresnel = arcForwardFresnelSchlick(nDotV, float3(0.04));
    float3 coatReflection = ambientRadiance * coatFresnel * clearCoat * lerp(1.0, 0.35, coatRoughness);
    float3 coatBaseEnergy = 1.0 - clearCoat * coatFresnel;
    ambient *= coatBaseEnergy;
    reflection = reflection * coatBaseEnergy + coatReflection;

    float2 viewportSize = max(
        float2(arcForwardScene.cameraPositionViewportWidth.w, arcForwardScene.fogParamsViewportHeight.w),
        float2(1.0));
    float2 screenUv = passInput.clipPosition.xy / max(abs(passInput.clipPosition.w), 1e-5) * float2(0.5, -0.5) + 0.5;
    float eta = 1.0 / max(surface.indexOfRefraction, 1.0001);
    float2 refractOffset = normalize(surface.normalWS).xz * (1.0 - eta) *
                           max(surface.thickness, 0.02) / viewportSize;
    float3 sceneTransmission = arcForwardSceneColor.SampleLevel(
        arcForwardSceneColorSampler, saturate(screenUv + refractOffset), 0.0).rgb;
    float3 attenuation = arcForwardBeerLambert(surface.attenuationColor, max(surface.thickness, 0.0),
                                               surface.attenuationDistance);
    float3 transmitted = sceneTransmission * attenuation * (1.0 - fresnel) * transmission;

    float3 color = direct + ambient + reflection + transmitted + surface.emissiveRadiance;
    float density = arcForwardScene.fogColorDensity.w;
    if (density > 0.0)
    {
        float distanceFromCamera = length(arcForwardScene.cameraPositionViewportWidth.xyz - input.positionWS);
        float startDistance = max(arcForwardScene.fogParamsViewportHeight.x, 0.0);
        float heightFalloff = max(arcForwardScene.fogParamsViewportHeight.y, 0.0);
        float maxOpacity = saturate(arcForwardScene.fogParamsViewportHeight.z);
        float distanceTerm = max(distanceFromCamera - startDistance, 0.0) * density;
        float heightTerm = exp(-max(input.positionWS.y, 0.0) * heightFalloff);
        float fogAmount = clamp(1.0 - exp(-distanceTerm * heightTerm), 0.0, maxOpacity);
        color = lerp(color, arcForwardScene.fogColorDensity.rgb, fogAmount);
    }
    return color;
}
)";
}

void append_forward(std::ostringstream& source, const material_descriptor& material)
{
    append_forward_lighting_library(source);
    source << "[shader(\"fragment\")] float4 main(ArcMaterialPassInput passInput) : SV_Target0\n"
              "{\n";
    append_surface_evaluation(source, material.alpha_mode);
    source << "    ArcSurfaceInput surfaceInput = arcMakeMaterialSurfaceInput(passInput);\n"
              "    float3 color = arcEvaluateForwardSurface(surface, surfaceInput, passInput);\n"
              "    return float4(color, surface.opacity);\n"
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

shader_compile_error custom_shader_error(std::string_view source_path, std::string message)
{
    return {.code = shader_compile_error_code::validation_failed,
            .source_path = std::string(source_path),
            .message = std::move(message)};
}

} // namespace

material_evaluator_result make_graph_material_evaluator(const material_graph_compilation& compilation)
{
    auto generated = generate_material_slang(compilation);
    if (!generated) return material_evaluator_result::failure(generated.error());

    auto evaluator = std::move(generated).value();
    const auto marker = evaluator.source.find(standalone_harness_marker);
    if (marker == std::string::npos)
        return material_evaluator_result::failure(
            {.code = shader_compile_error_code::validation_failed,
             .message = "generated material evaluator is missing the Stage 7 standalone harness boundary"});
    evaluator.source.resize(marker);
    return material_evaluator_result::success({.source = std::move(evaluator.source),
                                               .generated_line_nodes = std::move(evaluator.generated_line_nodes),
                                               .parameters = std::move(evaluator.parameters),
                                               .diagnostics = std::move(evaluator.diagnostics)});
}

material_evaluator_result make_custom_material_evaluator(std::string_view source, std::string_view source_path)
{
    if (source.empty())
        return material_evaluator_result::failure(custom_shader_error(source_path, "Material Shader source is empty"));
    if (source.find("arc_evaluate_material") == std::string_view::npos)
        return material_evaluator_result::failure(custom_shader_error(
            source_path, "Material Shader must implement ArcSurfaceData arc_evaluate_material(ArcSurfaceInput input)"));
    if (source.find("[shader(") != std::string_view::npos)
        return material_evaluator_result::failure(custom_shader_error(
            source_path, "Material Shader must not declare render-pass entry points; ARC owns all material passes"));

    std::string evaluator;
    evaluator.reserve(custom_material_abi.size() + source.size() + source_path.size() + 96u);
    evaluator.append(custom_material_abi);
    if (!source_path.empty())
    {
        evaluator.append("// ARC handwritten Material Shader: ");
        evaluator.append(source_path);
        evaluator.push_back('\n');
    }
    evaluator.append(source);
    if (!evaluator.ends_with('\n')) evaluator.push_back('\n');
    return material_evaluator_result::success({.source = std::move(evaluator), .handwritten = true});
}

material_pass_codegen_result generate_material_pass_slang(const material_evaluator_source& evaluator,
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

    std::ostringstream pass_source;
    pass_source << evaluator.source;
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
                                                  .generated_line_nodes = evaluator.generated_line_nodes,
                                                  .parameters = evaluator.parameters,
                                                  .diagnostics = evaluator.diagnostics});
}

material_pass_codegen_result generate_material_pass_slang(const material_graph_compilation& compilation,
                                                          const material_descriptor& material, material_pass pass,
                                                          std::uint8_t debug_view, bool wireframe)
{
    auto evaluator = make_graph_material_evaluator(compilation);
    if (!evaluator) return material_pass_codegen_result::failure(evaluator.error());
    return generate_material_pass_slang(evaluator.value(), material, pass, debug_view, wireframe);
}

} // namespace arc::render::tools
