// ARC shader-library v1. Tools-only source; cooked runtimes consume reflected packages.
module arc.surface;

public static const uint ARC_MATERIAL_ABI_VERSION = 1;

public struct ArcSurfaceInput
{
    float3 positionWS;
    float3 normalWS;
    float4 tangentWS;
    float2 uv0;
    float2 uv1;
    float4 vertexColor;
    float3 viewWS;
};

public struct ArcSurfaceData
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

public ArcSurfaceData arcDefaultSurface(float3 normalWS)
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
