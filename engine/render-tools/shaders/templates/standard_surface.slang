import arc.surface;

struct MaterialParameters
{
    float4 baseColor;
    float metallic;
    float roughness;
};
ParameterBlock<MaterialParameters> material;

ArcSurfaceData arcEvaluateSurface(ArcSurfaceInput input)
{
    ArcSurfaceData surface = arcDefaultSurface(input.normalWS);
    surface.baseColor = material.baseColor.rgb;
    surface.opacity = material.baseColor.a;
    surface.metallic = material.metallic;
    surface.roughness = material.roughness;
    return surface;
}
