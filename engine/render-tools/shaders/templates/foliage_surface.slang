import arc.surface;

struct FoliageParameters
{
    float3 baseColor;
    float roughness;
    float alphaCutoff;
    float translucency;
};

ParameterBlock<FoliageParameters> material;

ArcSurfaceData arcEvaluateSurface(ArcSurfaceInput input)
{
    ArcSurfaceData surface = arcDefaultSurface(input.normalWS);
    surface.baseColor = material.baseColor;
    surface.roughness = material.roughness;
    surface.alphaCutoff = material.alphaCutoff;
    surface.subsurfaceColor = material.baseColor;
    surface.subsurface = material.translucency;
    return surface;
}
