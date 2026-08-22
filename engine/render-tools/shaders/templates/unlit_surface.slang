import arc.surface;

struct MaterialParameters { float4 color; };
ParameterBlock<MaterialParameters> material;

ArcSurfaceData arcEvaluateSurface(ArcSurfaceInput input)
{
    ArcSurfaceData surface = arcDefaultSurface(input.normalWS);
    surface.baseColor = float3(0.0);
    surface.emissiveRadiance = material.color.rgb;
    surface.opacity = material.color.a;
    return surface;
}
