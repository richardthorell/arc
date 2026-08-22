import arc.surface;

struct MaterialParameters { float4 color; float bands; };
ParameterBlock<MaterialParameters> material;

ArcSurfaceData arcEvaluateSurface(ArcSurfaceInput input)
{
    ArcSurfaceData surface = arcDefaultSurface(input.normalWS);
    surface.baseColor = material.color.rgb;
    surface.roughness = 1.0;
    return surface;
}
