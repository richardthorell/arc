import arc.surface;

struct HologramParameters
{
    float3 color;
    float intensity;
    float scanFrequency;
};

ParameterBlock<HologramParameters> material;

ArcSurfaceData arcEvaluateSurface(ArcSurfaceInput input)
{
    ArcSurfaceData surface = arcDefaultSurface(input.normalWS);
    float scan = 0.5 + 0.5 * sin(input.positionWS.y * material.scanFrequency);
    surface.baseColor = float3(0.0);
    surface.emissiveRadiance = material.color * material.intensity * scan;
    surface.opacity = scan;
    return surface;
}
