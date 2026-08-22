import arc.surface;

struct WaterParameters
{
    float3 attenuationColor;
    float roughness;
    float opacity;
    float thickness;
    float indexOfRefraction;
};

ParameterBlock<WaterParameters> material;

ArcSurfaceData arcEvaluateSurface(ArcSurfaceInput input)
{
    ArcSurfaceData surface = arcDefaultSurface(input.normalWS);
    surface.baseColor = material.attenuationColor;
    surface.roughness = material.roughness;
    surface.opacity = material.opacity;
    surface.transmission = 1.0;
    surface.thickness = material.thickness;
    surface.indexOfRefraction = material.indexOfRefraction;
    surface.attenuationColor = material.attenuationColor;
    return surface;
}
