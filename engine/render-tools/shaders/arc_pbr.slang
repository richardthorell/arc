// ARC shader-library v1: backend-neutral production BRDF building blocks.
module arc.pbr;

static const float ARC_PI = 3.14159265358979323846;

public float3 arcF0FromIor(float ior)
{
    float ratio = (ior - 1.0) / max(ior + 1.0, 1e-4);
    return float3(ratio * ratio);
}

public float3 arcFresnelSchlick(float cosTheta, float3 f0)
{
    float factor = pow(saturate(1.0 - cosTheta), 5.0);
    return f0 + (1.0 - f0) * factor;
}

public float arcD_GGX(float nDotH, float roughness)
{
    float alpha = max(roughness * roughness, 0.002);
    float alpha2 = alpha * alpha;
    float denominator = nDotH * nDotH * (alpha2 - 1.0) + 1.0;
    return alpha2 / max(ARC_PI * denominator * denominator, 1e-6);
}

public float arcV_SmithGGXCorrelated(float nDotV, float nDotL, float roughness)
{
    float alpha2 = pow(max(roughness, 0.02), 4.0);
    float gv = nDotL * sqrt(max(nDotV * nDotV * (1.0 - alpha2) + alpha2, 1e-6));
    float gl = nDotV * sqrt(max(nDotL * nDotL * (1.0 - alpha2) + alpha2, 1e-6));
    return 0.5 / max(gv + gl, 1e-5);
}

public float arcBurleyDiffuse(float nDotV, float nDotL, float lDotH, float roughness)
{
    float energyBias = lerp(0.0, 0.5, roughness);
    float energyFactor = lerp(1.0, 1.0 / 1.51, roughness);
    float fd90 = energyBias + 2.0 * lDotH * lDotH * roughness;
    float lightScatter = 1.0 + (fd90 - 1.0) * pow(1.0 - nDotL, 5.0);
    float viewScatter = 1.0 + (fd90 - 1.0) * pow(1.0 - nDotV, 5.0);
    return lightScatter * viewScatter * energyFactor / ARC_PI;
}

public float arcD_Charlie(float nDotH, float roughness)
{
    float inverseRoughness = 1.0 / max(roughness, 0.02);
    float sin2h = max(1.0 - nDotH * nDotH, 0.0);
    return (2.0 + inverseRoughness) * pow(sin2h, inverseRoughness * 0.5) / (2.0 * ARC_PI);
}

public float3 arcBeerLambert(float3 attenuationColor, float distance, float attenuationDistance)
{
    if (attenuationDistance <= 0.0) return float3(1.0);
    float3 coefficient = -log(max(attenuationColor, float3(1e-4))) / attenuationDistance;
    return exp(-coefficient * max(distance, 0.0));
}

public float3 arcMultiscatterCompensation(float3 f0, float roughness)
{
    float averageFresnel = saturate((f0.x + f0.y + f0.z) / 3.0);
    float lostEnergy = roughness * (1.0 - averageFresnel) * 0.28;
    return 1.0 + f0 * lostEnergy / max(1.0 - lostEnergy, 1e-4);
}
