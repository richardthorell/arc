Texture2D<float4> arcInputColor;
SamplerState arcLinearClamp;

[shader("fragment")]
float4 arcPostProcess(float2 uv : TEXCOORD0) : SV_Target0
{
    return arcInputColor.Sample(arcLinearClamp, uv);
}
