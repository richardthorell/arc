// Engine-owned set 2 adapter for the shared GLSL/Slang VSM lookup.
struct ArcVirtualShadowAddressSpace { uint4 identityTopology; uint4 ranges; uint4 requestMetadata; };
struct ArcVirtualShadowView { float worldToShadowClip[16]; float4 snappedOriginWorldUnits; uint4 pageRange; };
struct ArcVirtualShadowPhysicalMapping { uint4 value; };
struct ArcVirtualShadowPageTableEntry
{
    ArcVirtualShadowPhysicalMapping staticDepth;
    ArcVirtualShadowPhysicalMapping dynamicDepth;
};
StructuredBuffer<ArcVirtualShadowAddressSpace> arcVirtualShadowAddresses : register(t10, space2);
StructuredBuffer<ArcVirtualShadowView> arcVirtualShadowViews : register(t11, space2);
StructuredBuffer<ArcVirtualShadowPageTableEntry> arcVirtualShadowPages : register(t12, space2);
Texture2D<float> arcVirtualShadowStaticAtlas : register(t13, space2);
Texture2D<float> arcVirtualShadowDynamicAtlas : register(t14, space2);
SamplerComparisonState arcVirtualShadowSampler : register(s15, space2);
static const uint ARC_VIRTUAL_SHADOW_INVALID_INDEX = 0xffffffffu;
static const uint ARC_VIRTUAL_SHADOW_PAGE_TEXELS = 128u;
static const uint ARC_VIRTUAL_SHADOW_PAGE_GUARD_TEXELS = 4u;
uint arcVirtualShadowLevelCount(ArcVirtualShadowAddressSpace a) { return a.identityTopology.w & 0xffffu; }
uint arcVirtualShadowViewIndex(ArcVirtualShadowAddressSpace a, uint face, uint level)
{
    return a.ranges.x + face * arcVirtualShadowLevelCount(a) + level;
}
uint arcVirtualShadowDensePageIndex(ArcVirtualShadowAddressSpace a, ArcVirtualShadowView v, uint2 page)
{
    return a.ranges.z + v.pageRange.x + page.y * v.pageRange.y + page.x;
}
float4 arcVirtualShadowTransform(ArcVirtualShadowView v, float3 position)
{
    float4 p = float4(position, 1.0);
    return float4(
        dot(float4(v.worldToShadowClip[0], v.worldToShadowClip[1], v.worldToShadowClip[2], v.worldToShadowClip[3]), p),
        dot(float4(v.worldToShadowClip[4], v.worldToShadowClip[5], v.worldToShadowClip[6], v.worldToShadowClip[7]), p),
        dot(float4(v.worldToShadowClip[8], v.worldToShadowClip[9], v.worldToShadowClip[10], v.worldToShadowClip[11]), p),
        dot(float4(v.worldToShadowClip[12], v.worldToShadowClip[13], v.worldToShadowClip[14], v.worldToShadowClip[15]), p));
}
int2 arc_virtual_shadow_atlas_extent()
{
    uint width, height;
    arcVirtualShadowStaticAtlas.GetDimensions(width, height);
    return int2(width, height);
}
uint arc_virtual_shadow_address_count()
{
    uint count, stride;
    arcVirtualShadowAddresses.GetDimensions(count, stride);
    return count;
}
uint arc_virtual_shadow_view_count()
{
    uint count, stride;
    arcVirtualShadowViews.GetDimensions(count, stride);
    return count;
}
uint arc_virtual_shadow_page_count()
{
    uint count, stride;
    arcVirtualShadowPages.GetDimensions(count, stride);
    return count;
}
float arc_virtual_shadow_compare(bool dynamicLayer, float2 uv, float depth)
{
    return dynamicLayer
        ? arcVirtualShadowDynamicAtlas.SampleCmpLevelZero(arcVirtualShadowSampler, uv, depth)
        : arcVirtualShadowStaticAtlas.SampleCmpLevelZero(arcVirtualShadowSampler, uv, depth);
}
// Scoped aliases; the composer undefines these after the shared lookup.
#define vec2 float2
#define vec3 float3
#define vec4 float4
#define uvec2 uint2
#define ivec2 int2
#define mix lerp
#define fract frac
#define lessThan(a, b) ((a) < (b))
#define greaterThanEqual(a, b) ((a) >= (b))
#define arc_virtual_shadow_addresses arcVirtualShadowAddresses
#define arc_virtual_shadow_views arcVirtualShadowViews
#define arc_virtual_shadow_page_table arcVirtualShadowPages
#define directional_light_data ArcForwardDirectionalLight
#define shadow_identity shadowIdentity
#define shadow_parameters shadowParameters
#define shadow_routing shadowRouting
