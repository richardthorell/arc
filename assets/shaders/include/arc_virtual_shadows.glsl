#ifndef ARC_VIRTUAL_SHADOWS_GLSL
#define ARC_VIRTUAL_SHADOWS_GLSL

const uint ARC_VIRTUAL_SHADOW_INVALID_INDEX = 0xffffffffu;
const uint ARC_VIRTUAL_SHADOW_PAGE_TEXELS = 128u;
const uint ARC_VIRTUAL_SHADOW_PAGE_GUARD_TEXELS = 4u;

struct ArcVirtualShadowAddressSpace
{
    uvec4 identityTopology; // generation, kind, virtual resolution, packed levels/faces
    uvec4 ranges; // view base/count, page-table base/count
};

struct ArcVirtualShadowView
{
    float worldToShadowClip[16]; // Row-major, matching the native ABI.
    vec4 snappedOriginWorldUnits;
    uvec4 pageRange; // level-relative table offset, pages/axis, face, level
};

struct ArcVirtualShadowPhysicalMapping
{
    uvec4 value; // physical index, generation, content revision low/high
};

struct ArcVirtualShadowPageTableEntry
{
    ArcVirtualShadowPhysicalMapping staticDepth;
    ArcVirtualShadowPhysicalMapping dynamicDepth;
};

uint arcVirtualShadowLevelCount(ArcVirtualShadowAddressSpace addressSpace)
{
    return addressSpace.identityTopology.w & 0xffffu;
}

uint arcVirtualShadowFaceCount(ArcVirtualShadowAddressSpace addressSpace)
{
    return addressSpace.identityTopology.w >> 16u;
}

uint arcVirtualShadowViewIndex(ArcVirtualShadowAddressSpace addressSpace, uint face, uint level)
{
    return addressSpace.ranges.x + face * arcVirtualShadowLevelCount(addressSpace) + level;
}

uint arcVirtualShadowDensePageIndex(ArcVirtualShadowAddressSpace addressSpace, ArcVirtualShadowView view,
                                    uvec2 page)
{
    return addressSpace.ranges.z + view.pageRange.x + page.y * view.pageRange.y + page.x;
}

vec4 arcVirtualShadowTransform(ArcVirtualShadowView view, vec3 worldPosition)
{
    vec4 position = vec4(worldPosition, 1.0);
    return vec4(dot(vec4(view.worldToShadowClip[0], view.worldToShadowClip[1], view.worldToShadowClip[2],
                         view.worldToShadowClip[3]),
                    position),
                dot(vec4(view.worldToShadowClip[4], view.worldToShadowClip[5], view.worldToShadowClip[6],
                         view.worldToShadowClip[7]),
                    position),
                dot(vec4(view.worldToShadowClip[8], view.worldToShadowClip[9], view.worldToShadowClip[10],
                         view.worldToShadowClip[11]),
                    position),
                dot(vec4(view.worldToShadowClip[12], view.worldToShadowClip[13], view.worldToShadowClip[14],
                         view.worldToShadowClip[15]),
                    position));
}

#endif
