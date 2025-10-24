#pragma once

#if defined(__ARM_NEON) || defined(__ARM_NEON__)
    #define ARC_SIMD_NEON
#endif

#if defined(__ARM_NEON_FP) && (__ARM_NEON_FP & 0x4) != 0
    #define ARC_SIMD_NEON_FP64
#endif
