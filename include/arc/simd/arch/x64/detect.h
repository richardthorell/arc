#pragma once

#include <immintrin.h>

#if defined(_MSC_VER)
    #if defined(__AVX512F__)
        #define ARC_SIMD_AVX512
    #endif
    #if defined(__AVX2__)
        #define ARC_SIMD_AVX2
    #endif
    #if defined(__AVX__)
        #define ARC_SIMD_AVX
    #endif
    #if defined(_M_X64)
        #define ARC_SIMD_SSE
        #define ARC_SIMD_SSE2
    #elif defined(_M_IX86_FP)
        #if _M_IX86_FP == 2
            #define ARC_SIMD_SSE2
            #define ARC_SIMD_SSE
        #elif _M_IX86_FP == 1
            #define ARC_SIMD_SSE
        #endif
    #endif
#else
    #if defined(__AVX512F__)
        #define ARC_SIMD_AVX512
    #endif
    #if defined(__AVX2__)
        #define ARC_SIMD_AVX2
    #endif
    #if defined(__AVX__)
        #define ARC_SIMD_AVX
    #endif
    #if defined(__SSE2__)
        #define ARC_SIMD_SSE2
    #endif
    #if defined(__SSE__)
        #define ARC_SIMD_SSE
    #endif
#endif
