#pragma once

#include "arc/simd/arch/x64/detect.h"

namespace arc
{

#if defined(ARC_SIMD_SSE)
template <>
struct simd_register<float, 4>
{
    using type = __m128;
};


template <>
struct simd_register<int32_t, 4>
{
    using type = __m128i;
};


template <>
struct simd_register<uint32_t, 4>
{
    using type = __m128i;
};


template <>
struct simd_register<int64_t, 2>
{
    using type = __m128i;
};


template <>
struct simd_register<uint64_t, 2>
{
    using type = __m128i;
};


template <>
struct simd_register<double, 2>
{
    using type = __m128d;
};
#endif


#if defined(ARC_SIMD_AVX)
template <>
struct simd_register<float, 8>
{
    using type = __m256;
};


template <>
struct simd_register<double, 4>
{
    using type = __m256d;
};


template <>
struct simd_register<int32_t, 8>
{
    using type = __m256i;
};


template <>
struct simd_register<uint32_t, 8>
{
    using type = __m256i;
};


template <>
struct simd_register<int64_t, 4>
{
    using type = __m256i;
};


template <>
struct simd_register<uint64_t, 4>
{
    using type = __m256i;
};
#endif


#if defined(ARC_SIMD_AVX512)
template <>
struct simd_register<float, 16>
{
    using type = __m512;
};


template <>
struct simd_register<double, 8>
{
    using type = __m512d;
};


template <>
struct simd_register<int32_t, 16>
{
    using type = __m512i;
};


template <>
struct simd_register<uint32_t, 16>
{
    using type = __m512i;
};


template <>
struct simd_register<int64_t, 8>
{ 
    using type = __m512i;
};


template <>
struct simd_register<uint64_t, 8>
{
    using type = __m512i;
};
#endif

} // namespace arc