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


template <>
inline constexpr std::size_t simd_max_lanes<float> =
#if defined(ARC_SIMD_AVX512)
    16;
#elif defined(ARC_SIMD_AVX)
    8;
#elif defined(ARC_SIMD_SSE)
    4;
#else
    1;
#endif

template <>
inline constexpr std::size_t simd_max_lanes<double> =
#if defined(ARC_SIMD_AVX512)
    8;
#elif defined(ARC_SIMD_AVX)
    4;
#elif defined(ARC_SIMD_SSE)
    2;
#else
    1;
#endif

template <>
inline constexpr std::size_t simd_max_lanes<int32_t> =
#if defined(ARC_SIMD_AVX512)
    16;
#elif defined(ARC_SIMD_AVX)
    8;
#elif defined(ARC_SIMD_SSE)
    4;
#else
    1;
#endif

template <>
inline constexpr std::size_t simd_max_lanes<uint32_t> =
#if defined(ARC_SIMD_AVX512)
    16;
#elif defined(ARC_SIMD_AVX)
    8;
#elif defined(ARC_SIMD_SSE)
    4;
#else
    1;
#endif

template <>
inline constexpr std::size_t simd_max_lanes<int64_t> =
#if defined(ARC_SIMD_AVX512)
    8;
#elif defined(ARC_SIMD_AVX)
    4;
#elif defined(ARC_SIMD_SSE)
    2;
#else
    1;
#endif

template <>
inline constexpr std::size_t simd_max_lanes<uint64_t> =
#if defined(ARC_SIMD_AVX512)
    8;
#elif defined(ARC_SIMD_AVX)
    4;
#elif defined(ARC_SIMD_SSE)
    2;
#else
    1;
#endif

} // namespace arc