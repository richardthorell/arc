#pragma once

#include "arc/simd/arch/x64/detect.h"

#include <arm_neon.h>

namespace arc
{

#if defined(ARC_SIMD_NEON)
template <>
struct simd_register<float, 4>
{
    using type = float32x4_t;
};

template <>
struct simd_register<int32_t, 4>
{
    using type = int32x4_t;
};

template <>
struct simd_register<uint32_t, 4>
{
    using type = uint32x4_t;
};

template <>
struct simd_register<int16_t, 8>
{
    using type = int16x8_t;
};

template <>
struct simd_register<uint16_t, 8>
{
    using type = uint16x8_t;
};

template <>
struct simd_register<int8_t, 16>
{
    using type = int8x16_t;
};

template <>
struct simd_register<uint8_t, 16>
{
    using type = uint8x16_t;
};

template <>
struct simd_register<int64_t, 2>
{
    using type = int64x2_t;
};

template <>
struct simd_register<uint64_t, 2>
{
    using type = uint64x2_t;
};

template <>
inline constexpr std::size_t simd_max_lanes<float> = 4;

template <>
inline constexpr std::size_t simd_max_lanes<int32_t> = 4;

template <>
inline constexpr std::size_t simd_max_lanes<uint32_t> = 4;

template <>
inline constexpr std::size_t simd_max_lanes<int64_t> = 2;

template <>
inline constexpr std::size_t simd_max_lanes<uint64_t> = 2;

template <>
inline constexpr std::size_t simd_max_lanes<int16_t> = 8;

template <>
inline constexpr std::size_t simd_max_lanes<uint16_t> = 8;

template <>
inline constexpr std::size_t simd_max_lanes<int8_t> = 16;

template <>
inline constexpr std::size_t simd_max_lanes<uint8_t> = 16;
#endif


#if defined(ARC_SIMD_NEON_FP64)
template <>
struct simd_register<double, 2>
{
    using type = float64x2_t;
};

template <>
inline constexpr std::size_t simd_max_lanes<double> = 2;
#endif

} // namespace arc
