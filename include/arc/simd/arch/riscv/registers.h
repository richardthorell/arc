#pragma once

#include "arc/simd/arch/riscv/detect.h"

#include <cstddef>
#include <cstdint>

namespace arc
{

#if defined(ARC_SIMD_RISCV)

template <class T, std::size_t N>
struct riscv_vec
{
    alignas(sizeof(T) * N) T lanes[N];
};

template <class T, std::size_t N>
struct simd_register
{
    using type = riscv_vec<T, N>;
};

template <>
inline constexpr std::size_t simd_max_lanes<float> = 4;

template <>
inline constexpr std::size_t simd_max_lanes<double> = 2;

template <>
inline constexpr std::size_t simd_max_lanes<int8_t> = 16;

template <>
inline constexpr std::size_t simd_max_lanes<uint8_t> = 16;

template <>
inline constexpr std::size_t simd_max_lanes<int16_t> = 8;

template <>
inline constexpr std::size_t simd_max_lanes<uint16_t> = 8;

template <>
inline constexpr std::size_t simd_max_lanes<int32_t> = 4;

template <>
inline constexpr std::size_t simd_max_lanes<uint32_t> = 4;

template <>
inline constexpr std::size_t simd_max_lanes<int64_t> = 2;

template <>
inline constexpr std::size_t simd_max_lanes<uint64_t> = 2;

#endif

} // namespace arc
