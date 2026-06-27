#pragma once

#include <arc/simd/core/ops/arithmetic.h>
#include <arc/simd/core/ops/bitwise.h>
#include <arc/simd/core/ops/conversions.h>
#include <arc/simd/core/ops/memory.h>

#include <cstdint>
#include <type_traits>

namespace arc
{

template <class T, std::size_t N>
constexpr simd<T, N> andnot(const simd<T, N>& a, const simd<T, N>& b) noexcept
{
    return bitwise_and(bitwise_not(a), b);
}

template <class T, std::size_t N>
constexpr simd<T, N> fmsub(const simd<T, N>& a, const simd<T, N>& b, const simd<T, N>& c) noexcept
{
    return a * b - c;
}

template <class T, std::size_t N>
constexpr simd<T, N> fnma(const simd<T, N>& a, const simd<T, N>& b, const simd<T, N>& c) noexcept
{
    return c - a * b;
}

template <std::size_t N>
constexpr simd<float, N> xor_sign(const simd<float, N>& value, const simd<float, N>& sign_source) noexcept
{
    const auto sign_mask = fill<uint32_t, N>(0x80000000u);
    auto bits = bit_cast<uint32_t>(value);
    auto sign = bitwise_and(bit_cast<uint32_t>(sign_source), sign_mask);
    return bit_cast<float>(bitwise_xor(bits, sign));
}

template <std::size_t N>
constexpr simd<double, N> xor_sign(const simd<double, N>& value, const simd<double, N>& sign_source) noexcept
{
    const auto sign_mask = fill<uint64_t, N>(0x8000000000000000ull);
    auto bits = bit_cast<uint64_t>(value);
    auto sign = bitwise_and(bit_cast<uint64_t>(sign_source), sign_mask);
    return bit_cast<double>(bitwise_xor(bits, sign));
}

} // namespace arc
