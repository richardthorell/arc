#pragma once

#include <arc/simd/core/ops/lanes.h>
#include <arc/simd/core/ops/memory.h>

#include <array>
#include <utility>

namespace arc::detail
{

template <class T, std::size_t N, std::size_t... I>
constexpr std::array<T, N> simd_to_array_impl(const simd<T, N>& value, std::index_sequence<I...>) noexcept
{
    return { extract<I>(value)... };
}

template <class T, std::size_t N>
constexpr std::array<T, N> simd_to_array(const simd<T, N>& value) noexcept
{
    return simd_to_array_impl(value, std::make_index_sequence<N>{});
}

template <class T, std::size_t N, std::size_t... I>
constexpr simd<T, N> simd_from_array_impl(const std::array<T, N>& values, std::index_sequence<I...>) noexcept
{
    auto result = fill<T, N>(T{});
    ((result = insert<I>(result, values[I])), ...);
    return result;
}

template <class T, std::size_t N>
constexpr simd<T, N> simd_from_array(const std::array<T, N>& values) noexcept
{
    return simd_from_array_impl<T, N>(values, std::make_index_sequence<N>{});
}

template <class T, std::size_t N, class Op>
constexpr simd<T, N> simd_map(const simd<T, N>& value, Op op) noexcept
{
    auto values = simd_to_array(value);
    for (auto& lane : values)
        lane = static_cast<T>(op(lane));
    return simd_from_array<T, N>(values);
}

template <class T, std::size_t N, class Op>
constexpr simd<T, N> simd_map(const simd<T, N>& a, const simd<T, N>& b, Op op) noexcept
{
    auto av = simd_to_array(a);
    auto bv = simd_to_array(b);
    std::array<T, N> result{};
    for (std::size_t i = 0; i < N; ++i)
        result[i] = static_cast<T>(op(av[i], bv[i]));
    return simd_from_array<T, N>(result);
}

} // namespace arc::detail
