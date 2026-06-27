#pragma once

#include <arc/simd/core/ops/lanes.h>

namespace arc
{

template <std::size_t... I, class T, std::size_t N>
constexpr simd<T, N> permute(const simd<T, N>& value) noexcept
{
    static_assert(sizeof...(I) == N, "permute must specify N indices");
    static_assert(((I < N) && ...), "permute index out of bounds");

    return simd<T, N>(extract<I>(value)...);
}

template <class T, std::size_t N>
constexpr simd<T, N> reverse(const simd<T, N>& value) noexcept
{
    return [&]<std::size_t... I>(std::index_sequence<I...>)
    {
        return permute<N - 1 - I...>(value);
    }(std::make_index_sequence<N>{});
}

template <std::size_t Offset, class T, std::size_t N>
constexpr simd<T, N> rotate_left(const simd<T, N>& value) noexcept
{
    static_assert(Offset < N, "rotate_left offset must be less than N");

    return [&]<std::size_t... I>(std::index_sequence<I...>)
    {
        return permute<(I + Offset) % N...>(value);
    }(std::make_index_sequence<N>{});
}

template <std::size_t Offset, class T, std::size_t N>
constexpr simd<T, N> rotate_right(const simd<T, N>& value) noexcept
{
    static_assert(Offset < N, "rotate_right offset must be less than N");

    return [&]<std::size_t... I>(std::index_sequence<I...>)
    {
        return permute<(I + N - Offset) % N...>(value);
    }(std::make_index_sequence<N>{});
}

template <std::size_t... I, class T, std::size_t N>
constexpr simd<T, N> shuffle(const simd<T, N>& a, const simd<T, N>& b) noexcept
{
    static_assert(sizeof...(I) == N, "shuffle must specify N indices");
    static_assert(((I < 2 * N) && ...), "shuffle index out of bounds");

    return simd<T, N>(extract<I < N ? I : I - N>(I < N ? a : b)...);
}

} // namespace arc