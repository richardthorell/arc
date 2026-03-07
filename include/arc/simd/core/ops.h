#pragma once

#include <type_traits>

#include <arc/simd/core/simd.h>
#include <arc/simd/core/mask.h>

namespace arc
{

template <class T, std::size_t N>
constexpr simd<T, N> select(const simd_mask<N>&, const simd<T, N>&, const simd<T, N>&) noexcept;

template <class T, std::size_t N, std::size_t... Index, class Op>
constexpr auto apply(std::index_sequence<Index...>, Op op) noexcept
{
    if constexpr (std::is_void_v<decltype(op(Index...))>) {
        (op(Index), ...);
    } else {
        return simd<T, N>(op(Index)...);
    }
}

template <class T, std::size_t N, class Op>
constexpr auto apply(Op op) noexcept
{
    return apply<T, N>(std::make_index_sequence<simd<T, N>::blocks()>{}, op);
}

template <class T, std::size_t N, std::size_t... Index, class Op>
constexpr auto apply(const simd<T, N>& a, std::index_sequence<Index...>, Op op) noexcept
{
    if constexpr (std::is_void_v<decltype(op(a.data[0]))>) {
        (op(a.data[Index]), ...);
    } else {
        return simd<T, N>(op(a.data[Index])...);
    }
}

template <class T, std::size_t N, class Op>
constexpr auto apply(const simd<T, N>& a, Op op) noexcept
{
    return apply(a, std::make_index_sequence<simd<T, N>::blocks()>{}, op);
}

template <class T, std::size_t N, std::size_t... Index, class Op>
constexpr auto apply(const simd<T, N>& a, const simd<T, N>& b, std::index_sequence<Index...>, Op op) noexcept
{
    if constexpr (std::is_void_v<decltype(op(a.data[0], b.data[0]))>) {
        (op(a.data[Index], b.data[Index]), ...);
    } else {
        return simd<T, N>(op(a.data[Index], b.data[Index])...);
    }
}

template <class T, std::size_t N, class Op>
constexpr auto apply(const simd<T, N>& a, const simd<T, N>& b, Op op) noexcept
{
    return apply(a, b, std::make_index_sequence<simd<T, N>::blocks()>{}, op);
}

template <class T, std::size_t N, std::size_t... Index, class Op>
constexpr auto apply(const simd<T, N>& a, const simd<T, N>& b, const simd<T, N>& c, std::index_sequence<Index...>, Op op) noexcept
{
    if constexpr (std::is_void_v<decltype(op(a.data[0], b.data[0], c.data[0]))>) {
        (op(a.data[Index], b.data[Index], c.data[Index]), ...);
    } else {
        return simd<T, N>(op(a.data[Index], b.data[Index], c.data[Index])...);
    }
}

template <class T, std::size_t N, class Op>
constexpr auto apply(const simd<T, N>& a, const simd<T, N>& b, const simd<T, N>& c, Op op) noexcept
{
    return apply(a, b, c, std::make_index_sequence<simd<T, N>::blocks()>{}, op);
}

template <std::size_t N, std::size_t... Index, class Op>
constexpr auto apply(std::index_sequence<Index...>, Op op) noexcept
{
    if constexpr (std::is_void_v<decltype(op(Index...))>) {
        (op(Index), ...);
    } else {
        return simd_mask<N>{ op(Index)... };
    }
}

template <std::size_t N, class Op>
constexpr auto apply(Op op) noexcept
{
    return apply<N>(std::make_index_sequence<simd_mask<N>::blocks()>{}, op);
}

template <std::size_t N, std::size_t... Index, class Op>
constexpr auto apply(const simd_mask<N>& a, std::index_sequence<Index...>, Op op) noexcept
{
    if constexpr (std::is_void_v<decltype(op(a.data[0]))>) {
        (op(a.data[Index]), ...);
    } else {
        return simd_mask<N>{ op(a.data[Index])... };
    }
}

template <std::size_t N, class Op>
constexpr auto apply(const simd_mask<N>& a, Op op) noexcept
{
    return apply(a, std::make_index_sequence<simd_mask<N>::blocks()>{}, op);
}

template <std::size_t N, std::size_t... Index, class Op>
constexpr auto apply(const simd_mask<N>& a, const simd_mask<N>& b, std::index_sequence<Index...>, Op op) noexcept
{
    if constexpr (std::is_void_v<decltype(op(a.data[0], b.data[0]))>) {
        (op(a.data[Index], b.data[Index]), ...);
    } else {
        return simd_mask<N>{ op(a.data[Index], b.data[Index])... };
    }
}

template <std::size_t N, class Op>
constexpr auto apply(const simd_mask<N>& a, const simd_mask<N>& b, Op op) noexcept
{
    return apply(a, b, std::make_index_sequence<simd_mask<N>::blocks()>{}, op);
}

template <class T, std::size_t N, std::size_t... Index, class Op>
constexpr simd_mask<N> compare(const simd<T, N>& a, const simd<T, N>& b, std::index_sequence<Index...>, Op op) noexcept
{
    return simd_mask<N>{ op(a.data[Index], b.data[Index])... };
}

template <class T, std::size_t N, class Op>
constexpr simd_mask<N> compare(const simd<T, N>& a, const simd<T, N>& b, Op op) noexcept
{
    return compare(a, b, std::make_index_sequence<simd_block<T, N>::blocks>{}, op);
}

template <class T, std::size_t N, std::size_t... Index, class Op>
constexpr simd<T, N> masked(const simd_mask<N>& mask, const simd<T, N>& a, const simd<T, N>& b, std::index_sequence<Index...>, Op op) noexcept
{
    return simd<T, N>{ select(mask.data[Index], a.data[Index], op(a.data[Index], b.data[Index]))... };
}

template <class T, std::size_t N, class Op>
constexpr simd<T, N> masked(const simd_mask<N>& mask, const simd<T, N>& a, const simd<T, N>& b, Op op) noexcept
{
    return masked(mask, a, b, std::make_index_sequence<simd<T, N>::blocks()>{}, op);
}

template <class T, std::size_t N, std::size_t... Index, class Op>
constexpr simd<T, N> masked(const simd_mask<N>& mask, const simd<T, N>& a, std::index_sequence<Index...>, Op op) noexcept
{
    return simd<T, N>{ select(mask.data[Index], a.data[Index], op(a.data[Index]))... };
}

template <class T, std::size_t N, class Op>
constexpr simd<T, N> masked(const simd_mask<N>& mask, const simd<T, N>& a, Op op) noexcept
{
    return masked(mask, a, std::make_index_sequence<simd<T, N>::blocks()>{}, op);
}


// Load/Store operations
template <class T, std::size_t N>
constexpr simd<T, N> load(const T* ptr) noexcept
{
    return apply<T, N>(
        [&](std::size_t block_index)
        {
            return ops_for<simd<T, N>>::load(ptr + block_index * simd_block<T, N>::lanes);
        }
    );
}

template <class T, std::size_t N>
constexpr void store(T* ptr, const simd<T, N>& value) noexcept
{
    apply<T, N>(
        [&](std::size_t block_index)
        {
            ops_for<simd<T, N>>::store(ptr + block_index * simd_block<T, N>::lanes, value.data[block_index]);
        }
    );
}

template <class T, std::size_t N>
constexpr simd<T, N> fill(T value) noexcept
{
    return apply<T, N>(
        [&](std::size_t)
        {
            return ops_for<simd<T, N>>::fill(value);
        }
    );
}


// Arithmetic operations
template <class T, std::size_t N>
constexpr simd<T, N> add(const simd<T, N>& a, const simd<T, N>& b) noexcept
{
    return apply(a, b, ops_for<simd<T, N>>::add);
}

template <class T, std::size_t N>
constexpr simd<T, N> sub(const simd<T, N>& a, const simd<T, N>& b) noexcept
{
    return apply(a, b, ops_for<simd<T, N>>::sub);
}

template <class T, std::size_t N>
constexpr simd<T, N> mul(const simd<T, N>& a, const simd<T, N>& b) noexcept
{
    return apply(a, b, ops_for<simd<T, N>>::mul);
}

template <class T, std::size_t N>
constexpr simd<T, N> div(const simd<T, N>& a, const simd<T, N>& b) noexcept
{
    return apply(a, b, ops_for<simd<T, N>>::div);
}

template <class T, std::size_t N>
constexpr simd<T, N> neg(const simd<T, N>& a) noexcept
{
    return apply(a, ops_for<simd<T, N>>::neg);
}


// Reduction operations
template <class T, std::size_t N>
constexpr T sum(const simd<T, N>& a) noexcept
{
    return apply(a, ops_for<simd<T, N>>::sum);
}

template <class T, std::size_t N>
constexpr T dot(const simd<T, N>& a, const simd<T, N>& b) noexcept
{
    return apply(a, b, ops_for<simd<T, N>>::dot);
}

template <class T, std::size_t N>
constexpr T min_element(const simd<T, N>& a) noexcept
{
    return apply(a, ops_for<simd<T, N>>::min_element);
}

template <class T, std::size_t N>
constexpr T max_element(const simd<T, N>& a) noexcept
{
    return apply(a, ops_for<simd<T, N>>::max_element);
}


// Min/Max operations
template <class T, std::size_t N>
constexpr simd<T, N> min(const simd<T, N>& a, const simd<T, N>& b) noexcept
{
    return apply(a, b, ops_for<simd<T, N>>::min);
}

template <class T, std::size_t N>
constexpr simd<T, N> max(const simd<T, N>& a, const simd<T, N>& b) noexcept
{
    return apply(a, b, ops_for<simd<T, N>>::max);
}


// Bitwise operations
template <class T, std::size_t N>
constexpr simd<T, N> bitwise_and(const simd<T, N>& a, const simd<T, N>& b) noexcept
{
    return apply(a, b, ops_for<simd<T, N>>::bitwise_and);
}

template <class T, std::size_t N>
constexpr simd<T, N> bitwise_or(const simd<T, N>& a, const simd<T, N>& b) noexcept
{
    return apply(a, b, ops_for<simd<T, N>>::bitwise_or);
}

template <class T, std::size_t N>
constexpr simd<T, N> bitwise_xor(const simd<T, N>& a, const simd<T, N>& b) noexcept
{
    return apply(a, b, ops_for<simd<T, N>>::bitwise_xor);
}

template <class T, std::size_t N>
constexpr simd<T, N> bitwise_not(const simd<T, N>& a) noexcept
{
    return apply(a, ops_for<simd<T, N>>::bitwise_not);
}

template <std::size_t N>
constexpr simd_mask<N> bitwise_and(const simd_mask<N>& a, const simd_mask<N>& b) noexcept
{
    return apply(a, b, ops_for<simd_mask<N>>::bitwise_and);
}

template <std::size_t N>
constexpr simd_mask<N> bitwise_or(const simd_mask<N>& a, const simd_mask<N>& b) noexcept
{
    return apply(a, b, ops_for<simd_mask<N>>::bitwise_or);
}

template <std::size_t N>
constexpr simd_mask<N> bitwise_xor(const simd_mask<N>& a, const simd_mask<N>& b) noexcept
{
    return apply(a, b, ops_for<simd_mask<N>>::bitwise_xor);
}

template <std::size_t N>
constexpr simd_mask<N> bitwise_not(const simd_mask<N>& a) noexcept
{
    return apply(a, ops_for<simd_mask<N>>::bitwise_not);
}


// Reduction operations
template <std::size_t N>
constexpr bool any(const simd_mask<N>& mask) noexcept
{
    return [&]<std::size_t... Index>(std::index_sequence<Index...>)
    {
        return (... || ops_for<simd_mask<N>>::any(mask.data[Index]));
    }(std::make_index_sequence<simd_mask<N>::blocks()>{});
}

template <std::size_t N>
constexpr bool all(const simd_mask<N>& mask) noexcept
{
    return [&]<std::size_t... Index>(std::index_sequence<Index...>)
    {
        return (... && ops_for<simd_mask<N>>::all(mask.data[Index]));
    }(std::make_index_sequence<simd_mask<N>::blocks()>{});
}

template <std::size_t N>
constexpr bool none(const simd_mask<N>& mask) noexcept
{
    return !any(mask);
}


// Comparison
template <class T, std::size_t N>
constexpr simd_mask<N> cmp_eq(const simd<T, N>& a, const simd<T, N>& b) noexcept
{
    return compare(a, b, ops_for<simd<T, N>>::cmp_eq);
}

template <class T, std::size_t N>
constexpr simd_mask<N> cmp_ne(const simd<T, N>& a, const simd<T, N>& b) noexcept
{
    return compare(a, b, ops_for<simd<T, N>>::cmp_ne);
}

template <class T, std::size_t N>
constexpr simd_mask<N> cmp_lt(const simd<T, N>& a, const simd<T, N>& b) noexcept
{
    return compare(a, b, ops_for<simd<T, N>>::cmp_lt);
}

template <class T, std::size_t N>
constexpr simd_mask<N> cmp_le(const simd<T, N>& a, const simd<T, N>& b) noexcept
{
    return compare(a, b, ops_for<simd<T, N>>::cmp_le);
}

template <class T, std::size_t N>
constexpr simd_mask<N> cmp_gt(const simd<T, N>& a, const simd<T, N>& b) noexcept
{
    return compare(a, b, ops_for<simd<T, N>>::cmp_gt);
}

template <class T, std::size_t N>
constexpr simd_mask<N> cmp_ge(const simd<T, N>& a, const simd<T, N>& b) noexcept
{
    return compare(a, b, ops_for<simd<T, N>>::cmp_ge);
}


// Blending
template <class T, std::size_t N>
constexpr simd<T, N> select(const simd_mask<N>& mask, const simd<T, N>& a, const simd<T, N>& b) noexcept
{
    return [&]<std::size_t... Index>(std::index_sequence<Index...>)
    {
        return simd<T, N>{ ops_for<simd<T, N>>::blend(a.data[Index], b.data[Index], mask.data[Index])... };
    }(std::make_index_sequence<simd<T, N>::blocks()>{});
}

template <class T, std::size_t N>
constexpr simd<T, N> abs(const simd<T, N>& value) noexcept
{
    return max(value, -value);
}

template <class T, std::size_t N>
constexpr simd<T, N> clamp(const simd<T, N>& value, const simd<T, N>& min_value, const simd<T, N>& max_value) noexcept
{
    return max(min_value, min(max_value, value));
}

template <class T, std::size_t N>
constexpr simd<T, N> clamp_min(const simd<T, N>& value, const simd<T, N>& min_value) noexcept
{
    return max(min_value, value);
}

template <class T, std::size_t N>
constexpr simd<T, N> clamp_max(const simd<T, N>& value, const simd<T, N>& max_value) noexcept
{
    return min(max_value, value);
}

template <class T, std::size_t N>
constexpr simd<T, N> saturate(const simd<T, N>& value) noexcept
{
    return clamp(value, fill<T, N>(0), fill<T, N>(1));
}

template <class T, std::size_t N>
constexpr simd<T, N> sqrt(const simd<T, N>& value) noexcept
{
    return apply(value, ops_for<simd<T, N>>::sqrt);
}

template <class T, std::size_t N>
constexpr simd<T, N> rsqrt(const simd<T, N>& value) noexcept
{
    return apply(value, ops_for<simd<T, N>>::rsqrt);
}

template <class T, std::size_t N>
constexpr simd<T, N> reciprocal(const simd<T, N>& value) noexcept
{
    return apply(value, ops_for<simd<T, N>>::reciprocal);
}

template <class T, std::size_t N>
constexpr simd<T, N> fma(const simd<T, N>& a, const simd<T, N>& b, const simd<T, N>& c) noexcept
{
    return apply(a, b, c, ops_for<simd<T, N>>::fma);
}

template <class T, std::size_t N>
constexpr simd<T, N> round(const simd<T, N>& value) noexcept
{
    return apply(value, ops_for<simd<T, N>>::round);
}

template <class T, std::size_t N>
constexpr simd<T, N> floor(const simd<T, N>& value) noexcept
{
    return apply(value, ops_for<simd<T, N>>::floor);
}

template <class T, std::size_t N>
constexpr simd<T, N> ceil(const simd<T, N>& value) noexcept
{
    return apply(value, ops_for<simd<T, N>>::ceil);
}

template <class T, std::size_t N>
constexpr simd<T, N> trunc(const simd<T, N>& value) noexcept
{
    return apply(value, ops_for<simd<T, N>>::trunc);
}

// Masked operations
template <class T, std::size_t N>
constexpr simd<T, N> masked_fill(const simd_mask<N>& mask, const simd<T, N>& a, T value) noexcept
{
    return masked(mask, a, [&](auto) { return fill<T, N>(value); });
}

template <class T, std::size_t N>
constexpr simd<T, N> masked_fill(const simd_mask<N>& mask, T value) noexcept
{
    return masked_fill(mask, fill<T, N>(T(0)), value);
}

template <class T, std::size_t N>
constexpr simd<T, N> masked_add(const simd_mask<N>& mask, const simd<T, N>& a, const simd<T, N>& b) noexcept
{
    return masked(mask, a, b, add<T, N>);
}

template <class T, std::size_t N>
constexpr simd<T, N> masked_sub(const simd_mask<N>& mask, const simd<T, N>& a, const simd<T, N>& b) noexcept
{
    return masked(mask, a, b, sub<T, N>);
}

template <class T, std::size_t N>
constexpr simd<T, N> masked_mul(const simd_mask<N>& mask, const simd<T, N>& a, const simd<T, N>& b) noexcept
{
    return masked(mask, a, b, mul<T, N>);
}

template <class T, std::size_t N>
constexpr simd<T, N> masked_div(const simd_mask<N>& mask, const simd<T, N>& a, const simd<T, N>& b) noexcept
{
    return masked(mask, a, b, div<T, N>);
}

template <class T, std::size_t N>
constexpr simd<T, N> masked_min(const simd_mask<N>& mask, const simd<T, N>& a, const simd<T, N>& b) noexcept
{
    return masked(mask, a, b, [](auto x, auto y) { return select(x < y, x, y); });
}

template <class T, std::size_t N>
constexpr simd<T, N> masked_max(const simd_mask<N>& mask, const simd<T, N>& a, const simd<T, N>& b) noexcept
{
    return masked(mask, a, b, [](auto x, auto y) { return select(x > y, x, y); });
}

template <class T, std::size_t N>
constexpr simd<T, N> masked_clamp(const simd_mask<N>& mask, const simd<T, N>& value, const simd<T, N>& min_value, const simd<T, N>& max_value) noexcept
{
    return masked(mask, min_value, max_value, [&](auto lo, auto hi) { return clamp(value, lo, hi); });
}

template <class T, std::size_t N>
constexpr simd<T, N> masked_saturate(const simd_mask<N>& mask, const simd<T, N>& value) noexcept
{
    return masked(mask, fill<T, N>(0), fill<T, N>(1), [&](auto zero, auto one) { return clamp(value, zero, one); });
}

template <class T, std::size_t N>
constexpr simd<T, N> masked_neg(const simd_mask<N>& mask, const simd<T, N>& a) noexcept
{
    return masked(mask, a, a, neg<T, N>);
}

template <class T, std::size_t N>
constexpr simd<T, N> masked_abs(const simd_mask<N>& mask, const simd<T, N>& a) noexcept
{
    return masked(mask, a, a, abs<T, N>);
}

template <class T, std::size_t N>
constexpr simd<T, N> masked_sqrt(const simd_mask<N>& mask, const simd<T, N>& a) noexcept
{
    return masked(mask, a, a, sqrt<T, N>);
}

template <class T, std::size_t N>
constexpr simd<T, N> masked_round(const simd_mask<N>& mask, const simd<T, N>& a) noexcept
{
    return masked(mask, a, a, round<T, N>);
}

template <class T, std::size_t N>
constexpr simd<T, N> masked_floor(const simd_mask<N>& mask, const simd<T, N>& a) noexcept
{
    return masked(mask, a, a, floor<T, N>);
}

template <class T, std::size_t N>
constexpr simd<T, N> masked_ceil(const simd_mask<N>& mask, const simd<T, N>& a) noexcept
{
    return masked(mask, a, a, ceil<T, N>);
}

template <class T, std::size_t N>
constexpr simd<T, N> masked_trunc(const simd_mask<N>& mask, const simd<T, N>& a) noexcept
{
    return masked(mask, a, a, trunc<T, N>);
}

template <class T, std::size_t N>
constexpr simd<T, N> masked_bitwise_and(const simd_mask<N>& mask, const simd<T, N>& a, const simd<T, N>& b) noexcept
{
    return masked(mask, a, b, bitwise_and<T, N>);
}

template <class T, std::size_t N>
constexpr simd<T, N> masked_bitwise_or(const simd_mask<N>& mask, const simd<T, N>& a, const simd<T, N>& b) noexcept
{
    return masked(mask, a, b, bitwise_or<T, N>);
}

template <class T, std::size_t N>
constexpr simd<T, N> masked_bitwise_xor(const simd_mask<N>& mask, const simd<T, N>& a, const simd<T, N>& b) noexcept
{
    return masked(mask, a, b, bitwise_xor<T, N>);
}

template <class T, std::size_t N>
constexpr simd<T, N> masked_bitwise_not(const simd_mask<N>& mask, const simd<T, N>& a) noexcept
{
    return masked(mask, a, a, bitwise_not<T, N>);
}

template <class T, std::size_t N>
constexpr simd<T, N> masked_compare_eq(const simd_mask<N>& mask, const simd<T, N>& a, const simd<T, N>& b) noexcept
{
    return masked(mask, a, b, cmp_eq<T, N>);
}

template <class T, std::size_t N>
constexpr simd<T, N> masked_compare_ne(const simd_mask<N>& mask, const simd<T, N>& a, const simd<T, N>& b) noexcept
{
    return masked(mask, a, b, cmp_ne<T, N>);
}

template <class T, std::size_t N>
constexpr simd<T, N> masked_compare_lt(const simd_mask<N>& mask, const simd<T, N>& a, const simd<T, N>& b) noexcept
{
    return masked(mask, a, b, cmp_lt<T, N>);
}

template <class T, std::size_t N>
constexpr simd<T, N> masked_compare_le(const simd_mask<N>& mask, const simd<T, N>& a, const simd<T, N>& b) noexcept
{
    return masked(mask, a, b, cmp_le<T, N>);
}

template <class T, std::size_t N>
constexpr simd<T, N> masked_compare_gt(const simd_mask<N>& mask, const simd<T, N>& a, const simd<T, N>& b) noexcept
{
    return masked(mask, a, b, cmp_gt<T, N>);
}

template <class T, std::size_t N>
constexpr simd<T, N> masked_compare_ge(const simd_mask<N>& mask, const simd<T, N>& a, const simd<T, N>& b) noexcept
{
    return masked(mask, a, b, cmp_ge<T, N>);
}


// Operator overloads
template <class T, std::size_t N>
constexpr simd<T, N> operator+(const simd<T, N>& a, const simd<T, N>& b) noexcept
{
    return add(a, b);
}

template <class T, std::size_t N>
constexpr simd<T, N> operator-(const simd<T, N>& a, const simd<T, N>& b) noexcept
{
    return sub(a, b);
}

template <class T, std::size_t N>
constexpr simd<T, N> operator*(const simd<T, N>& a, const simd<T, N>& b) noexcept
{
    return mul(a, b);
}

template <class T, std::size_t N>
constexpr simd<T, N> operator/(const simd<T, N>& a, const simd<T, N>& b) noexcept
{
    return div(a, b);
}

template <class T, std::size_t N>
constexpr simd<T, N> operator-(const simd<T, N>& a) noexcept
{
    return neg(a);
}

template <class T, std::size_t N>
constexpr simd<T, N> operator&(const simd<T, N>& a, const simd<T, N>& b) noexcept
{
    return bitwise_and(a, b);
}

template <class T, std::size_t N>
constexpr simd<T, N> operator|(const simd<T, N>& a, const simd<T, N>& b) noexcept
{
    return bitwise_or(a, b);
}

template <class T, std::size_t N>
constexpr simd<T, N> operator^(const simd<T, N>& a, const simd<T, N>& b) noexcept
{
    return bitwise_xor(a, b);
}

template <class T, std::size_t N>
constexpr simd<T, N> operator~(const simd<T, N>& a) noexcept
{
    return bitwise_not(a);
}

template <class T, std::size_t N>
constexpr simd_mask<N> operator==(const simd<T, N>& a, const simd<T, N>& b) noexcept
{
    return cmp_eq(a, b);
}

template <class T, std::size_t N>
constexpr simd_mask<N> operator!=(const simd<T, N>& a, const simd<T, N>& b) noexcept
{
    return cmp_ne(a, b);
}

template <class T, std::size_t N>
constexpr simd_mask<N> operator<(const simd<T, N>& a, const simd<T, N>& b) noexcept
{
    return cmp_lt(a, b);
}

template <class T, std::size_t N>
constexpr simd_mask<N> operator<=(const simd<T, N>& a, const simd<T, N>& b) noexcept
{
    return cmp_le(a, b);
}

template <class T, std::size_t N>
constexpr simd_mask<N> operator>(const simd<T, N>& a, const simd<T, N>& b) noexcept
{
    return cmp_gt(a, b);
}

template <class T, std::size_t N>
constexpr simd_mask<N> operator>=(const simd<T, N>& a, const simd<T, N>& b) noexcept
{
    return cmp_ge(a, b);
}

template <std::size_t N>
constexpr simd_mask<N> operator&(const simd_mask<N>& a, const simd_mask<N>& b) noexcept
{
    return bitwise_and(a, b);
}

template <std::size_t N>
constexpr simd_mask<N> operator|(const simd_mask<N>& a, const simd_mask<N>& b) noexcept
{
    return bitwise_or(a, b);
}

template <std::size_t N>
constexpr simd_mask<N> operator^(const simd_mask<N>& a, const simd_mask<N>& b) noexcept
{
    return bitwise_xor(a, b);
}

template <std::size_t N>
constexpr simd_mask<N> operator~(const simd_mask<N>& a) noexcept
{
    return bitwise_not(a);
}

template <std::size_t N>
constexpr simd_mask<N>& operator&=(simd_mask<N>& a, const simd_mask<N>& b) noexcept
{
    return (a = a & b);
}

template <std::size_t N>
simd_mask<N>& operator|=(simd_mask<N>& a, const simd_mask<N>& b) noexcept
{
    a = a | b;
    return a;
}

template <std::size_t N>
simd_mask<N>& operator^=(simd_mask<N>& a, const simd_mask<N>& b) noexcept
{
    a = a ^ b;
    return a;
}

}