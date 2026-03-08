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
    if constexpr (std::is_void_v<decltype(op(Index...))>)
    {
        (op(Index), ...);
    }
    else
    {
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
    if constexpr (std::is_void_v<decltype(op(a.data[0]))>)
    {
        (op(a.data[Index]), ...);
    }
    else
    {
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
    if constexpr (std::is_void_v<decltype(op(a.data[0], b.data[0]))>)
    {
        (op(a.data[Index], b.data[Index]), ...);
    }
    else
    {
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
    if constexpr (std::is_void_v<decltype(op(a.data[0], b.data[0], c.data[0]))>)
    {
        (op(a.data[Index], b.data[Index], c.data[Index]), ...);
    }
    else
    {
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
    if constexpr (std::is_void_v<decltype(op(Index...))>)
    {
        (op(Index), ...);
    }
    else
    {
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
    if constexpr (std::is_void_v<decltype(op(a.data[0]))>)
    {
        (op(a.data[Index]), ...);
    }
    else
    {
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
    if constexpr (std::is_void_v<decltype(op(a.data[0], b.data[0]))>)
    {
        (op(a.data[Index], b.data[Index]), ...);
    }
    else
    {
        return simd_mask<N>{ op(a.data[Index], b.data[Index])... };
    }
}

template <std::size_t N, class Op>
constexpr auto apply(const simd_mask<N>& a, const simd_mask<N>& b, Op op) noexcept
{
    return apply(a, b, std::make_index_sequence<simd_mask<N>::blocks()>{}, op);
}

template <class T, std::size_t N, std::size_t... Index, class Map, class ReduceOp>
constexpr T reduce(const simd<T, N>& a, std::index_sequence<Index...>, Map map, ReduceOp op) noexcept
{
    T result{};
    ((result = Index == 0 ? map(a.data[Index]) : op(result, map(a.data[Index]))), ...);
    return result;
}

template <class T, std::size_t N, class Map, class ReduceOp>
constexpr T reduce(const simd<T, N>& a, Map map, ReduceOp op) noexcept
{
    return reduce(a, std::make_index_sequence<simd<T, N>::blocks()>{}, map, op);
}

template <class T, std::size_t N, std::size_t... Index, class Map, class ReduceOp>
constexpr T reduce(const simd<T, N>& a, const simd<T, N>& b, std::index_sequence<Index...>, Map map, ReduceOp op) noexcept
{
    T result{};
    ((result = Index == 0 ? map(a.data[Index], b.data[Index]) : op(result, map(a.data[Index], b.data[Index]))), ...);
    return result;
}

template <class T, std::size_t N, class Map, class ReduceOp>
constexpr T reduce(const simd<T, N>& a, const simd<T, N>& b, Map map, ReduceOp op) noexcept
{
    return reduce(a, b, std::make_index_sequence<simd<T, N>::blocks()>{}, map, op);
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

} // namespace arc
