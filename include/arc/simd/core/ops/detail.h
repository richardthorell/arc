#pragma once

#include <arc/simd/core/traits.h>
#include <arc/simd/core/simd.h>
#include <arc/simd/core/mask.h>

namespace arc
{
namespace detail
{

struct simd_access
{
    template <class T, std::size_t N, class... Blocks>
    static constexpr simd<T, N> make_simd(Blocks... blocks) noexcept
    {
        return simd<T, N>{ blocks... };
    }

    template <std::size_t N, class... Blocks>
    static constexpr simd_mask<N> make_mask(Blocks... blocks) noexcept
    {
        return simd_mask<N>{ blocks... };
    }

    template <class T, std::size_t N>
    static constexpr auto& data(simd<T, N>& value) noexcept
    {
        return value.data;
    }

    template <class T, std::size_t N>
    static constexpr const auto& data(const simd<T, N>& value) noexcept
    {
        return value.data;
    }

    template <class T, std::size_t N>
    static constexpr auto& block(simd<T, N>& value, std::size_t index) noexcept
    {
        return value.data[index];
    }

    template <class T, std::size_t N>
    static constexpr const auto& block(const simd<T, N>& value, std::size_t index) noexcept
    {
        return value.data[index];
    }

    template <std::size_t N>
    static constexpr auto& data(simd_mask<N>& value) noexcept
    {
        return value.data;
    }

    template <std::size_t N>
    static constexpr const auto& data(const simd_mask<N>& value) noexcept
    {
        return value.data;
    }

    template <std::size_t N>
    static constexpr auto& block(simd_mask<N>& value, std::size_t index) noexcept
    {
        return value.data[index];
    }

    template <std::size_t N>
    static constexpr const auto& block(const simd_mask<N>& value, std::size_t index) noexcept
    {
        return value.data[index];
    }
};

} // namespace detail

template <class T, std::size_t N>
constexpr simd<T, N> select(const simd_mask<N>&, const simd<T, N>&, const simd<T, N>&) noexcept;

template <class Ret, std::size_t... Index, class Op, class... Args>
    requires (sizeof...(Args) > 0) &&
(apply_operand_for<Args, Ret> && ...)
constexpr auto apply(std::index_sequence<Index...>, Op op, const Args&... args) noexcept
{
    using detail::simd_access;

    using block_result_t = std::invoke_result_t<
        Op,
        decltype(simd_access::block(args, 0))...
    >;

    auto invoke_at = [&](std::size_t i) noexcept
    {
        return op(simd_access::block(args, i)...);
    };

    if constexpr (std::is_void_v<block_result_t>)
    {
        (invoke_at(Index), ...);
    }
    else if constexpr (simd_traits<Ret>::is_mask)
    {
        return simd_access::template make_mask<simd_traits<Ret>::size>(
            invoke_at(Index)...
        );
    }
    else
    {
        return simd_access::template make_simd<typename simd_traits<Ret>::value_type, simd_traits<Ret>::size>(
            invoke_at(Index)...
        );
    }
}

template <class Ret, class Op, class... Args>
    requires (sizeof...(Args) > 0) && (apply_operand_for<Args, Ret> && ...)
constexpr auto apply(Op op, const Args&... args) noexcept
{
    return apply<Ret>(
        std::make_index_sequence<simd_traits<Ret>::blocks>{},
        op,
        args...
    );
}

template <class Ret, std::size_t... Index, class Op>
constexpr auto apply(std::index_sequence<Index...>, Op op) noexcept
{
    using detail::simd_access;

    using block_result_t = std::invoke_result_t<Op, std::size_t>;

    if constexpr (std::is_void_v<block_result_t>)
    {
        (op(Index), ...);
    }
    else if constexpr (simd_traits<Ret>::is_mask)
    {
        return simd_access::template make_mask<simd_traits<Ret>::size>(
            op(Index)...
        );
    }
    else
    {
        return simd_access::template make_simd<typename simd_traits<Ret>::value_type,  simd_traits<Ret>::size>(
            op(Index)...
        );
    }
}

template <class Ret, class Op>
constexpr auto apply(Op op) noexcept
{
    return apply<Ret>(
        std::make_index_sequence<simd_traits<Ret>::blocks>{},
        op
    );
}

template <class T, std::size_t N, std::size_t... Index, class Map, class ReduceOp>
constexpr T reduce(const simd<T, N>& a, std::index_sequence<Index...>, Map map, ReduceOp op) noexcept
{
    T result{};
    ((result = Index == 0 ? map(detail::simd_access::block(a, Index)) : op(result, map(detail::simd_access::block(a, Index)))), ...);
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
    ((result = Index == 0 ? map(detail::simd_access::block(a, Index), detail::simd_access::block(b, Index)) : op(result, map(detail::simd_access::block(a, Index), detail::simd_access::block(b, Index)))), ...);
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
    using detail::simd_access;

    return simd_access::template make_mask<N>(
        op(simd_access::block(a, Index), simd_access::block(b, Index))...
    );
}

template <class T, std::size_t N, class Op>
constexpr simd_mask<N> compare(const simd<T, N>& a, const simd<T, N>& b, Op op) noexcept
{
    return compare(a, b, std::make_index_sequence<simd_block<T, N>::blocks>{}, op);
}

template <class T, std::size_t N, std::size_t... Index, class Op>
constexpr simd<T, N> masked(const simd_mask<N>& mask, const simd<T, N>& a, const simd<T, N>& b, std::index_sequence<Index...>, Op op) noexcept
{
    using detail::simd_access;

    return simd_access::template make_simd<T, N>(
        select(simd_access::block(mask, Index), simd_access::block(a, Index), op(simd_access::block(a, Index), simd_access::block(b, Index)))...
    );
}

template <class T, std::size_t N, class Op>
constexpr simd<T, N> masked(const simd_mask<N>& mask, const simd<T, N>& a, const simd<T, N>& b, Op op) noexcept
{
    return masked(mask, a, b, std::make_index_sequence<simd<T, N>::blocks()>{}, op);
}

template <class T, std::size_t N, std::size_t... Index, class Op>
constexpr simd<T, N> masked(const simd_mask<N>& mask, const simd<T, N>& a, std::index_sequence<Index...>, Op op) noexcept
{
    using detail::simd_access;

    return simd_access::template make_simd<T, N>(
        select(simd_access::block(mask, Index), simd_access::block(a, Index), op(simd_access::block(a, Index)))...
    );
}

template <class T, std::size_t N, class Op>
constexpr simd<T, N> masked(const simd_mask<N>& mask, const simd<T, N>& a, Op op) noexcept
{
    return masked(mask, a, std::make_index_sequence<simd<T, N>::blocks()>{}, op);
}

} // namespace arc
