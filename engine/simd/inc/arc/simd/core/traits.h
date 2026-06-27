#pragma once

#include <concepts>

namespace arc
{

template <class T, std::size_t N>
struct simd;

template <std::size_t N>
struct simd_mask;

template <class>
struct simd_traits;

template <class T, std::size_t N>
struct simd_traits<simd<T, N>>
{
    using value_type = T;
    static constexpr std::size_t size = N;
    static constexpr std::size_t blocks = simd<T, N>::blocks();
    static constexpr bool is_mask = false;
};

template <std::size_t N>
struct simd_traits<simd_mask<N>>
{
    using value_type = void;
    static constexpr std::size_t size = N;
    static constexpr std::size_t blocks = simd_mask<N>::blocks();
    static constexpr bool is_mask = true;
};

template <class V>
concept simd_like =
    requires
    {
        simd_traits<std::remove_cvref_t<V>>::size;
    };

template <class Arg, class Ret>
concept apply_operand_for =
    simd_like<Arg> &&
    simd_like<Ret> &&
    (simd_traits<std::remove_cvref_t<Arg>>::size ==
     simd_traits<std::remove_cvref_t<Ret>>::size) &&
    (
        simd_traits<std::remove_cvref_t<Arg>>::is_mask ||
        simd_traits<std::remove_cvref_t<Ret>>::is_mask ||
        std::same_as<
            typename simd_traits<std::remove_cvref_t<Arg>>::value_type,
            typename simd_traits<std::remove_cvref_t<Ret>>::value_type>
    );

} // namespace arc
