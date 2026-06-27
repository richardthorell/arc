#pragma once

#include <arc/simd/core/ops/scalar.h>

#include <type_traits>

namespace arc
{

template <int Bits, class T, std::size_t N>
    requires std::is_integral_v<T>
constexpr simd<T, N> shift_left(const simd<T, N>& value) noexcept
{
    return detail::simd_map(value, [](T lane) { return static_cast<T>(lane << Bits); });
}

template <int Bits, class T, std::size_t N>
    requires std::is_integral_v<T>
constexpr simd<T, N> shift_right_arithmetic(const simd<T, N>& value) noexcept
{
    return detail::simd_map(value, [](T lane) { return static_cast<T>(lane >> Bits); });
}

template <int Bits, class T, std::size_t N>
    requires std::is_integral_v<T>
constexpr simd<T, N> shift_right_logical(const simd<T, N>& value) noexcept
{
    using unsigned_t = std::make_unsigned_t<T>;
    return detail::simd_map(value, [](T lane)
    {
        return static_cast<T>(static_cast<unsigned_t>(lane) >> Bits);
    });
}

} // namespace arc
