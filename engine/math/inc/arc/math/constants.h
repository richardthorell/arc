#pragma once

#include <concepts>
#include <numbers>

namespace arc::math
{

template<std::floating_point T>
inline constexpr T pi = std::numbers::pi_v<T>;

template<std::floating_point T>
inline constexpr T tau = T{ 2 } * pi<T>;

template<std::floating_point T>
inline constexpr T degrees_to_radians = pi<T> / T{ 180 };

template<std::floating_point T>
inline constexpr T radians_to_degrees = T{ 180 } / pi<T>;

template<std::floating_point T>
[[nodiscard]] constexpr T to_radians(T degrees) noexcept
{
    return degrees * degrees_to_radians<T>;
}

template<std::floating_point T>
[[nodiscard]] constexpr T to_degrees(T radians) noexcept
{
    return radians * radians_to_degrees<T>;
}

} // namespace arc::math
