#pragma once

#include <geometric/point.h>

namespace arc::geometric
{

/**
 * @brief 2D circle represented by center point and radius.
 *
 * Negative radius inputs are stored as their absolute value.
 */
template <class T>
struct circle
{
    /// @brief Scalar coordinate and radius type.
    using value_type = T;

    /// @brief Circle center point.
    point<T, 2> center{};
    /// @brief Non-negative circle radius.
    T radius{};

    /// @brief Construct a zero-radius circle at the origin.
    constexpr circle() noexcept = default;
    /// @brief Construct from center and radius.
    constexpr circle(const point<T, 2>& center_value, T radius_value) noexcept
        : center(center_value)
        , radius(radius_value < T{} ? -radius_value : radius_value)
    {
    }
};

/**
 * @brief 3D sphere represented by center point and radius.
 *
 * Negative radius inputs are stored as their absolute value.
 */
template <class T>
struct sphere
{
    /// @brief Scalar coordinate and radius type.
    using value_type = T;

    /// @brief Sphere center point.
    point<T, 3> center{};
    /// @brief Non-negative sphere radius.
    T radius{};

    /// @brief Construct a zero-radius sphere at the origin.
    constexpr sphere() noexcept = default;
    /// @brief Construct from center and radius.
    constexpr sphere(const point<T, 3>& center_value, T radius_value) noexcept
        : center(center_value)
        , radius(radius_value < T{} ? -radius_value : radius_value)
    {
    }
};

/// @name Common circle and sphere aliases
/// @{
using circlef = circle<float>;
using circled = circle<double>;
using spheref = sphere<float>;
using sphered = sphere<double>;
/// @}

template <class T, class U>
/// @brief Return whether a 2D point lies inside or on a circle.
constexpr bool contains(const circle<T>& shape, const point<U, 2>& value) noexcept
{
    return distance_squared(shape.center, value) <= shape.radius * shape.radius;
}

template <class T, class U>
/// @brief Return whether a 3D point lies inside or on a sphere.
constexpr bool contains(const sphere<T>& shape, const point<U, 3>& value) noexcept
{
    return distance_squared(shape.center, value) <= shape.radius * shape.radius;
}

template <class T, class U>
/// @brief Return whether two circles overlap or touch.
constexpr bool intersects(const circle<T>& lhs, const circle<U>& rhs) noexcept
{
    const auto radius = lhs.radius + rhs.radius;
    return distance_squared(lhs.center, rhs.center) <= radius * radius;
}

template <class T, class U>
/// @brief Return whether two spheres overlap or touch.
constexpr bool intersects(const sphere<T>& lhs, const sphere<U>& rhs) noexcept
{
    const auto radius = lhs.radius + rhs.radius;
    return distance_squared(lhs.center, rhs.center) <= radius * radius;
}

template <class T, class U>
/// @brief Return the closest point on the circle boundary to a 2D point.
inline auto closest_point(const circle<T>& shape, const point<U, 2>& value) noexcept
{
    using value_type = std::common_type_t<T, U>;
    const auto direction = value - shape.center;
    if (arc::math::length_squared(direction) == value_type{})
        return point<value_type, 2>{ shape.center + arc::math::vector<value_type, 2>{ shape.radius, value_type{} } };
    return point<value_type, 2>{ shape.center + arc::math::mul(arc::math::normalize(direction), shape.radius) };
}

template <class T, class U>
/// @brief Return the closest point on the sphere boundary to a 3D point.
inline auto closest_point(const sphere<T>& shape, const point<U, 3>& value) noexcept
{
    using value_type = std::common_type_t<T, U>;
    const auto direction = value - shape.center;
    if (arc::math::length_squared(direction) == value_type{})
        return point<value_type, 3>{ shape.center + arc::math::vector<value_type, 3>{ shape.radius, value_type{}, value_type{} } };
    return point<value_type, 3>{ shape.center + arc::math::mul(arc::math::normalize(direction), shape.radius) };
}

} // namespace arc::geometric
