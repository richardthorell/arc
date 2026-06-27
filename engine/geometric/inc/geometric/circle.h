#pragma once

#include <geometric/point.h>

namespace arc::geometric
{

template <class T>
struct circle
{
    using value_type = T;

    point<T, 2> center{};
    T radius{};

    constexpr circle() noexcept = default;
    constexpr circle(const point<T, 2>& center_value, T radius_value) noexcept
        : center(center_value)
        , radius(radius_value < T{} ? -radius_value : radius_value)
    {
    }
};

template <class T>
struct sphere
{
    using value_type = T;

    point<T, 3> center{};
    T radius{};

    constexpr sphere() noexcept = default;
    constexpr sphere(const point<T, 3>& center_value, T radius_value) noexcept
        : center(center_value)
        , radius(radius_value < T{} ? -radius_value : radius_value)
    {
    }
};

using circlef = circle<float>;
using circled = circle<double>;
using spheref = sphere<float>;
using sphered = sphere<double>;

template <class T, class U>
constexpr bool contains(const circle<T>& shape, const point<U, 2>& value) noexcept
{
    return distance_squared(shape.center, value) <= shape.radius * shape.radius;
}

template <class T, class U>
constexpr bool contains(const sphere<T>& shape, const point<U, 3>& value) noexcept
{
    return distance_squared(shape.center, value) <= shape.radius * shape.radius;
}

template <class T, class U>
constexpr bool intersects(const circle<T>& lhs, const circle<U>& rhs) noexcept
{
    const auto radius = lhs.radius + rhs.radius;
    return distance_squared(lhs.center, rhs.center) <= radius * radius;
}

template <class T, class U>
constexpr bool intersects(const sphere<T>& lhs, const sphere<U>& rhs) noexcept
{
    const auto radius = lhs.radius + rhs.radius;
    return distance_squared(lhs.center, rhs.center) <= radius * radius;
}

template <class T, class U>
inline auto closest_point(const circle<T>& shape, const point<U, 2>& value) noexcept
{
    using value_type = std::common_type_t<T, U>;
    const auto direction = value - shape.center;
    if (arc::math::length_squared(direction) == value_type{})
        return point<value_type, 2>{ shape.center + arc::math::vector<value_type, 2>{ shape.radius, value_type{} } };
    return point<value_type, 2>{ shape.center + arc::math::mul(arc::math::normalize(direction), shape.radius) };
}

template <class T, class U>
inline auto closest_point(const sphere<T>& shape, const point<U, 3>& value) noexcept
{
    using value_type = std::common_type_t<T, U>;
    const auto direction = value - shape.center;
    if (arc::math::length_squared(direction) == value_type{})
        return point<value_type, 3>{ shape.center + arc::math::vector<value_type, 3>{ shape.radius, value_type{}, value_type{} } };
    return point<value_type, 3>{ shape.center + arc::math::mul(arc::math::normalize(direction), shape.radius) };
}

} // namespace arc::geometric
