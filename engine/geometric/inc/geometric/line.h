#pragma once

#include <geometric/point.h>

namespace arc::geometric
{

/**
 * @brief Infinite parametric line in `N` dimensions.
 *
 * Points on the line are sampled as `origin + direction * t`.
 */
template <class T, std::size_t N>
struct line
{
    /// @brief Scalar coordinate type.
    using value_type = T;
    /// @brief Number of dimensions.
    static constexpr std::size_t size = N;

    /// @brief Point on the line.
    point<T, N> origin{};
    /// @brief Direction vector. It is not normalized automatically.
    arc::math::vector<T, N> direction{};

    /// @brief Construct a zero-initialized line.
    constexpr line() noexcept = default;
    /// @brief Construct from an origin point and direction vector.
    constexpr line(const point<T, N>& origin_value, const arc::math::vector<T, N>& direction_value) noexcept
        : origin(origin_value)
        , direction(direction_value)
    {
    }
};

/**
 * @brief Half-infinite parametric ray in `N` dimensions.
 *
 * Points on the ray are sampled as `origin + direction * t`; callers should use
 * non-negative `t` for the conventional ray domain.
 */
template <class T, std::size_t N>
struct ray
{
    /// @brief Scalar coordinate type.
    using value_type = T;
    /// @brief Number of dimensions.
    static constexpr std::size_t size = N;

    /// @brief Ray origin point.
    point<T, N> origin{};
    /// @brief Ray direction vector. It is not normalized automatically.
    arc::math::vector<T, N> direction{};

    /// @brief Construct a zero-initialized ray.
    constexpr ray() noexcept = default;
    /// @brief Construct from an origin point and direction vector.
    constexpr ray(const point<T, N>& origin_value, const arc::math::vector<T, N>& direction_value) noexcept
        : origin(origin_value)
        , direction(direction_value)
    {
    }
};

/**
 * @brief Finite segment between two points.
 *
 * Points on the segment are sampled as `start + (end - start) * t`; callers
 * should use `0 <= t <= 1` for the conventional segment domain.
 */
template <class T, std::size_t N>
struct segment
{
    /// @brief Scalar coordinate type.
    using value_type = T;
    /// @brief Number of dimensions.
    static constexpr std::size_t size = N;

    /// @brief Segment start point.
    point<T, N> start{};
    /// @brief Segment end point.
    point<T, N> end{};

    /// @brief Construct a zero-initialized segment.
    constexpr segment() noexcept = default;
    /// @brief Construct from start and end points.
    constexpr segment(const point<T, N>& start_value, const point<T, N>& end_value) noexcept
        : start(start_value)
        , end(end_value)
    {
    }
};

/// @name Common line/ray/segment aliases
/// @{
using line2f = line<float, 2>;
using line3f = line<float, 3>;
using ray2f = ray<float, 2>;
using ray3f = ray<float, 3>;
using segment2f = segment<float, 2>;
using segment3f = segment<float, 3>;
/// @}

template <class T, std::size_t N>
/// @brief Sample an infinite line at parameter `t`.
constexpr auto point_at(const line<T, N>& value, T t) noexcept
{
    return value.origin + arc::math::mul(value.direction, t);
}

template <class T, std::size_t N>
/// @brief Sample a ray at parameter `t`.
constexpr auto point_at(const ray<T, N>& value, T t) noexcept
{
    return value.origin + arc::math::mul(value.direction, t);
}

template <class T, std::size_t N>
/// @brief Sample a segment at parameter `t`.
constexpr auto point_at(const segment<T, N>& value, T t) noexcept
{
    return value.start + arc::math::mul(value.end - value.start, t);
}

} // namespace arc::geometric
