#pragma once

#include <geometric/point.h>

namespace arc::geometric
{

template <class T, std::size_t N>
struct line
{
    using value_type = T;
    static constexpr std::size_t size = N;

    point<T, N> origin{};
    arc::math::vector<T, N> direction{};

    constexpr line() noexcept = default;
    constexpr line(const point<T, N>& origin_value, const arc::math::vector<T, N>& direction_value) noexcept
        : origin(origin_value)
        , direction(direction_value)
    {
    }
};

template <class T, std::size_t N>
struct ray
{
    using value_type = T;
    static constexpr std::size_t size = N;

    point<T, N> origin{};
    arc::math::vector<T, N> direction{};

    constexpr ray() noexcept = default;
    constexpr ray(const point<T, N>& origin_value, const arc::math::vector<T, N>& direction_value) noexcept
        : origin(origin_value)
        , direction(direction_value)
    {
    }
};

template <class T, std::size_t N>
struct segment
{
    using value_type = T;
    static constexpr std::size_t size = N;

    point<T, N> start{};
    point<T, N> end{};

    constexpr segment() noexcept = default;
    constexpr segment(const point<T, N>& start_value, const point<T, N>& end_value) noexcept
        : start(start_value)
        , end(end_value)
    {
    }
};

using line2f = line<float, 2>;
using line3f = line<float, 3>;
using ray2f = ray<float, 2>;
using ray3f = ray<float, 3>;
using segment2f = segment<float, 2>;
using segment3f = segment<float, 3>;

template <class T, std::size_t N>
constexpr auto point_at(const line<T, N>& value, T t) noexcept
{
    return value.origin + arc::math::mul(value.direction, t);
}

template <class T, std::size_t N>
constexpr auto point_at(const ray<T, N>& value, T t) noexcept
{
    return value.origin + arc::math::mul(value.direction, t);
}

template <class T, std::size_t N>
constexpr auto point_at(const segment<T, N>& value, T t) noexcept
{
    return value.start + arc::math::mul(value.end - value.start, t);
}

} // namespace arc::geometric
