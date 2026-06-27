#pragma once

#include <geometric/point.h>

namespace arc::geometric
{

/**
 * @brief Axis-aligned box in `N` dimensions.
 *
 * The constructor normalizes `min` and `max` per coordinate so bounds remain
 * ordered even when inputs are reversed.
 */
template <class T, std::size_t N>
struct box
{
    /// @brief Scalar coordinate type.
    using value_type = T;
    /// @brief Number of dimensions.
    static constexpr std::size_t size = N;

    /// @brief Minimum corner.
    point<T, N> min{};
    /// @brief Maximum corner.
    point<T, N> max{};

    /// @brief Construct a zero-sized box at the origin.
    constexpr box() noexcept = default;

    /// @brief Construct from minimum and maximum corners, normalizing reversed coordinates.
    constexpr box(const point<T, N>& minimum, const point<T, N>& maximum) noexcept
        : min(minimum)
        , max(maximum)
    {
        for (std::size_t i = 0; i < N; ++i)
        {
            if (max[i] < min[i])
            {
                const T temp = min[i];
                min[i] = max[i];
                max[i] = temp;
            }
        }
    }
};

/// @name Common box aliases
/// @{
using box2f = box<float, 2>;
using box3f = box<float, 3>;
using box2d = box<double, 2>;
using box3d = box<double, 3>;
/// @}

template <class T, std::size_t N>
/// @brief Return the full side lengths of a box.
constexpr auto size(const box<T, N>& value) noexcept
{
    return value.max - value.min;
}

template <class T, std::size_t N>
/// @brief Return half side lengths of a box.
constexpr auto extents(const box<T, N>& value) noexcept
{
    return arc::math::mul(size(value), T{ 0.5 });
}

template <class T, std::size_t N>
/// @brief Return the center point of a box.
constexpr auto center(const box<T, N>& value) noexcept
{
    return value.min + extents(value);
}

template <class T, class U, std::size_t N>
/// @brief Return whether a point lies inside or on a box.
constexpr bool contains(const box<T, N>& bounds, const point<U, N>& value) noexcept
{
    for (std::size_t i = 0; i < N; ++i)
    {
        if (value[i] < bounds.min[i] || bounds.max[i] < value[i])
            return false;
    }
    return true;
}

template <class T, class U, std::size_t N>
/// @brief Return whether two boxes overlap or touch.
constexpr bool intersects(const box<T, N>& lhs, const box<U, N>& rhs) noexcept
{
    for (std::size_t i = 0; i < N; ++i)
    {
        if (lhs.max[i] < rhs.min[i] || rhs.max[i] < lhs.min[i])
            return false;
    }
    return true;
}

template <class T, class U, std::size_t N>
/// @brief Clamp a point to the nearest point on or inside a box.
constexpr auto closest_point(const box<T, N>& bounds, const point<U, N>& value) noexcept
{
    using value_type = std::common_type_t<T, U>;
    point<value_type, N> result{};
    for (std::size_t i = 0; i < N; ++i)
    {
        const value_type v = static_cast<value_type>(value[i]);
        const value_type minimum = static_cast<value_type>(bounds.min[i]);
        const value_type maximum = static_cast<value_type>(bounds.max[i]);
        result[i] = v < minimum ? minimum : (maximum < v ? maximum : v);
    }
    return result;
}

template <class T, class U, std::size_t N>
/// @brief Return a box expanded just enough to include a point.
constexpr auto expand(const box<T, N>& bounds, const point<U, N>& value) noexcept
{
    using value_type = std::common_type_t<T, U>;
    point<value_type, N> minimum{};
    point<value_type, N> maximum{};
    for (std::size_t i = 0; i < N; ++i)
    {
        const value_type current = static_cast<value_type>(value[i]);
        const value_type min_value = static_cast<value_type>(bounds.min[i]);
        const value_type max_value = static_cast<value_type>(bounds.max[i]);
        minimum[i] = current < min_value ? current : min_value;
        maximum[i] = max_value < current ? current : max_value;
    }
    return box<value_type, N>{ minimum, maximum };
}

template <class T, std::size_t N>
/// @brief Return a box expanded uniformly in every direction by `amount`.
constexpr auto expand(const box<T, N>& bounds, T amount) noexcept
{
    arc::math::vector<T, N> offset(amount);
    return box<T, N>{ bounds.min - offset, bounds.max + offset };
}

} // namespace arc::geometric
