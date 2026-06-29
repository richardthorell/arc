#pragma once

#include <arc/math/vector.h>

namespace arc::geometric
{

/**
 * @brief Strong point type backed by `arc::math::vector<T, N>`.
 *
 * Points represent positions. Subtracting two points returns a vector, while
 * adding or subtracting a vector moves the point.
 */
template <class T, std::size_t N>
class point
{
public:
    /// @brief Scalar coordinate type.
    using value_type = T;
    /// @brief Backing vector type used for storage and math interop.
    using vector_type = arc::math::vector<T, N>;
    /// @brief Number of coordinates.
    static constexpr std::size_t size = N;

    /// @brief Construct a zero-initialized point.
    constexpr point() noexcept = default;

    /// @brief Fill every coordinate with the same scalar value.
    constexpr explicit point(T value) noexcept
        : values_(value)
    {
    }

    /**
     * @brief Construct from exactly `N` coordinate values.
     * @throws std::length_error when the initializer length does not match `N`.
     */
    constexpr point(std::initializer_list<T> values)
        : values_(values)
    {
    }

    /// @brief Construct from a backing vector.
    constexpr explicit point(const vector_type& values) noexcept
        : values_(values)
    {
    }

    template <arc::math::detail::vector_expression Expr>
        requires (arc::math::detail::expr_traits<std::remove_cvref_t<Expr>>::size == N)
    /// @brief Materialize a compatible vector expression as a point.
    constexpr explicit point(const Expr& expr) noexcept
        : values_(expr)
    {
    }

    /// @brief Return mutable contiguous coordinate storage.
    constexpr T* data() noexcept
    {
        return values_.data();
    }

    /// @brief Return immutable contiguous coordinate storage.
    constexpr const T* data() const noexcept
    {
        return values_.data();
    }

    /// @brief Access a coordinate by zero-based index.
    constexpr T& operator[](std::size_t index) noexcept
    {
        return values_[index];
    }

    /// @brief Access a coordinate by zero-based index.
    constexpr const T& operator[](std::size_t index) const noexcept
    {
        return values_[index];
    }

    /// @brief Return the mutable backing vector.
    constexpr vector_type& as_vector() noexcept
    {
        return values_;
    }

    /// @brief Return the immutable backing vector.
    constexpr const vector_type& as_vector() const noexcept
    {
        return values_;
    }

private:
    vector_type values_{};
};

/// @name Common point aliases
/// @{
using point2f = point<float, 2>;
using point3f = point<float, 3>;
using point2d = point<double, 2>;
using point3d = point<double, 3>;
using point2i = point<int, 2>;
using point3i = point<int, 3>;
/// @}

template <class T, std::size_t N, arc::math::detail::vector_expression Vec>
    requires (arc::math::detail::expr_traits<std::remove_cvref_t<Vec>>::size == N)
/// @brief Move a point by a vector.
constexpr auto add(const point<T, N>& lhs, const Vec& rhs) noexcept
{
    using value_type = std::common_type_t<T, arc::math::detail::expr_value_t<Vec>>;
    return point<value_type, N>{ arc::math::add(lhs.as_vector(), rhs) };
}

template <arc::math::detail::vector_expression Vec, class T, std::size_t N>
    requires (arc::math::detail::expr_traits<std::remove_cvref_t<Vec>>::size == N)
/// @brief Move a point by a vector.
constexpr auto add(const Vec& lhs, const point<T, N>& rhs) noexcept
{
    return add(rhs, lhs);
}

template <class T, std::size_t N, arc::math::detail::vector_expression Vec>
    requires (arc::math::detail::expr_traits<std::remove_cvref_t<Vec>>::size == N)
/// @brief Move a point backward by a vector.
constexpr auto sub(const point<T, N>& lhs, const Vec& rhs) noexcept
{
    using value_type = std::common_type_t<T, arc::math::detail::expr_value_t<Vec>>;
    return point<value_type, N>{ arc::math::sub(lhs.as_vector(), rhs) };
}

template <class T, class U, std::size_t N>
/// @brief Return the vector from `rhs` to `lhs`.
constexpr auto sub(const point<T, N>& lhs, const point<U, N>& rhs) noexcept
{
    using value_type = std::common_type_t<T, U>;
    arc::math::vector<value_type, N> result{};
    for (std::size_t i = 0; i < N; ++i)
        result[i] = static_cast<value_type>(lhs[i]) - static_cast<value_type>(rhs[i]);
    return result;
}

template <class T, std::size_t N, arc::math::detail::vector_expression Vec>
    requires (arc::math::detail::expr_traits<std::remove_cvref_t<Vec>>::size == N)
constexpr auto operator+(const point<T, N>& lhs, const Vec& rhs) noexcept
{
    return add(lhs, rhs);
}

template <arc::math::detail::vector_expression Vec, class T, std::size_t N>
    requires (arc::math::detail::expr_traits<std::remove_cvref_t<Vec>>::size == N)
constexpr auto operator+(const Vec& lhs, const point<T, N>& rhs) noexcept
{
    return add(lhs, rhs);
}

template <class T, std::size_t N, arc::math::detail::vector_expression Vec>
    requires (arc::math::detail::expr_traits<std::remove_cvref_t<Vec>>::size == N)
constexpr auto operator-(const point<T, N>& lhs, const Vec& rhs) noexcept
{
    return sub(lhs, rhs);
}

template <class T, class U, std::size_t N>
constexpr auto operator-(const point<T, N>& lhs, const point<U, N>& rhs) noexcept
{
    return sub(lhs, rhs);
}

template <class T, class U, std::size_t N>
/// @brief Return the Euclidean distance between two points.
inline auto distance(const point<T, N>& lhs, const point<U, N>& rhs) noexcept
{
    return arc::math::distance(lhs.as_vector(), rhs.as_vector());
}

template <class T, class U, std::size_t N>
/// @brief Return the squared Euclidean distance between two points.
constexpr auto distance_squared(const point<T, N>& lhs, const point<U, N>& rhs) noexcept
{
    return arc::math::distance_squared(lhs.as_vector(), rhs.as_vector());
}

} // namespace arc::geometric
