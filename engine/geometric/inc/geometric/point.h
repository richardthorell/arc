#pragma once

#include <math/vector.h>

namespace arc::geometric
{

template <class T, std::size_t N>
class point
{
public:
    using value_type = T;
    using vector_type = arc::math::vector<T, N>;
    static constexpr std::size_t size = N;

    constexpr point() noexcept = default;

    constexpr explicit point(T value) noexcept
        : values_(value)
    {
    }

    constexpr point(std::initializer_list<T> values)
        : values_(values)
    {
    }

    constexpr explicit point(const vector_type& values) noexcept
        : values_(values)
    {
    }

    template <arc::math::detail::vector_expression Expr>
        requires (arc::math::detail::expr_traits<std::remove_cvref_t<Expr>>::size == N)
    constexpr explicit point(const Expr& expr) noexcept
        : values_(expr)
    {
    }

    constexpr T* data() noexcept
    {
        return values_.data();
    }

    constexpr const T* data() const noexcept
    {
        return values_.data();
    }

    constexpr T& operator[](std::size_t index) noexcept
    {
        return values_[index];
    }

    constexpr const T& operator[](std::size_t index) const noexcept
    {
        return values_[index];
    }

    constexpr vector_type& as_vector() noexcept
    {
        return values_;
    }

    constexpr const vector_type& as_vector() const noexcept
    {
        return values_;
    }

private:
    vector_type values_{};
};

using point2f = point<float, 2>;
using point3f = point<float, 3>;
using point2d = point<double, 2>;
using point3d = point<double, 3>;
using point2i = point<int, 2>;
using point3i = point<int, 3>;

template <class T, std::size_t N, arc::math::detail::vector_expression Vec>
    requires (arc::math::detail::expr_traits<std::remove_cvref_t<Vec>>::size == N)
constexpr auto add(const point<T, N>& lhs, const Vec& rhs) noexcept
{
    using value_type = std::common_type_t<T, arc::math::detail::expr_value_t<Vec>>;
    return point<value_type, N>{ arc::math::add(lhs.as_vector(), rhs) };
}

template <arc::math::detail::vector_expression Vec, class T, std::size_t N>
    requires (arc::math::detail::expr_traits<std::remove_cvref_t<Vec>>::size == N)
constexpr auto add(const Vec& lhs, const point<T, N>& rhs) noexcept
{
    return add(rhs, lhs);
}

template <class T, std::size_t N, arc::math::detail::vector_expression Vec>
    requires (arc::math::detail::expr_traits<std::remove_cvref_t<Vec>>::size == N)
constexpr auto sub(const point<T, N>& lhs, const Vec& rhs) noexcept
{
    using value_type = std::common_type_t<T, arc::math::detail::expr_value_t<Vec>>;
    return point<value_type, N>{ arc::math::sub(lhs.as_vector(), rhs) };
}

template <class T, class U, std::size_t N>
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
inline auto distance(const point<T, N>& lhs, const point<U, N>& rhs) noexcept
{
    return arc::math::distance(lhs.as_vector(), rhs.as_vector());
}

template <class T, class U, std::size_t N>
constexpr auto distance_squared(const point<T, N>& lhs, const point<U, N>& rhs) noexcept
{
    return arc::math::distance_squared(lhs.as_vector(), rhs.as_vector());
}

} // namespace arc::geometric
