#pragma once

#include <math/detail/evaluation.h>

#include <cstdint>

namespace arc::math
{

template <class T, std::size_t N>
class vector final : public detail::vector_expr<vector<T, N>>
{
public:
    using value_type = T;
    static constexpr std::size_t size = N;

    constexpr vector() noexcept = default;

    constexpr explicit vector(T value) noexcept
    {
        values_.fill(value);
    }

    constexpr vector(std::initializer_list<T> values)
    {
        if (values.size() != N)
            throw std::length_error("vector initializer size mismatch");

        std::size_t index = 0;
        for (const T& value : values)
            values_[index++] = value;
    }

    template <detail::vector_expression Expr>
        requires (detail::expr_traits<std::remove_cvref_t<Expr>>::size == N)
    constexpr vector(const Expr& expr) noexcept
    {
        assign(expr);
    }

    template <detail::vector_expression Expr>
        requires (detail::expr_traits<std::remove_cvref_t<Expr>>::size == N)
    constexpr vector& operator=(const Expr& expr) noexcept
    {
        assign(expr);
        return *this;
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

private:
    template <detail::vector_expression Expr>
    constexpr void assign(const Expr& expr) noexcept
    {
        detail::assign_vector<T, N>(values_, expr);
    }

    std::array<T, N> values_{};
};

using vector2f = vector<float, 2>;
using vector3f = vector<float, 3>;
using vector4f = vector<float, 4>;
using vector2d = vector<double, 2>;
using vector3d = vector<double, 3>;
using vector4d = vector<double, 4>;
using vector2i = vector<int, 2>;
using vector3i = vector<int, 3>;
using vector4i = vector<int, 4>;
using vector2u = vector<std::uint32_t, 2>;
using vector3u = vector<std::uint32_t, 3>;
using vector4u = vector<std::uint32_t, 4>;

namespace detail
{

template <class T, std::size_t N>
struct expr_traits<vector<T, N>>
{
    using value_type = T;
    static constexpr int kind = 0;
    static constexpr std::size_t size = N;
};

template <class Op, class Lhs, class Rhs>
struct expr_traits<vector_binary_expr<Op, Lhs, Rhs>>
{
    using value_type = typename vector_binary_expr<Op, Lhs, Rhs>::value_type;
    static constexpr int kind = 0;
    static constexpr std::size_t size = vector_binary_expr<Op, Lhs, Rhs>::size;
};

template <class Expr>
struct expr_traits<vector_neg_expr<Expr>>
{
    using value_type = typename vector_neg_expr<Expr>::value_type;
    static constexpr int kind = 0;
    static constexpr std::size_t size = vector_neg_expr<Expr>::size;
};

template <class Op, bool ScalarOnLeft, class Scalar, class Expr>
struct expr_traits<vector_scalar_expr<Op, ScalarOnLeft, Scalar, Expr>>
{
    using value_type = typename vector_scalar_expr<Op, ScalarOnLeft, Scalar, Expr>::value_type;
    static constexpr int kind = 0;
    static constexpr std::size_t size = vector_scalar_expr<Op, ScalarOnLeft, Scalar, Expr>::size;
};

template <class A, class B, class C>
struct expr_traits<vector_fma_expr<A, B, C>>
{
    using value_type = typename vector_fma_expr<A, B, C>::value_type;
    static constexpr int kind = 0;
    static constexpr std::size_t size = vector_fma_expr<A, B, C>::size;
};

} // namespace detail

template <detail::vector_expression Expr>
constexpr auto eval(const Expr& expr) noexcept
{
    using traits = detail::expr_traits<std::remove_cvref_t<Expr>>;
    return vector<typename traits::value_type, traits::size>{ expr };
}

template <detail::vector_expression Lhs, detail::vector_expression Rhs>
    requires detail::same_vector_dimensions<Lhs, Rhs>
constexpr auto add(Lhs&& lhs, Rhs&& rhs)
{
    return detail::vector_binary_expr<detail::add_op, Lhs, Rhs>{ std::forward<Lhs>(lhs), std::forward<Rhs>(rhs) };
}

template <detail::vector_expression Lhs, detail::vector_expression Rhs>
    requires detail::same_vector_dimensions<Lhs, Rhs>
constexpr auto sub(Lhs&& lhs, Rhs&& rhs)
{
    return detail::vector_binary_expr<detail::sub_op, Lhs, Rhs>{ std::forward<Lhs>(lhs), std::forward<Rhs>(rhs) };
}

template <detail::vector_expression Lhs, detail::vector_expression Rhs>
    requires detail::same_vector_dimensions<Lhs, Rhs>
constexpr auto mul(Lhs&& lhs, Rhs&& rhs)
{
    return detail::vector_binary_expr<detail::mul_op, Lhs, Rhs>{ std::forward<Lhs>(lhs), std::forward<Rhs>(rhs) };
}

template <detail::vector_expression Lhs, detail::vector_expression Rhs>
    requires detail::same_vector_dimensions<Lhs, Rhs>
constexpr auto div(Lhs&& lhs, Rhs&& rhs)
{
    return detail::vector_binary_expr<detail::div_op, Lhs, Rhs>{ std::forward<Lhs>(lhs), std::forward<Rhs>(rhs) };
}

template <detail::vector_expression Expr>
constexpr auto neg(Expr&& expr)
{
    return detail::vector_neg_expr<Expr>{ std::forward<Expr>(expr) };
}

template <detail::vector_expression Expr, class Scalar>
    requires detail::scalar_for<Scalar, detail::expr_value_t<Expr>>
constexpr auto add(Expr&& expr, Scalar scalar)
{
    return detail::vector_scalar_expr<detail::add_op, false, Scalar, Expr>{ scalar, std::forward<Expr>(expr) };
}

template <class Scalar, detail::vector_expression Expr>
    requires detail::scalar_for<Scalar, detail::expr_value_t<Expr>>
constexpr auto add(Scalar scalar, Expr&& expr)
{
    return detail::vector_scalar_expr<detail::add_op, true, Scalar, Expr>{ scalar, std::forward<Expr>(expr) };
}

template <detail::vector_expression Expr, class Scalar>
    requires detail::scalar_for<Scalar, detail::expr_value_t<Expr>>
constexpr auto sub(Expr&& expr, Scalar scalar)
{
    return detail::vector_scalar_expr<detail::sub_op, false, Scalar, Expr>{ scalar, std::forward<Expr>(expr) };
}

template <class Scalar, detail::vector_expression Expr>
    requires detail::scalar_for<Scalar, detail::expr_value_t<Expr>>
constexpr auto sub(Scalar scalar, Expr&& expr)
{
    return detail::vector_scalar_expr<detail::sub_op, true, Scalar, Expr>{ scalar, std::forward<Expr>(expr) };
}

template <detail::vector_expression Expr, class Scalar>
    requires detail::scalar_for<Scalar, detail::expr_value_t<Expr>>
constexpr auto mul(Expr&& expr, Scalar scalar)
{
    return detail::vector_scalar_expr<detail::mul_op, false, Scalar, Expr>{ scalar, std::forward<Expr>(expr) };
}

template <class Scalar, detail::vector_expression Expr>
    requires detail::scalar_for<Scalar, detail::expr_value_t<Expr>>
constexpr auto mul(Scalar scalar, Expr&& expr)
{
    return detail::vector_scalar_expr<detail::mul_op, true, Scalar, Expr>{ scalar, std::forward<Expr>(expr) };
}

template <detail::vector_expression Expr, class Scalar>
    requires detail::scalar_for<Scalar, detail::expr_value_t<Expr>>
constexpr auto div(Expr&& expr, Scalar scalar)
{
    return detail::vector_scalar_expr<detail::div_op, false, Scalar, Expr>{ scalar, std::forward<Expr>(expr) };
}

template <class Scalar, detail::vector_expression Expr>
    requires detail::scalar_for<Scalar, detail::expr_value_t<Expr>>
constexpr auto div(Scalar scalar, Expr&& expr)
{
    return detail::vector_scalar_expr<detail::div_op, true, Scalar, Expr>{ scalar, std::forward<Expr>(expr) };
}

template <detail::vector_expression A, detail::vector_expression B, detail::vector_expression C>
    requires (detail::same_vector_dimensions<A, B> && detail::same_vector_dimensions<A, C>)
constexpr auto fma(A&& a, B&& b, C&& c)
{
    return detail::vector_fma_expr<A, B, C>{ std::forward<A>(a), std::forward<B>(b), std::forward<C>(c) };
}

template <detail::vector_expression Lhs, detail::vector_expression Rhs>
    requires detail::same_vector_dimensions<Lhs, Rhs>
constexpr auto dot(const Lhs& lhs, const Rhs& rhs) noexcept
{
    using lhs_traits = detail::expr_traits<std::remove_cvref_t<Lhs>>;
    using value_type = std::common_type_t<typename lhs_traits::value_type, detail::expr_value_t<Rhs>>;

    value_type result{};
    for (std::size_t i = 0; i < lhs_traits::size; ++i)
        result += static_cast<value_type>(lhs[i]) * static_cast<value_type>(rhs[i]);
    return result;
}

template <detail::vector_expression Expr>
constexpr auto length_squared(const Expr& expr) noexcept
{
    return dot(expr, expr);
}

template <detail::vector_expression Expr>
inline auto length(const Expr& expr) noexcept
{
    using value_type = detail::expr_value_t<Expr>;
    return static_cast<value_type>(std::sqrt(length_squared(expr)));
}

template <detail::vector_expression Expr>
inline auto normalize(const Expr& expr, detail::expr_value_t<Expr> fallback = detail::expr_value_t<Expr>{}) noexcept
{
    using traits = detail::expr_traits<std::remove_cvref_t<Expr>>;
    using value_type = typename traits::value_type;

    vector<value_type, traits::size> result{};
    const value_type len = length(expr);
    if (len == value_type{})
    {
        for (std::size_t i = 0; i < traits::size; ++i)
            result[i] = fallback;
    }
    else
    {
        for (std::size_t i = 0; i < traits::size; ++i)
            result[i] = static_cast<value_type>(expr[i] / len);
    }
    return result;
}

template <detail::vector_expression Lhs, detail::vector_expression Rhs>
    requires detail::same_vector_dimensions<Lhs, Rhs>
constexpr auto distance_squared(const Lhs& lhs, const Rhs& rhs) noexcept
{
    return length_squared(sub(lhs, rhs));
}

template <detail::vector_expression Lhs, detail::vector_expression Rhs>
    requires detail::same_vector_dimensions<Lhs, Rhs>
inline auto distance(const Lhs& lhs, const Rhs& rhs) noexcept
{
    using value_type = std::common_type_t<detail::expr_value_t<Lhs>, detail::expr_value_t<Rhs>>;
    return static_cast<value_type>(std::sqrt(distance_squared(lhs, rhs)));
}

template <detail::vector_expression Lhs, detail::vector_expression Rhs>
    requires (detail::same_vector_dimensions<Lhs, Rhs> && detail::expr_traits<std::remove_cvref_t<Lhs>>::size == 3)
constexpr auto cross(const Lhs& lhs, const Rhs& rhs) noexcept
{
    using value_type = std::common_type_t<detail::expr_value_t<Lhs>, detail::expr_value_t<Rhs>>;
    return vector<value_type, 3>{
        static_cast<value_type>(lhs[1] * rhs[2] - lhs[2] * rhs[1]),
        static_cast<value_type>(lhs[2] * rhs[0] - lhs[0] * rhs[2]),
        static_cast<value_type>(lhs[0] * rhs[1] - lhs[1] * rhs[0])
    };
}

template <detail::vector_expression Lhs, detail::vector_expression Rhs>
    requires (detail::same_vector_dimensions<Lhs, Rhs> && detail::expr_traits<std::remove_cvref_t<Lhs>>::size == 2)
constexpr auto cross2(const Lhs& lhs, const Rhs& rhs) noexcept
{
    using value_type = std::common_type_t<detail::expr_value_t<Lhs>, detail::expr_value_t<Rhs>>;
    return static_cast<value_type>(lhs[0] * rhs[1] - lhs[1] * rhs[0]);
}

template <detail::vector_expression A, detail::vector_expression B, class T>
    requires (detail::same_vector_dimensions<A, B> && detail::scalar_for<T, detail::expr_value_t<A>>)
constexpr auto lerp(A&& a, B&& b, T t)
{
    return add(std::forward<A>(a), mul(sub(std::forward<B>(b), std::forward<A>(a)), t));
}

template <detail::vector_expression Lhs, detail::vector_expression Rhs>
    requires detail::same_vector_dimensions<Lhs, Rhs>
constexpr auto min(const Lhs& lhs, const Rhs& rhs) noexcept
{
    using traits = detail::expr_traits<std::remove_cvref_t<Lhs>>;
    using value_type = std::common_type_t<typename traits::value_type, detail::expr_value_t<Rhs>>;

    vector<value_type, traits::size> result{};
    for (std::size_t i = 0; i < traits::size; ++i)
    {
        const value_type a = static_cast<value_type>(lhs[i]);
        const value_type b = static_cast<value_type>(rhs[i]);
        result[i] = b < a ? b : a;
    }
    return result;
}

template <detail::vector_expression Lhs, detail::vector_expression Rhs>
    requires detail::same_vector_dimensions<Lhs, Rhs>
constexpr auto max(const Lhs& lhs, const Rhs& rhs) noexcept
{
    using traits = detail::expr_traits<std::remove_cvref_t<Lhs>>;
    using value_type = std::common_type_t<typename traits::value_type, detail::expr_value_t<Rhs>>;

    vector<value_type, traits::size> result{};
    for (std::size_t i = 0; i < traits::size; ++i)
    {
        const value_type a = static_cast<value_type>(lhs[i]);
        const value_type b = static_cast<value_type>(rhs[i]);
        result[i] = a < b ? b : a;
    }
    return result;
}

template <detail::vector_expression Value, detail::vector_expression Min, detail::vector_expression Max>
    requires (detail::same_vector_dimensions<Value, Min> && detail::same_vector_dimensions<Value, Max>)
constexpr auto clamp(const Value& value, const Min& minimum, const Max& maximum) noexcept
{
    return min(max(value, minimum), maximum);
}

template <detail::vector_expression Value, detail::vector_expression Normal>
    requires detail::same_vector_dimensions<Value, Normal>
constexpr auto reflect(const Value& value, const Normal& normal)
{
    using value_type = std::common_type_t<detail::expr_value_t<Value>, detail::expr_value_t<Normal>>;
    return sub(value, mul(normal, static_cast<value_type>(2) * dot(value, normal)));
}

template <detail::vector_expression Value, detail::vector_expression Onto>
    requires detail::same_vector_dimensions<Value, Onto>
constexpr auto project(const Value& value, const Onto& onto)
{
    using traits = detail::expr_traits<std::remove_cvref_t<Value>>;
    using value_type = std::common_type_t<typename traits::value_type, detail::expr_value_t<Onto>>;

    const value_type denom = dot(onto, onto);
    if (denom == value_type{})
        return vector<value_type, traits::size>{};
    return vector<value_type, traits::size>{ mul(onto, dot(value, onto) / denom) };
}

template <detail::vector_expression Lhs, detail::vector_expression Rhs>
    requires detail::same_vector_dimensions<Lhs, Rhs>
constexpr auto operator+(Lhs&& lhs, Rhs&& rhs)
{
    return add(std::forward<Lhs>(lhs), std::forward<Rhs>(rhs));
}

template <detail::vector_expression Lhs, detail::vector_expression Rhs>
    requires detail::same_vector_dimensions<Lhs, Rhs>
constexpr auto operator-(Lhs&& lhs, Rhs&& rhs)
{
    return sub(std::forward<Lhs>(lhs), std::forward<Rhs>(rhs));
}

template <detail::vector_expression Lhs, detail::vector_expression Rhs>
    requires detail::same_vector_dimensions<Lhs, Rhs>
constexpr auto operator*(Lhs&& lhs, Rhs&& rhs)
{
    return mul(std::forward<Lhs>(lhs), std::forward<Rhs>(rhs));
}

template <detail::vector_expression Lhs, detail::vector_expression Rhs>
    requires detail::same_vector_dimensions<Lhs, Rhs>
constexpr auto operator/(Lhs&& lhs, Rhs&& rhs)
{
    return div(std::forward<Lhs>(lhs), std::forward<Rhs>(rhs));
}

template <detail::vector_expression Expr>
constexpr auto operator-(Expr&& expr)
{
    return neg(std::forward<Expr>(expr));
}

template <detail::vector_expression Expr, class Scalar>
    requires detail::scalar_for<Scalar, detail::expr_value_t<Expr>>
constexpr auto operator+(Expr&& expr, Scalar scalar)
{
    return add(std::forward<Expr>(expr), scalar);
}

template <class Scalar, detail::vector_expression Expr>
    requires detail::scalar_for<Scalar, detail::expr_value_t<Expr>>
constexpr auto operator+(Scalar scalar, Expr&& expr)
{
    return add(scalar, std::forward<Expr>(expr));
}

template <detail::vector_expression Expr, class Scalar>
    requires detail::scalar_for<Scalar, detail::expr_value_t<Expr>>
constexpr auto operator-(Expr&& expr, Scalar scalar)
{
    return sub(std::forward<Expr>(expr), scalar);
}

template <class Scalar, detail::vector_expression Expr>
    requires detail::scalar_for<Scalar, detail::expr_value_t<Expr>>
constexpr auto operator-(Scalar scalar, Expr&& expr)
{
    return sub(scalar, std::forward<Expr>(expr));
}

template <detail::vector_expression Expr, class Scalar>
    requires detail::scalar_for<Scalar, detail::expr_value_t<Expr>>
constexpr auto operator*(Expr&& expr, Scalar scalar)
{
    return mul(std::forward<Expr>(expr), scalar);
}

template <class Scalar, detail::vector_expression Expr>
    requires detail::scalar_for<Scalar, detail::expr_value_t<Expr>>
constexpr auto operator*(Scalar scalar, Expr&& expr)
{
    return mul(scalar, std::forward<Expr>(expr));
}

template <detail::vector_expression Expr, class Scalar>
    requires detail::scalar_for<Scalar, detail::expr_value_t<Expr>>
constexpr auto operator/(Expr&& expr, Scalar scalar)
{
    return div(std::forward<Expr>(expr), scalar);
}

template <class Scalar, detail::vector_expression Expr>
    requires detail::scalar_for<Scalar, detail::expr_value_t<Expr>>
constexpr auto operator/(Scalar scalar, Expr&& expr)
{
    return div(scalar, std::forward<Expr>(expr));
}

} // namespace arc::math
