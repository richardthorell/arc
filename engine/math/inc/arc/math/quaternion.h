#pragma once

#include <arc/math/vector.h>

namespace arc::math
{

/**
 * @brief Quaternion value stored as `(x, y, z, w)`.
 *
 * Quaternions default to the identity rotation `{0, 0, 0, 1}` and support
 * lazy elementwise arithmetic. Rotation helpers are provided as free functions.
 */
template <class T>
class quaternion final : public detail::quaternion_expr<quaternion<T>>
{
public:
    /// @brief Scalar component type.
    using value_type = T;
    /// @brief Number of stored components.
    static constexpr std::size_t size = 4;

    /// @brief Construct the identity quaternion.
    constexpr quaternion() noexcept
        : values_{ T{}, T{}, T{}, T{ 1 } }
    {
    }

    /// @brief Fill all four components with the same scalar value.
    constexpr explicit quaternion(T value) noexcept
        : values_{ value, value, value, value }
    {
    }

    /// @brief Construct from explicit `(x, y, z, w)` components.
    constexpr quaternion(T x, T y, T z, T w) noexcept
        : values_{ x, y, z, w }
    {
    }

    /**
     * @brief Construct from exactly four initializer-list values.
     * @throws std::length_error when the initializer length is not four.
     */
    constexpr quaternion(std::initializer_list<T> values)
    {
        if (values.size() != size)
            throw std::length_error("quaternion initializer size mismatch");

        std::size_t index = 0;
        for (const T& value : values)
            values_[index++] = value;
    }

    template <detail::quaternion_expression Expr>
    /// @brief Materialize a compatible quaternion expression.
    constexpr quaternion(const Expr& expr) noexcept
    {
        assign(expr);
    }

    template <detail::quaternion_expression Expr>
    /// @brief Assign and materialize a compatible quaternion expression.
    constexpr quaternion& operator=(const Expr& expr) noexcept
    {
        assign(expr);
        return *this;
    }

    /// @brief Return mutable contiguous component storage in `(x, y, z, w)` order.
    constexpr T* data() noexcept
    {
        return values_.data();
    }

    /// @brief Return immutable contiguous component storage in `(x, y, z, w)` order.
    constexpr const T* data() const noexcept
    {
        return values_.data();
    }

    /// @brief Access a component by zero-based index.
    constexpr T& operator[](std::size_t index) noexcept
    {
        return values_[index];
    }

    /// @brief Access a component by zero-based index.
    constexpr const T& operator[](std::size_t index) const noexcept
    {
        return values_[index];
    }

    /// @brief Mutable X component accessor.
    constexpr T& x() noexcept { return values_[0]; }
    /// @brief Mutable Y component accessor.
    constexpr T& y() noexcept { return values_[1]; }
    /// @brief Mutable Z component accessor.
    constexpr T& z() noexcept { return values_[2]; }
    /// @brief Mutable W component accessor.
    constexpr T& w() noexcept { return values_[3]; }

    /// @brief Immutable X component accessor.
    constexpr const T& x() const noexcept { return values_[0]; }
    /// @brief Immutable Y component accessor.
    constexpr const T& y() const noexcept { return values_[1]; }
    /// @brief Immutable Z component accessor.
    constexpr const T& z() const noexcept { return values_[2]; }
    /// @brief Immutable W component accessor.
    constexpr const T& w() const noexcept { return values_[3]; }

private:
    template <detail::quaternion_expression Expr>
    constexpr void assign(const Expr& expr) noexcept
    {
        detail::assign_quaternion<T>(values_, expr);
    }

    std::array<T, size> values_{};
};

/// @name Common quaternion aliases
/// @{
using quaternionf = quaternion<float>;
using quaterniond = quaternion<double>;
using quatf = quaternion<float>;
using quatd = quaternion<double>;
/// @}

namespace detail
{

template <class T>
struct expr_traits<quaternion<T>>
{
    using value_type = T;
    static constexpr int kind = 2;
    static constexpr std::size_t size = 4;
};

template <class Op, class Lhs, class Rhs>
struct expr_traits<quaternion_binary_expr<Op, Lhs, Rhs>>
{
    using value_type = typename quaternion_binary_expr<Op, Lhs, Rhs>::value_type;
    static constexpr int kind = 2;
    static constexpr std::size_t size = quaternion_binary_expr<Op, Lhs, Rhs>::size;
};

template <class Expr>
struct expr_traits<quaternion_neg_expr<Expr>>
{
    using value_type = typename quaternion_neg_expr<Expr>::value_type;
    static constexpr int kind = 2;
    static constexpr std::size_t size = quaternion_neg_expr<Expr>::size;
};

template <class Op, bool ScalarOnLeft, class Scalar, class Expr>
struct expr_traits<quaternion_scalar_expr<Op, ScalarOnLeft, Scalar, Expr>>
{
    using value_type = typename quaternion_scalar_expr<Op, ScalarOnLeft, Scalar, Expr>::value_type;
    static constexpr int kind = 2;
    static constexpr std::size_t size = quaternion_scalar_expr<Op, ScalarOnLeft, Scalar, Expr>::size;
};

} // namespace detail

template <detail::quaternion_expression Expr>
/// @brief Materialize a quaternion expression as a concrete `quaternion`.
constexpr auto eval(const Expr& expr) noexcept
{
    using traits = detail::expr_traits<std::remove_cvref_t<Expr>>;
    return quaternion<typename traits::value_type>{ expr };
}

template <detail::quaternion_expression Lhs, detail::quaternion_expression Rhs>
/// @brief Build a lazy elementwise quaternion addition expression.
constexpr auto add(Lhs&& lhs, Rhs&& rhs)
{
    return detail::quaternion_binary_expr<detail::add_op, Lhs, Rhs>{ std::forward<Lhs>(lhs), std::forward<Rhs>(rhs) };
}

template <detail::quaternion_expression Lhs, detail::quaternion_expression Rhs>
/// @brief Build a lazy elementwise quaternion subtraction expression.
constexpr auto sub(Lhs&& lhs, Rhs&& rhs)
{
    return detail::quaternion_binary_expr<detail::sub_op, Lhs, Rhs>{ std::forward<Lhs>(lhs), std::forward<Rhs>(rhs) };
}

template <detail::quaternion_expression Lhs, detail::quaternion_expression Rhs>
/// @brief Build a lazy elementwise quaternion multiplication expression.
constexpr auto mul(Lhs&& lhs, Rhs&& rhs)
{
    return detail::quaternion_binary_expr<detail::mul_op, Lhs, Rhs>{ std::forward<Lhs>(lhs), std::forward<Rhs>(rhs) };
}

template <detail::quaternion_expression Lhs, detail::quaternion_expression Rhs>
/// @brief Build a lazy elementwise quaternion division expression.
constexpr auto div(Lhs&& lhs, Rhs&& rhs)
{
    return detail::quaternion_binary_expr<detail::div_op, Lhs, Rhs>{ std::forward<Lhs>(lhs), std::forward<Rhs>(rhs) };
}

template <detail::quaternion_expression Expr>
/// @brief Build a lazy quaternion negation expression.
constexpr auto neg(Expr&& expr)
{
    return detail::quaternion_neg_expr<Expr>{ std::forward<Expr>(expr) };
}

template <detail::quaternion_expression Expr, class Scalar>
    requires detail::scalar_for<Scalar, detail::expr_value_t<Expr>>
/// @brief Build a lazy expression adding a scalar to every quaternion component.
constexpr auto add(Expr&& expr, Scalar scalar)
{
    return detail::quaternion_scalar_expr<detail::add_op, false, Scalar, Expr>{ scalar, std::forward<Expr>(expr) };
}

template <class Scalar, detail::quaternion_expression Expr>
    requires detail::scalar_for<Scalar, detail::expr_value_t<Expr>>
/// @brief Build a lazy expression adding every quaternion component to a scalar.
constexpr auto add(Scalar scalar, Expr&& expr)
{
    return detail::quaternion_scalar_expr<detail::add_op, true, Scalar, Expr>{ scalar, std::forward<Expr>(expr) };
}

template <detail::quaternion_expression Expr, class Scalar>
    requires detail::scalar_for<Scalar, detail::expr_value_t<Expr>>
/// @brief Build a lazy expression subtracting a scalar from every quaternion component.
constexpr auto sub(Expr&& expr, Scalar scalar)
{
    return detail::quaternion_scalar_expr<detail::sub_op, false, Scalar, Expr>{ scalar, std::forward<Expr>(expr) };
}

template <class Scalar, detail::quaternion_expression Expr>
    requires detail::scalar_for<Scalar, detail::expr_value_t<Expr>>
/// @brief Build a lazy expression subtracting every quaternion component from a scalar.
constexpr auto sub(Scalar scalar, Expr&& expr)
{
    return detail::quaternion_scalar_expr<detail::sub_op, true, Scalar, Expr>{ scalar, std::forward<Expr>(expr) };
}

template <detail::quaternion_expression Expr, class Scalar>
    requires detail::scalar_for<Scalar, detail::expr_value_t<Expr>>
/// @brief Build a lazy expression multiplying every quaternion component by a scalar.
constexpr auto mul(Expr&& expr, Scalar scalar)
{
    return detail::quaternion_scalar_expr<detail::mul_op, false, Scalar, Expr>{ scalar, std::forward<Expr>(expr) };
}

template <class Scalar, detail::quaternion_expression Expr>
    requires detail::scalar_for<Scalar, detail::expr_value_t<Expr>>
/// @brief Build a lazy expression multiplying a scalar by every quaternion component.
constexpr auto mul(Scalar scalar, Expr&& expr)
{
    return detail::quaternion_scalar_expr<detail::mul_op, true, Scalar, Expr>{ scalar, std::forward<Expr>(expr) };
}

template <detail::quaternion_expression Expr, class Scalar>
    requires detail::scalar_for<Scalar, detail::expr_value_t<Expr>>
/// @brief Build a lazy expression dividing every quaternion component by a scalar.
constexpr auto div(Expr&& expr, Scalar scalar)
{
    return detail::quaternion_scalar_expr<detail::div_op, false, Scalar, Expr>{ scalar, std::forward<Expr>(expr) };
}

template <class Scalar, detail::quaternion_expression Expr>
    requires detail::scalar_for<Scalar, detail::expr_value_t<Expr>>
/// @brief Build a lazy expression dividing a scalar by every quaternion component.
constexpr auto div(Scalar scalar, Expr&& expr)
{
    return detail::quaternion_scalar_expr<detail::div_op, true, Scalar, Expr>{ scalar, std::forward<Expr>(expr) };
}

template <detail::quaternion_expression Expr>
/// @brief Return the quaternion conjugate.
constexpr auto conjugate(const Expr& expr) noexcept
{
    using value_type = detail::expr_value_t<Expr>;
    return quaternion<value_type>{
        static_cast<value_type>(-expr[0]),
        static_cast<value_type>(-expr[1]),
        static_cast<value_type>(-expr[2]),
        static_cast<value_type>(expr[3])
    };
}

template <detail::quaternion_expression Expr>
/// @brief Return the squared quaternion length.
constexpr auto length_squared(const Expr& expr) noexcept
{
    using value_type = detail::expr_value_t<Expr>;
    value_type result{};
    for (std::size_t i = 0; i < 4; ++i)
        result += static_cast<value_type>(expr[i]) * static_cast<value_type>(expr[i]);
    return result;
}

template <detail::quaternion_expression Expr>
/// @brief Return the quaternion length.
inline auto length(const Expr& expr) noexcept
{
    using value_type = detail::expr_value_t<Expr>;
    return static_cast<value_type>(std::sqrt(length_squared(expr)));
}

template <detail::quaternion_expression Expr>
/// @brief Return the quaternion inverse, or a zero-filled quaternion for zero length.
constexpr auto inverse(const Expr& expr) noexcept
{
    using value_type = detail::expr_value_t<Expr>;
    const value_type len_sq = length_squared(expr);
    if (len_sq == value_type{})
        return quaternion<value_type>{ value_type{} };
    return quaternion<value_type>{ div(conjugate(expr), len_sq) };
}

template <detail::quaternion_expression Expr>
/// @brief Return a normalized quaternion, or identity for zero length.
inline auto normalize(const Expr& expr) noexcept
{
    using value_type = detail::expr_value_t<Expr>;
    const value_type len = length(expr);
    if (len == value_type{})
        return quaternion<value_type>{};
    return quaternion<value_type>{ div(expr, len) };
}

template <detail::vector_expression Axis>
    requires (detail::expr_traits<std::remove_cvref_t<Axis>>::size == 3)
/// @brief Build a rotation quaternion from a 3D axis and angle in radians.
inline auto from_axis_angle(const Axis& axis, detail::expr_value_t<Axis> radians) noexcept
{
    using value_type = detail::expr_value_t<Axis>;
    const auto unit_axis = normalize(axis);
    const value_type half = static_cast<value_type>(radians / value_type{ 2 });
    const value_type s = static_cast<value_type>(std::sin(half));
    const value_type c = static_cast<value_type>(std::cos(half));
    return quaternion<value_type>{
        static_cast<value_type>(unit_axis[0] * s),
        static_cast<value_type>(unit_axis[1] * s),
        static_cast<value_type>(unit_axis[2] * s),
        c
    };
}

template <detail::quaternion_expression Quat, detail::vector_expression Vec>
    requires (detail::expr_traits<std::remove_cvref_t<Vec>>::size == 3)
/// @brief Rotate a 3D vector by a quaternion.
inline auto rotate(const Quat& quat, const Vec& vec) noexcept
{
    using value_type = std::common_type_t<detail::expr_value_t<Quat>, detail::expr_value_t<Vec>>;
    const auto q = normalize(quat);
    const vector<value_type, 3> qvec{ q[0], q[1], q[2] };
    const vector<value_type, 3> v{ vec };
    const vector<value_type, 3> t = mul(cross(qvec, v), value_type{ 2 });
    return vector<value_type, 3>{ add(add(v, mul(t, q[3])), cross(qvec, t)) };
}

template <detail::quaternion_expression Lhs, detail::quaternion_expression Rhs>
constexpr auto operator+(Lhs&& lhs, Rhs&& rhs)
{
    return add(std::forward<Lhs>(lhs), std::forward<Rhs>(rhs));
}

template <detail::quaternion_expression Lhs, detail::quaternion_expression Rhs>
constexpr auto operator-(Lhs&& lhs, Rhs&& rhs)
{
    return sub(std::forward<Lhs>(lhs), std::forward<Rhs>(rhs));
}

template <detail::quaternion_expression Lhs, detail::quaternion_expression Rhs>
constexpr auto operator*(Lhs&& lhs, Rhs&& rhs)
{
    return mul(std::forward<Lhs>(lhs), std::forward<Rhs>(rhs));
}

template <detail::quaternion_expression Lhs, detail::quaternion_expression Rhs>
constexpr auto operator/(Lhs&& lhs, Rhs&& rhs)
{
    return div(std::forward<Lhs>(lhs), std::forward<Rhs>(rhs));
}

template <detail::quaternion_expression Expr>
constexpr auto operator-(Expr&& expr)
{
    return neg(std::forward<Expr>(expr));
}

template <detail::quaternion_expression Expr, class Scalar>
    requires detail::scalar_for<Scalar, detail::expr_value_t<Expr>>
constexpr auto operator+(Expr&& expr, Scalar scalar)
{
    return add(std::forward<Expr>(expr), scalar);
}

template <class Scalar, detail::quaternion_expression Expr>
    requires detail::scalar_for<Scalar, detail::expr_value_t<Expr>>
constexpr auto operator+(Scalar scalar, Expr&& expr)
{
    return add(scalar, std::forward<Expr>(expr));
}

template <detail::quaternion_expression Expr, class Scalar>
    requires detail::scalar_for<Scalar, detail::expr_value_t<Expr>>
constexpr auto operator-(Expr&& expr, Scalar scalar)
{
    return sub(std::forward<Expr>(expr), scalar);
}

template <class Scalar, detail::quaternion_expression Expr>
    requires detail::scalar_for<Scalar, detail::expr_value_t<Expr>>
constexpr auto operator-(Scalar scalar, Expr&& expr)
{
    return sub(scalar, std::forward<Expr>(expr));
}

template <detail::quaternion_expression Expr, class Scalar>
    requires detail::scalar_for<Scalar, detail::expr_value_t<Expr>>
constexpr auto operator*(Expr&& expr, Scalar scalar)
{
    return mul(std::forward<Expr>(expr), scalar);
}

template <class Scalar, detail::quaternion_expression Expr>
    requires detail::scalar_for<Scalar, detail::expr_value_t<Expr>>
constexpr auto operator*(Scalar scalar, Expr&& expr)
{
    return mul(scalar, std::forward<Expr>(expr));
}

template <detail::quaternion_expression Expr, class Scalar>
    requires detail::scalar_for<Scalar, detail::expr_value_t<Expr>>
constexpr auto operator/(Expr&& expr, Scalar scalar)
{
    return div(std::forward<Expr>(expr), scalar);
}

template <class Scalar, detail::quaternion_expression Expr>
    requires detail::scalar_for<Scalar, detail::expr_value_t<Expr>>
constexpr auto operator/(Scalar scalar, Expr&& expr)
{
    return div(scalar, std::forward<Expr>(expr));
}

} // namespace arc::math
