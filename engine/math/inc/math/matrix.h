#pragma once

#include <math/vector.h>

namespace arc::math
{

template <class T, std::size_t Rows, std::size_t Cols = Rows, matrix_layout Layout = matrix_layout::column_major>
class matrix final : public detail::matrix_expr<matrix<T, Rows, Cols, Layout>>
{
public:
    using value_type = T;
    static constexpr std::size_t rows = Rows;
    static constexpr std::size_t cols = Cols;
    static constexpr matrix_layout layout = Layout;
    static constexpr std::size_t size = Rows * Cols;

    constexpr matrix() noexcept = default;

    constexpr explicit matrix(T value) noexcept
    {
        values_.fill(value);
    }

    constexpr matrix(std::initializer_list<T> values)
    {
        if (values.size() != size)
            throw std::length_error("matrix initializer size mismatch");

        std::size_t index = 0;
        for (const T& value : values)
            values_[index++] = value;
    }

    template <detail::matrix_expression Expr>
        requires (
            detail::expr_traits<std::remove_cvref_t<Expr>>::rows == Rows &&
            detail::expr_traits<std::remove_cvref_t<Expr>>::cols == Cols)
    constexpr matrix(const Expr& expr) noexcept
    {
        assign(expr);
    }

    template <detail::matrix_expression Expr>
        requires (
            detail::expr_traits<std::remove_cvref_t<Expr>>::rows == Rows &&
            detail::expr_traits<std::remove_cvref_t<Expr>>::cols == Cols)
    constexpr matrix& operator=(const Expr& expr) noexcept
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

    constexpr T& operator()(std::size_t row, std::size_t col) noexcept
    {
        return values_[storage_index(row, col)];
    }

    constexpr const T& operator()(std::size_t row, std::size_t col) const noexcept
    {
        return values_[storage_index(row, col)];
    }

private:
    static constexpr std::size_t storage_index(std::size_t row, std::size_t col) noexcept
    {
        if constexpr (Layout == matrix_layout::row_major)
            return row * Cols + col;
        else
            return col * Rows + row;
    }

    template <detail::matrix_expression Expr>
    constexpr void assign(const Expr& expr) noexcept
    {
        detail::assign_matrix<T, Rows, Cols>(*this, expr);
    }

    std::array<T, size> values_{};
};

using matrix2f = matrix<float, 2>;
using matrix3f = matrix<float, 3>;
using matrix4f = matrix<float, 4>;
using matrix2d = matrix<double, 2>;
using matrix3d = matrix<double, 3>;
using matrix4d = matrix<double, 4>;
using matrix2x2f = matrix<float, 2, 2>;
using matrix3x3f = matrix<float, 3, 3>;
using matrix4x4f = matrix<float, 4, 4>;
using matrix2x2d = matrix<double, 2, 2>;
using matrix3x3d = matrix<double, 3, 3>;
using matrix4x4d = matrix<double, 4, 4>;

namespace detail
{

template <class T, std::size_t Rows, std::size_t Cols, matrix_layout Layout>
struct expr_traits<matrix<T, Rows, Cols, Layout>>
{
    using value_type = T;
    static constexpr int kind = 1;
    static constexpr std::size_t rows = Rows;
    static constexpr std::size_t cols = Cols;
    static constexpr matrix_layout layout = Layout;
};

template <class Op, class Lhs, class Rhs>
struct expr_traits<matrix_binary_expr<Op, Lhs, Rhs>>
{
    using value_type = typename matrix_binary_expr<Op, Lhs, Rhs>::value_type;
    static constexpr int kind = 1;
    static constexpr std::size_t rows = matrix_binary_expr<Op, Lhs, Rhs>::rows;
    static constexpr std::size_t cols = matrix_binary_expr<Op, Lhs, Rhs>::cols;
    static constexpr matrix_layout layout = matrix_binary_expr<Op, Lhs, Rhs>::layout;
};

template <class Expr>
struct expr_traits<matrix_neg_expr<Expr>>
{
    using value_type = typename matrix_neg_expr<Expr>::value_type;
    static constexpr int kind = 1;
    static constexpr std::size_t rows = matrix_neg_expr<Expr>::rows;
    static constexpr std::size_t cols = matrix_neg_expr<Expr>::cols;
    static constexpr matrix_layout layout = matrix_neg_expr<Expr>::layout;
};

template <class Op, bool ScalarOnLeft, class Scalar, class Expr>
struct expr_traits<matrix_scalar_expr<Op, ScalarOnLeft, Scalar, Expr>>
{
    using value_type = typename matrix_scalar_expr<Op, ScalarOnLeft, Scalar, Expr>::value_type;
    static constexpr int kind = 1;
    static constexpr std::size_t rows = matrix_scalar_expr<Op, ScalarOnLeft, Scalar, Expr>::rows;
    static constexpr std::size_t cols = matrix_scalar_expr<Op, ScalarOnLeft, Scalar, Expr>::cols;
    static constexpr matrix_layout layout = matrix_scalar_expr<Op, ScalarOnLeft, Scalar, Expr>::layout;
};

} // namespace detail

template <detail::matrix_expression Expr>
constexpr auto eval(const Expr& expr) noexcept
{
    using traits = detail::expr_traits<std::remove_cvref_t<Expr>>;
    return matrix<typename traits::value_type, traits::rows, traits::cols, traits::layout>{ expr };
}

template <detail::matrix_expression Lhs, detail::matrix_expression Rhs>
    requires detail::same_matrix_dimensions<Lhs, Rhs>
constexpr auto add(Lhs&& lhs, Rhs&& rhs)
{
    return detail::matrix_binary_expr<detail::add_op, Lhs, Rhs>{ std::forward<Lhs>(lhs), std::forward<Rhs>(rhs) };
}

template <detail::matrix_expression Lhs, detail::matrix_expression Rhs>
    requires detail::same_matrix_dimensions<Lhs, Rhs>
constexpr auto sub(Lhs&& lhs, Rhs&& rhs)
{
    return detail::matrix_binary_expr<detail::sub_op, Lhs, Rhs>{ std::forward<Lhs>(lhs), std::forward<Rhs>(rhs) };
}

template <detail::matrix_expression Lhs, detail::matrix_expression Rhs>
    requires detail::same_matrix_dimensions<Lhs, Rhs>
constexpr auto mul(Lhs&& lhs, Rhs&& rhs)
{
    return detail::matrix_binary_expr<detail::mul_op, Lhs, Rhs>{ std::forward<Lhs>(lhs), std::forward<Rhs>(rhs) };
}

template <detail::matrix_expression Lhs, detail::matrix_expression Rhs>
    requires detail::same_matrix_dimensions<Lhs, Rhs>
constexpr auto div(Lhs&& lhs, Rhs&& rhs)
{
    return detail::matrix_binary_expr<detail::div_op, Lhs, Rhs>{ std::forward<Lhs>(lhs), std::forward<Rhs>(rhs) };
}

template <detail::matrix_expression Expr>
constexpr auto neg(Expr&& expr)
{
    return detail::matrix_neg_expr<Expr>{ std::forward<Expr>(expr) };
}

template <detail::matrix_expression Expr, class Scalar>
    requires detail::scalar_for<Scalar, detail::expr_value_t<Expr>>
constexpr auto add(Expr&& expr, Scalar scalar)
{
    return detail::matrix_scalar_expr<detail::add_op, false, Scalar, Expr>{ scalar, std::forward<Expr>(expr) };
}

template <class Scalar, detail::matrix_expression Expr>
    requires detail::scalar_for<Scalar, detail::expr_value_t<Expr>>
constexpr auto add(Scalar scalar, Expr&& expr)
{
    return detail::matrix_scalar_expr<detail::add_op, true, Scalar, Expr>{ scalar, std::forward<Expr>(expr) };
}

template <detail::matrix_expression Expr, class Scalar>
    requires detail::scalar_for<Scalar, detail::expr_value_t<Expr>>
constexpr auto sub(Expr&& expr, Scalar scalar)
{
    return detail::matrix_scalar_expr<detail::sub_op, false, Scalar, Expr>{ scalar, std::forward<Expr>(expr) };
}

template <class Scalar, detail::matrix_expression Expr>
    requires detail::scalar_for<Scalar, detail::expr_value_t<Expr>>
constexpr auto sub(Scalar scalar, Expr&& expr)
{
    return detail::matrix_scalar_expr<detail::sub_op, true, Scalar, Expr>{ scalar, std::forward<Expr>(expr) };
}

template <detail::matrix_expression Expr, class Scalar>
    requires detail::scalar_for<Scalar, detail::expr_value_t<Expr>>
constexpr auto mul(Expr&& expr, Scalar scalar)
{
    return detail::matrix_scalar_expr<detail::mul_op, false, Scalar, Expr>{ scalar, std::forward<Expr>(expr) };
}

template <class Scalar, detail::matrix_expression Expr>
    requires detail::scalar_for<Scalar, detail::expr_value_t<Expr>>
constexpr auto mul(Scalar scalar, Expr&& expr)
{
    return detail::matrix_scalar_expr<detail::mul_op, true, Scalar, Expr>{ scalar, std::forward<Expr>(expr) };
}

template <detail::matrix_expression Expr, class Scalar>
    requires detail::scalar_for<Scalar, detail::expr_value_t<Expr>>
constexpr auto div(Expr&& expr, Scalar scalar)
{
    return detail::matrix_scalar_expr<detail::div_op, false, Scalar, Expr>{ scalar, std::forward<Expr>(expr) };
}

template <class Scalar, detail::matrix_expression Expr>
    requires detail::scalar_for<Scalar, detail::expr_value_t<Expr>>
constexpr auto div(Scalar scalar, Expr&& expr)
{
    return detail::matrix_scalar_expr<detail::div_op, true, Scalar, Expr>{ scalar, std::forward<Expr>(expr) };
}

template <class T, std::size_t N, matrix_layout Layout = matrix_layout::column_major>
constexpr auto identity() noexcept
{
    matrix<T, N, N, Layout> result{};
    for (std::size_t i = 0; i < N; ++i)
        result(i, i) = T{ 1 };
    return result;
}

template <detail::matrix_expression Expr>
constexpr auto transpose(const Expr& expr) noexcept
{
    using traits = detail::expr_traits<std::remove_cvref_t<Expr>>;
    matrix<typename traits::value_type, traits::cols, traits::rows, traits::layout> result{};

    for (std::size_t row = 0; row < traits::rows; ++row)
        for (std::size_t col = 0; col < traits::cols; ++col)
            result(col, row) = expr(row, col);
    return result;
}

template <detail::matrix_expression Lhs, detail::matrix_expression Rhs>
    requires (detail::expr_traits<std::remove_cvref_t<Lhs>>::cols == detail::expr_traits<std::remove_cvref_t<Rhs>>::rows)
constexpr auto matmul(const Lhs& lhs, const Rhs& rhs) noexcept
{
    using lhs_traits = detail::expr_traits<std::remove_cvref_t<Lhs>>;
    using rhs_traits = detail::expr_traits<std::remove_cvref_t<Rhs>>;
    using value_type = std::common_type_t<typename lhs_traits::value_type, typename rhs_traits::value_type>;

    matrix<value_type, lhs_traits::rows, rhs_traits::cols, lhs_traits::layout> result{};
    for (std::size_t row = 0; row < lhs_traits::rows; ++row)
    {
        for (std::size_t col = 0; col < rhs_traits::cols; ++col)
        {
            value_type sum{};
            for (std::size_t k = 0; k < lhs_traits::cols; ++k)
                sum += static_cast<value_type>(lhs(row, k)) * static_cast<value_type>(rhs(k, col));
            result(row, col) = sum;
        }
    }
    return result;
}

template <detail::matrix_expression Mat, detail::vector_expression Vec>
    requires (
        detail::expr_traits<std::remove_cvref_t<Mat>>::rows == 4 &&
        detail::expr_traits<std::remove_cvref_t<Mat>>::cols == 4 &&
        detail::expr_traits<std::remove_cvref_t<Vec>>::size == 3)
constexpr auto transform_vector(const Mat& mat, const Vec& vec) noexcept
{
    using value_type = std::common_type_t<detail::expr_value_t<Mat>, detail::expr_value_t<Vec>>;
    return vector<value_type, 3>{
        static_cast<value_type>(mat(0, 0) * vec[0] + mat(0, 1) * vec[1] + mat(0, 2) * vec[2]),
        static_cast<value_type>(mat(1, 0) * vec[0] + mat(1, 1) * vec[1] + mat(1, 2) * vec[2]),
        static_cast<value_type>(mat(2, 0) * vec[0] + mat(2, 1) * vec[1] + mat(2, 2) * vec[2])
    };
}

template <detail::matrix_expression Mat, detail::vector_expression Vec>
    requires (
        detail::expr_traits<std::remove_cvref_t<Mat>>::rows == 4 &&
        detail::expr_traits<std::remove_cvref_t<Mat>>::cols == 4 &&
        detail::expr_traits<std::remove_cvref_t<Vec>>::size == 3)
constexpr auto transform_point(const Mat& mat, const Vec& vec) noexcept
{
    using value_type = std::common_type_t<detail::expr_value_t<Mat>, detail::expr_value_t<Vec>>;

    vector<value_type, 3> result{
        static_cast<value_type>(mat(0, 0) * vec[0] + mat(0, 1) * vec[1] + mat(0, 2) * vec[2] + mat(0, 3)),
        static_cast<value_type>(mat(1, 0) * vec[0] + mat(1, 1) * vec[1] + mat(1, 2) * vec[2] + mat(1, 3)),
        static_cast<value_type>(mat(2, 0) * vec[0] + mat(2, 1) * vec[1] + mat(2, 2) * vec[2] + mat(2, 3))
    };

    const value_type w = static_cast<value_type>(
        mat(3, 0) * vec[0] + mat(3, 1) * vec[1] + mat(3, 2) * vec[2] + mat(3, 3));
    if (w != value_type{} && w != value_type{ 1 })
        result = div(result, w);
    return result;
}

template <detail::vector_expression Expr, matrix_layout Layout = matrix_layout::column_major>
    requires (detail::expr_traits<std::remove_cvref_t<Expr>>::size == 3)
constexpr auto translation(const Expr& offset) noexcept
{
    using value_type = detail::expr_value_t<Expr>;
    auto result = identity<value_type, 4, Layout>();
    result(0, 3) = offset[0];
    result(1, 3) = offset[1];
    result(2, 3) = offset[2];
    return result;
}

template <class T, matrix_layout Layout = matrix_layout::column_major>
constexpr auto translation(T x, T y, T z) noexcept
{
    return translation<vector<T, 3>, Layout>(vector<T, 3>{ x, y, z });
}

template <detail::vector_expression Expr, matrix_layout Layout = matrix_layout::column_major>
    requires (detail::expr_traits<std::remove_cvref_t<Expr>>::size == 3)
constexpr auto scaling(const Expr& scale) noexcept
{
    using value_type = detail::expr_value_t<Expr>;
    auto result = identity<value_type, 4, Layout>();
    result(0, 0) = scale[0];
    result(1, 1) = scale[1];
    result(2, 2) = scale[2];
    return result;
}

template <class T, matrix_layout Layout = matrix_layout::column_major>
constexpr auto scaling(T x, T y, T z) noexcept
{
    return scaling<vector<T, 3>, Layout>(vector<T, 3>{ x, y, z });
}

template <class T, matrix_layout Layout = matrix_layout::column_major>
inline auto rotation_x(T radians) noexcept
{
    auto result = identity<T, 4, Layout>();
    const T c = static_cast<T>(std::cos(radians));
    const T s = static_cast<T>(std::sin(radians));
    result(1, 1) = c;
    result(1, 2) = -s;
    result(2, 1) = s;
    result(2, 2) = c;
    return result;
}

template <class T, matrix_layout Layout = matrix_layout::column_major>
inline auto rotation_y(T radians) noexcept
{
    auto result = identity<T, 4, Layout>();
    const T c = static_cast<T>(std::cos(radians));
    const T s = static_cast<T>(std::sin(radians));
    result(0, 0) = c;
    result(0, 2) = s;
    result(2, 0) = -s;
    result(2, 2) = c;
    return result;
}

template <class T, matrix_layout Layout = matrix_layout::column_major>
inline auto rotation_z(T radians) noexcept
{
    auto result = identity<T, 4, Layout>();
    const T c = static_cast<T>(std::cos(radians));
    const T s = static_cast<T>(std::sin(radians));
    result(0, 0) = c;
    result(0, 1) = -s;
    result(1, 0) = s;
    result(1, 1) = c;
    return result;
}

template <detail::matrix_expression Lhs, detail::matrix_expression Rhs>
    requires detail::same_matrix_dimensions<Lhs, Rhs>
constexpr auto operator+(Lhs&& lhs, Rhs&& rhs)
{
    return add(std::forward<Lhs>(lhs), std::forward<Rhs>(rhs));
}

template <detail::matrix_expression Lhs, detail::matrix_expression Rhs>
    requires detail::same_matrix_dimensions<Lhs, Rhs>
constexpr auto operator-(Lhs&& lhs, Rhs&& rhs)
{
    return sub(std::forward<Lhs>(lhs), std::forward<Rhs>(rhs));
}

template <detail::matrix_expression Lhs, detail::matrix_expression Rhs>
    requires detail::same_matrix_dimensions<Lhs, Rhs>
constexpr auto operator*(Lhs&& lhs, Rhs&& rhs)
{
    return mul(std::forward<Lhs>(lhs), std::forward<Rhs>(rhs));
}

template <detail::matrix_expression Lhs, detail::matrix_expression Rhs>
    requires detail::same_matrix_dimensions<Lhs, Rhs>
constexpr auto operator/(Lhs&& lhs, Rhs&& rhs)
{
    return div(std::forward<Lhs>(lhs), std::forward<Rhs>(rhs));
}

template <detail::matrix_expression Expr>
constexpr auto operator-(Expr&& expr)
{
    return neg(std::forward<Expr>(expr));
}

template <detail::matrix_expression Expr, class Scalar>
    requires detail::scalar_for<Scalar, detail::expr_value_t<Expr>>
constexpr auto operator+(Expr&& expr, Scalar scalar)
{
    return add(std::forward<Expr>(expr), scalar);
}

template <class Scalar, detail::matrix_expression Expr>
    requires detail::scalar_for<Scalar, detail::expr_value_t<Expr>>
constexpr auto operator+(Scalar scalar, Expr&& expr)
{
    return add(scalar, std::forward<Expr>(expr));
}

template <detail::matrix_expression Expr, class Scalar>
    requires detail::scalar_for<Scalar, detail::expr_value_t<Expr>>
constexpr auto operator-(Expr&& expr, Scalar scalar)
{
    return sub(std::forward<Expr>(expr), scalar);
}

template <class Scalar, detail::matrix_expression Expr>
    requires detail::scalar_for<Scalar, detail::expr_value_t<Expr>>
constexpr auto operator-(Scalar scalar, Expr&& expr)
{
    return sub(scalar, std::forward<Expr>(expr));
}

template <detail::matrix_expression Expr, class Scalar>
    requires detail::scalar_for<Scalar, detail::expr_value_t<Expr>>
constexpr auto operator*(Expr&& expr, Scalar scalar)
{
    return mul(std::forward<Expr>(expr), scalar);
}

template <class Scalar, detail::matrix_expression Expr>
    requires detail::scalar_for<Scalar, detail::expr_value_t<Expr>>
constexpr auto operator*(Scalar scalar, Expr&& expr)
{
    return mul(scalar, std::forward<Expr>(expr));
}

template <detail::matrix_expression Expr, class Scalar>
    requires detail::scalar_for<Scalar, detail::expr_value_t<Expr>>
constexpr auto operator/(Expr&& expr, Scalar scalar)
{
    return div(std::forward<Expr>(expr), scalar);
}

template <class Scalar, detail::matrix_expression Expr>
    requires detail::scalar_for<Scalar, detail::expr_value_t<Expr>>
constexpr auto operator/(Scalar scalar, Expr&& expr)
{
    return div(scalar, std::forward<Expr>(expr));
}

} // namespace arc::math
