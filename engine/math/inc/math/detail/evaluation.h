#pragma once

#include <arc/simd.h>

#include <math/detail/expressions.h>

namespace arc::math::detail
{

template <class T, std::size_t N>
concept vector_simd_exact_register_available =
    requires
    {
        typename arc::simd_register<T, N>::type;
    };

template <class T, std::size_t N, bool ExactRegister = vector_simd_exact_register_available<T, N>>
struct vector_simd_io_available_impl : std::false_type
{
};

template <class T, std::size_t N>
struct vector_simd_io_available_impl<T, N, true>
    : std::bool_constant<requires(const T* input, T* output, arc::simd<T, N> value)
    {
        { arc::load_unaligned<T, N>(input) } -> std::same_as<arc::simd<T, N>>;
        arc::store_unaligned<T, N>(output, value);
    }>
{
};

template <class T, std::size_t N>
constexpr bool vector_simd_io_available = vector_simd_io_available_impl<T, N>::value;

template <class Op, class T, std::size_t N, bool Available = vector_simd_io_available<T, N>>
struct vector_simd_binary
{
    static constexpr bool available = false;
};

template <class T, std::size_t N>
struct vector_simd_binary<add_op, T, N, true>
{
    static constexpr bool available = requires(arc::simd<T, N> lhs, arc::simd<T, N> rhs)
        {
            { arc::add(lhs, rhs) } -> std::same_as<arc::simd<T, N>>;
        };

    static auto apply(const arc::simd<T, N>& lhs, const arc::simd<T, N>& rhs) noexcept
    {
        return arc::add(lhs, rhs);
    }
};

template <class T, std::size_t N>
struct vector_simd_binary<sub_op, T, N, true>
{
    static constexpr bool available = requires(arc::simd<T, N> lhs, arc::simd<T, N> rhs)
        {
            { arc::sub(lhs, rhs) } -> std::same_as<arc::simd<T, N>>;
        };

    static auto apply(const arc::simd<T, N>& lhs, const arc::simd<T, N>& rhs) noexcept
    {
        return arc::sub(lhs, rhs);
    }
};

template <class T, std::size_t N>
struct vector_simd_binary<mul_op, T, N, true>
{
    static constexpr bool available = requires(arc::simd<T, N> lhs, arc::simd<T, N> rhs)
        {
            { arc::mul(lhs, rhs) } -> std::same_as<arc::simd<T, N>>;
        };

    static auto apply(const arc::simd<T, N>& lhs, const arc::simd<T, N>& rhs) noexcept
    {
        return arc::mul(lhs, rhs);
    }
};

template <class T, std::size_t N>
struct vector_simd_binary<div_op, T, N, true>
{
    static constexpr bool available = requires(arc::simd<T, N> lhs, arc::simd<T, N> rhs)
        {
            { arc::div(lhs, rhs) } -> std::same_as<arc::simd<T, N>>;
        };

    static auto apply(const arc::simd<T, N>& lhs, const arc::simd<T, N>& rhs) noexcept
    {
        return arc::div(lhs, rhs);
    }
};

template <class T, std::size_t N, bool Available = vector_simd_io_available<T, N>>
struct vector_simd_neg_available_impl : std::false_type
{
};

template <class T, std::size_t N>
struct vector_simd_neg_available_impl<T, N, true>
    : std::bool_constant<requires(arc::simd<T, N> value)
    {
        { arc::neg(value) } -> std::same_as<arc::simd<T, N>>;
    }>
{
};

template <class T, std::size_t N>
constexpr bool vector_simd_neg_available = vector_simd_neg_available_impl<T, N>::value;

template <class T, std::size_t N, bool Available = vector_simd_io_available<T, N>>
struct vector_simd_fma_available_impl : std::false_type
{
};

template <class T, std::size_t N>
struct vector_simd_fma_available_impl<T, N, true>
    : std::bool_constant<std::is_floating_point_v<T> && requires(arc::simd<T, N> a, arc::simd<T, N> b, arc::simd<T, N> c)
    {
        { arc::fma(a, b, c) } -> std::same_as<arc::simd<T, N>>;
    }>
{
};

template <class T, std::size_t N>
constexpr bool vector_simd_fma_available = vector_simd_fma_available_impl<T, N>::value;

template <class T, std::size_t N, vector_expression Expr>
constexpr void assign_vector_scalar(std::array<T, N>& output, const Expr& expr) noexcept
{
    for (std::size_t i = 0; i < N; ++i)
        output[i] = eval_vector_element<T>(expr, i);
}

template <class T, std::size_t N, vector_expression Expr>
constexpr void fill_vector_array(std::array<T, N>& output, const Expr& expr) noexcept
{
    for (std::size_t i = 0; i < N; ++i)
        output[i] = eval_vector_element<T>(expr, i);
}

template <class T, std::size_t N, class Expr, bool IsBinary = binary_expression_node<Expr>>
struct can_assign_vector_simd_binary_impl : std::false_type
{
};

template <class T, std::size_t N, class Expr>
struct can_assign_vector_simd_binary_impl<T, N, Expr, true>
    : std::bool_constant<vector_simd_binary<typename std::remove_cvref_t<Expr>::op_type, T, N>::available>
{
};

template <class T, std::size_t N, vector_expression Expr>
constexpr bool can_assign_vector_simd_binary = can_assign_vector_simd_binary_impl<T, N, Expr>::value;

template <class T, std::size_t N, vector_expression Expr>
constexpr void assign_vector_simd_binary(std::array<T, N>& output, const Expr& expr) noexcept
{
    using op_type = typename std::remove_cvref_t<Expr>::op_type;

    std::array<T, N> lhs{};
    std::array<T, N> rhs{};

    fill_vector_array(lhs, expr.lhs());
    fill_vector_array(rhs, expr.rhs());

    arc::store_unaligned<T, N>(
        output.data(),
        vector_simd_binary<op_type, T, N>::apply(
            arc::load_unaligned<T, N>(lhs.data()),
            arc::load_unaligned<T, N>(rhs.data())));
}

template <class T, std::size_t N, vector_expression Expr>
constexpr bool can_assign_vector_simd_neg =
    is_vector_neg_expr_v<Expr> &&
    vector_simd_neg_available<T, N>;

template <class T, std::size_t N, vector_expression Expr>
constexpr void assign_vector_simd_neg(std::array<T, N>& output, const Expr& expr) noexcept
{
    std::array<T, N> input{};
    fill_vector_array(input, expr.expr());

    arc::store_unaligned<T, N>(
        output.data(),
        arc::neg(arc::load_unaligned<T, N>(input.data())));
}

template <class T, std::size_t N, vector_expression Expr>
constexpr bool can_assign_vector_simd_fma =
    can_fma_vector_expr<Expr> &&
    vector_simd_fma_available<T, N>;

template <class T, std::size_t N, vector_expression Expr>
constexpr void assign_vector_simd_fma(std::array<T, N>& output, const Expr& expr) noexcept
{
    std::array<T, N> a{};
    std::array<T, N> b{};
    std::array<T, N> c{};

    if constexpr (is_vector_fma_expr_v<Expr>)
    {
        fill_vector_array(a, expr.a());
        fill_vector_array(b, expr.b());
        fill_vector_array(c, expr.c());
    }
    else if constexpr (is_mul_expr_v<decltype(expr.lhs())>)
    {
        const auto& product = expr.lhs();
        fill_vector_array(a, product.lhs());
        fill_vector_array(b, product.rhs());
        fill_vector_array(c, expr.rhs());
    }
    else
    {
        const auto& product = expr.rhs();
        fill_vector_array(a, product.lhs());
        fill_vector_array(b, product.rhs());
        fill_vector_array(c, expr.lhs());
    }

    arc::store_unaligned<T, N>(
        output.data(),
        arc::fma(
            arc::load_unaligned<T, N>(a.data()),
            arc::load_unaligned<T, N>(b.data()),
            arc::load_unaligned<T, N>(c.data())));
}

template <class T, std::size_t N, vector_expression Expr>
constexpr void assign_vector(std::array<T, N>& output, const Expr& expr) noexcept
{
    if (!std::is_constant_evaluated())
    {
        if constexpr (can_assign_vector_simd_fma<T, N, Expr>)
        {
            assign_vector_simd_fma(output, expr);
            return;
        }
        else if constexpr (can_assign_vector_simd_binary<T, N, Expr>)
        {
            assign_vector_simd_binary(output, expr);
            return;
        }
        else if constexpr (can_assign_vector_simd_neg<T, N, Expr>)
        {
            assign_vector_simd_neg(output, expr);
            return;
        }
    }

    assign_vector_scalar(output, expr);
}

template <class T, std::size_t Rows, std::size_t Cols, matrix_expression Expr, class Element>
constexpr void assign_matrix(Element& target, const Expr& expr) noexcept
{
    for (std::size_t row = 0; row < Rows; ++row)
        for (std::size_t col = 0; col < Cols; ++col)
            target(row, col) = static_cast<T>(expr(row, col));
}

template <class T, quaternion_expression Expr>
constexpr void assign_quaternion(std::array<T, 4>& output, const Expr& expr) noexcept
{
    for (std::size_t i = 0; i < 4; ++i)
        output[i] = static_cast<T>(expr[i]);
}

} // namespace arc::math::detail
