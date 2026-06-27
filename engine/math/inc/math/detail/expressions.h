#pragma once

#include <array>
#include <cmath>
#include <concepts>
#include <cstddef>
#include <initializer_list>
#include <stdexcept>
#include <type_traits>
#include <utility>

namespace arc::math
{

enum class matrix_layout
{
    row_major,
    column_major
};

template <class T, std::size_t N>
class vector;

template <class T, std::size_t Rows, std::size_t Cols, matrix_layout Layout>
class matrix;

template <class T>
class quaternion;

namespace detail
{

template <class Derived>
class vector_expr
{
public:
    constexpr const Derived& derived() const noexcept
    {
        return static_cast<const Derived&>(*this);
    }

    constexpr decltype(auto) operator[](std::size_t index) const noexcept
    {
        return derived()[index];
    }
};

template <class Derived>
class matrix_expr
{
public:
    constexpr const Derived& derived() const noexcept
    {
        return static_cast<const Derived&>(*this);
    }

    constexpr decltype(auto) operator()(std::size_t row, std::size_t col) const noexcept
    {
        return derived()(row, col);
    }
};

template <class Derived>
class quaternion_expr
{
public:
    constexpr const Derived& derived() const noexcept
    {
        return static_cast<const Derived&>(*this);
    }

    constexpr decltype(auto) operator[](std::size_t index) const noexcept
    {
        return derived()[index];
    }
};

template <class T>
struct expr_traits;

template <class T>
using expr_value_t = typename expr_traits<std::remove_cvref_t<T>>::value_type;

template <class T>
concept vector_expression = requires
{
    typename expr_traits<std::remove_cvref_t<T>>::value_type;
    requires expr_traits<std::remove_cvref_t<T>>::kind == 0;
};

template <class T>
concept matrix_expression = requires
{
    typename expr_traits<std::remove_cvref_t<T>>::value_type;
    requires expr_traits<std::remove_cvref_t<T>>::kind == 1;
};

template <class T>
concept quaternion_expression = requires
{
    typename expr_traits<std::remove_cvref_t<T>>::value_type;
    requires expr_traits<std::remove_cvref_t<T>>::kind == 2;
};

template <class T, class Value>
concept scalar_for =
    std::convertible_to<T, Value> &&
    !vector_expression<T> &&
    !matrix_expression<T> &&
    !quaternion_expression<T>;

template <class T>
using operand_storage_t = std::conditional_t<
    std::is_lvalue_reference_v<T>,
    std::add_lvalue_reference_t<const std::remove_reference_t<T>>,
    std::remove_cvref_t<T>>;

template <class T>
constexpr operand_storage_t<T&&> store_operand(T&& value)
{
    return std::forward<T>(value);
}

struct add_op
{
    template <class A, class B>
    static constexpr auto apply(const A& a, const B& b) noexcept
    {
        return a + b;
    }
};

struct sub_op
{
    template <class A, class B>
    static constexpr auto apply(const A& a, const B& b) noexcept
    {
        return a - b;
    }
};

struct mul_op
{
    template <class A, class B>
    static constexpr auto apply(const A& a, const B& b) noexcept
    {
        return a * b;
    }
};

struct div_op
{
    template <class A, class B>
    static constexpr auto apply(const A& a, const B& b) noexcept
    {
        return a / b;
    }
};

template <class Op, class Lhs, class Rhs, class Base>
class binary_expr : public Base
{
public:
    using lhs_type = std::remove_cvref_t<Lhs>;
    using rhs_type = std::remove_cvref_t<Rhs>;
    using op_type = Op;
    using value_type = std::common_type_t<expr_value_t<lhs_type>, expr_value_t<rhs_type>>;

    constexpr binary_expr(Lhs&& lhs, Rhs&& rhs)
        : lhs_(store_operand(std::forward<Lhs>(lhs)))
        , rhs_(store_operand(std::forward<Rhs>(rhs)))
    {
    }

    constexpr const auto& lhs() const noexcept
    {
        return lhs_;
    }

    constexpr const auto& rhs() const noexcept
    {
        return rhs_;
    }

protected:
    operand_storage_t<Lhs&&> lhs_;
    operand_storage_t<Rhs&&> rhs_;
};

template <class Op, class Lhs, class Rhs>
class vector_binary_expr final
    : public binary_expr<Op, Lhs, Rhs, vector_expr<vector_binary_expr<Op, Lhs, Rhs>>>
{
    using base_type = binary_expr<Op, Lhs, Rhs, vector_expr<vector_binary_expr<Op, Lhs, Rhs>>>;

public:
    using value_type = typename base_type::value_type;
    static constexpr std::size_t size = expr_traits<typename base_type::lhs_type>::size;

    using base_type::base_type;

    constexpr value_type operator[](std::size_t index) const noexcept
    {
        return static_cast<value_type>(Op::apply(this->lhs_[index], this->rhs_[index]));
    }
};

template <class A, class B, class C>
class vector_fma_expr final : public vector_expr<vector_fma_expr<A, B, C>>
{
public:
    using a_type = std::remove_cvref_t<A>;
    using b_type = std::remove_cvref_t<B>;
    using c_type = std::remove_cvref_t<C>;
    using value_type = std::common_type_t<expr_value_t<a_type>, expr_value_t<b_type>, expr_value_t<c_type>>;

    static constexpr std::size_t size = expr_traits<a_type>::size;

    constexpr vector_fma_expr(A&& a, B&& b, C&& c)
        : a_(store_operand(std::forward<A>(a)))
        , b_(store_operand(std::forward<B>(b)))
        , c_(store_operand(std::forward<C>(c)))
    {
    }

    constexpr const auto& a() const noexcept
    {
        return a_;
    }

    constexpr const auto& b() const noexcept
    {
        return b_;
    }

    constexpr const auto& c() const noexcept
    {
        return c_;
    }

    constexpr value_type operator[](std::size_t index) const noexcept
    {
        return static_cast<value_type>(std::fma(a_[index], b_[index], c_[index]));
    }

private:
    operand_storage_t<A&&> a_;
    operand_storage_t<B&&> b_;
    operand_storage_t<C&&> c_;
};

template <class Op, class Lhs, class Rhs>
class matrix_binary_expr final
    : public binary_expr<Op, Lhs, Rhs, matrix_expr<matrix_binary_expr<Op, Lhs, Rhs>>>
{
    using base_type = binary_expr<Op, Lhs, Rhs, matrix_expr<matrix_binary_expr<Op, Lhs, Rhs>>>;

public:
    using value_type = typename base_type::value_type;
    static constexpr std::size_t rows = expr_traits<typename base_type::lhs_type>::rows;
    static constexpr std::size_t cols = expr_traits<typename base_type::lhs_type>::cols;
    static constexpr matrix_layout layout = expr_traits<typename base_type::lhs_type>::layout;

    using base_type::base_type;

    constexpr value_type operator()(std::size_t row, std::size_t col) const noexcept
    {
        return static_cast<value_type>(Op::apply(this->lhs_(row, col), this->rhs_(row, col)));
    }
};

template <class Op, class Lhs, class Rhs>
class quaternion_binary_expr final
    : public binary_expr<Op, Lhs, Rhs, quaternion_expr<quaternion_binary_expr<Op, Lhs, Rhs>>>
{
    using base_type = binary_expr<Op, Lhs, Rhs, quaternion_expr<quaternion_binary_expr<Op, Lhs, Rhs>>>;

public:
    using value_type = typename base_type::value_type;
    static constexpr std::size_t size = 4;

    using base_type::base_type;

    constexpr value_type operator[](std::size_t index) const noexcept
    {
        return static_cast<value_type>(Op::apply(this->lhs_[index], this->rhs_[index]));
    }
};

template <class Expr, class Base>
class neg_expr_base : public Base
{
public:
    using expr_type = std::remove_cvref_t<Expr>;
    using value_type = expr_value_t<expr_type>;

    constexpr explicit neg_expr_base(Expr&& expr)
        : expr_(store_operand(std::forward<Expr>(expr)))
    {
    }

    constexpr const auto& expr() const noexcept
    {
        return expr_;
    }

protected:
    operand_storage_t<Expr&&> expr_;
};

template <class Expr>
class vector_neg_expr final : public neg_expr_base<Expr, vector_expr<vector_neg_expr<Expr>>>
{
    using base_type = neg_expr_base<Expr, vector_expr<vector_neg_expr<Expr>>>;

public:
    using value_type = typename base_type::value_type;
    static constexpr std::size_t size = expr_traits<typename base_type::expr_type>::size;

    using base_type::base_type;

    constexpr value_type operator[](std::size_t index) const noexcept
    {
        return -static_cast<value_type>(this->expr_[index]);
    }
};

template <class Expr>
class matrix_neg_expr final : public neg_expr_base<Expr, matrix_expr<matrix_neg_expr<Expr>>>
{
    using base_type = neg_expr_base<Expr, matrix_expr<matrix_neg_expr<Expr>>>;

public:
    using value_type = typename base_type::value_type;
    static constexpr std::size_t rows = expr_traits<typename base_type::expr_type>::rows;
    static constexpr std::size_t cols = expr_traits<typename base_type::expr_type>::cols;
    static constexpr matrix_layout layout = expr_traits<typename base_type::expr_type>::layout;

    using base_type::base_type;

    constexpr value_type operator()(std::size_t row, std::size_t col) const noexcept
    {
        return -static_cast<value_type>(this->expr_(row, col));
    }
};

template <class Expr>
class quaternion_neg_expr final : public neg_expr_base<Expr, quaternion_expr<quaternion_neg_expr<Expr>>>
{
    using base_type = neg_expr_base<Expr, quaternion_expr<quaternion_neg_expr<Expr>>>;

public:
    using value_type = typename base_type::value_type;
    static constexpr std::size_t size = 4;

    using base_type::base_type;

    constexpr value_type operator[](std::size_t index) const noexcept
    {
        return -static_cast<value_type>(this->expr_[index]);
    }
};

template <class Scalar, class Expr, class Base>
class scalar_binary_expr : public Base
{
public:
    using scalar_type = std::remove_cvref_t<Scalar>;
    using expr_type = std::remove_cvref_t<Expr>;
    using value_type = std::common_type_t<scalar_type, expr_value_t<expr_type>>;

    constexpr scalar_binary_expr(Scalar scalar, Expr&& expr)
        : scalar_(static_cast<value_type>(scalar))
        , expr_(store_operand(std::forward<Expr>(expr)))
    {
    }

protected:
    value_type scalar_;
    operand_storage_t<Expr&&> expr_;
};

template <class Op, bool ScalarOnLeft, class Scalar, class Expr>
class vector_scalar_expr final
    : public scalar_binary_expr<Scalar, Expr, vector_expr<vector_scalar_expr<Op, ScalarOnLeft, Scalar, Expr>>>
{
    using base_type = scalar_binary_expr<Scalar, Expr, vector_expr<vector_scalar_expr<Op, ScalarOnLeft, Scalar, Expr>>>;

public:
    using value_type = typename base_type::value_type;
    static constexpr std::size_t size = expr_traits<typename base_type::expr_type>::size;

    using base_type::base_type;

    constexpr value_type operator[](std::size_t index) const noexcept
    {
        if constexpr (ScalarOnLeft)
            return static_cast<value_type>(Op::apply(this->scalar_, this->expr_[index]));
        else
            return static_cast<value_type>(Op::apply(this->expr_[index], this->scalar_));
    }
};

template <class Op, bool ScalarOnLeft, class Scalar, class Expr>
class matrix_scalar_expr final
    : public scalar_binary_expr<Scalar, Expr, matrix_expr<matrix_scalar_expr<Op, ScalarOnLeft, Scalar, Expr>>>
{
    using base_type = scalar_binary_expr<Scalar, Expr, matrix_expr<matrix_scalar_expr<Op, ScalarOnLeft, Scalar, Expr>>>;

public:
    using value_type = typename base_type::value_type;
    static constexpr std::size_t rows = expr_traits<typename base_type::expr_type>::rows;
    static constexpr std::size_t cols = expr_traits<typename base_type::expr_type>::cols;
    static constexpr matrix_layout layout = expr_traits<typename base_type::expr_type>::layout;

    using base_type::base_type;

    constexpr value_type operator()(std::size_t row, std::size_t col) const noexcept
    {
        if constexpr (ScalarOnLeft)
            return static_cast<value_type>(Op::apply(this->scalar_, this->expr_(row, col)));
        else
            return static_cast<value_type>(Op::apply(this->expr_(row, col), this->scalar_));
    }
};

template <class Op, bool ScalarOnLeft, class Scalar, class Expr>
class quaternion_scalar_expr final
    : public scalar_binary_expr<Scalar, Expr, quaternion_expr<quaternion_scalar_expr<Op, ScalarOnLeft, Scalar, Expr>>>
{
    using base_type = scalar_binary_expr<Scalar, Expr, quaternion_expr<quaternion_scalar_expr<Op, ScalarOnLeft, Scalar, Expr>>>;

public:
    using value_type = typename base_type::value_type;
    static constexpr std::size_t size = 4;

    using base_type::base_type;

    constexpr value_type operator[](std::size_t index) const noexcept
    {
        if constexpr (ScalarOnLeft)
            return static_cast<value_type>(Op::apply(this->scalar_, this->expr_[index]));
        else
            return static_cast<value_type>(Op::apply(this->expr_[index], this->scalar_));
    }
};

template <class T>
struct is_mul_expr : std::false_type
{
};

template <class Lhs, class Rhs>
struct is_mul_expr<vector_binary_expr<mul_op, Lhs, Rhs>> : std::true_type
{
};

template <class T>
inline constexpr bool is_mul_expr_v = is_mul_expr<std::remove_cvref_t<T>>::value;

template <class T>
struct is_vector_fma_expr : std::false_type
{
};

template <class A, class B, class C>
struct is_vector_fma_expr<vector_fma_expr<A, B, C>> : std::true_type
{
};

template <class T>
inline constexpr bool is_vector_fma_expr_v = is_vector_fma_expr<std::remove_cvref_t<T>>::value;

template <class T>
struct is_vector_neg_expr : std::false_type
{
};

template <class Expr>
struct is_vector_neg_expr<vector_neg_expr<Expr>> : std::true_type
{
};

template <class T>
inline constexpr bool is_vector_neg_expr_v = is_vector_neg_expr<std::remove_cvref_t<T>>::value;

template <class T>
concept binary_expression_node = requires(const T& value)
{
    typename std::remove_cvref_t<T>::op_type;
    value.lhs();
    value.rhs();
};

template <class Expr, bool IsBinary = binary_expression_node<Expr>>
struct is_implicit_fma_expr : std::false_type
{
};

template <class Expr>
struct is_implicit_fma_expr<Expr, true>
    : std::bool_constant<
        std::same_as<typename std::remove_cvref_t<Expr>::op_type, add_op> &&
        (
            is_mul_expr_v<decltype(std::declval<const Expr&>().lhs())> ||
            is_mul_expr_v<decltype(std::declval<const Expr&>().rhs())>
        )>
{
};

template <class Expr>
inline constexpr bool is_implicit_fma_expr_v = is_implicit_fma_expr<Expr>::value;

template <class Expr>
constexpr bool can_fma_vector_expr =
    std::is_floating_point_v<expr_value_t<Expr>> &&
    (is_vector_fma_expr_v<Expr> || is_implicit_fma_expr_v<Expr>);

template <class Value, class Expr>
constexpr Value eval_vector_element(const Expr& expr, std::size_t index) noexcept
{
    if constexpr (can_fma_vector_expr<Expr>)
    {
        if constexpr (is_vector_fma_expr_v<Expr>)
        {
            return static_cast<Value>(std::fma(expr.a()[index], expr.b()[index], expr.c()[index]));
        }
        else if constexpr (is_mul_expr_v<decltype(expr.lhs())>)
        {
            const auto& product = expr.lhs();
            return static_cast<Value>(std::fma(product.lhs()[index], product.rhs()[index], expr.rhs()[index]));
        }
        else
        {
            const auto& product = expr.rhs();
            return static_cast<Value>(std::fma(product.lhs()[index], product.rhs()[index], expr.lhs()[index]));
        }
    }
    else
    {
        return static_cast<Value>(expr[index]);
    }
}

template <class Lhs, class Rhs>
constexpr bool same_vector_dimensions =
    expr_traits<std::remove_cvref_t<Lhs>>::size == expr_traits<std::remove_cvref_t<Rhs>>::size;

template <class Lhs, class Rhs>
constexpr bool same_matrix_dimensions =
    expr_traits<std::remove_cvref_t<Lhs>>::rows == expr_traits<std::remove_cvref_t<Rhs>>::rows &&
    expr_traits<std::remove_cvref_t<Lhs>>::cols == expr_traits<std::remove_cvref_t<Rhs>>::cols;

} // namespace detail
} // namespace arc::math
