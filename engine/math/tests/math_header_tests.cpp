#include <math/vector.h>
#include <math/matrix.h>
#include <math/quaternion.h>
#include <math/math.h>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <cmath>
#include <type_traits>

namespace
{

constexpr bool constexpr_vector_expression_works()
{
    arc::math::vector<int, 2> a{ 1, 2 };
    arc::math::vector<int, 2> b{ 3, 4 };
    const auto result = arc::math::eval(arc::math::add(arc::math::mul(a, b), 1));

    return result[0] == 4 && result[1] == 9;
}

static_assert(constexpr_vector_expression_works());

} // namespace

TEST_CASE("vector supports construction indexing and lazy arithmetic")
{
    using namespace arc::math;

    vector<float, 4> a{ 1.0f, 2.0f, 3.0f, 4.0f };
    vector<float, 4> b{ 2.0f, 3.0f, 4.0f, 5.0f };
    vector<float, 4> c{ 10.0f, 20.0f, 30.0f, 40.0f };

    const auto expr = add(mul(a, b), c);
    static_assert(!std::is_same_v<decltype(expr), vector<float, 4>>);
    static_assert(std::is_same_v<decltype(a + b), decltype(add(a, b))>);
    static_assert(std::is_same_v<decltype(a * 2.0f), decltype(mul(a, 2.0f))>);

    const vector<float, 4> result = expr;

    REQUIRE(result[0] == 12.0f);
    REQUIRE(result[1] == 26.0f);
    REQUIRE(result[2] == 42.0f);
    REQUIRE(result[3] == 60.0f);

    const vector<float, 4> direct_add = add(a, b);
    const vector<float, 4> direct_sub = sub(c, b);
    const vector<float, 4> direct_mul = mul(a, b);
    const vector<float, 4> direct_div = div(c, b);

    REQUIRE(direct_add[2] == 7.0f);
    REQUIRE(direct_sub[0] == 8.0f);
    REQUIRE(direct_mul[3] == 20.0f);
    REQUIRE(direct_div[1] == 20.0f / 3.0f);

    const auto materialized = eval(add(c, mul(a, b)));
    REQUIRE(materialized[0] == 12.0f);
    REQUIRE(materialized[3] == 60.0f);

    const vector<float, 4> explicit_fma = fma(a, b, c);
    REQUIRE(explicit_fma[0] == 12.0f);
    REQUIRE(explicit_fma[3] == 60.0f);

    const vector<float, 4> negated = neg(a);
    REQUIRE(negated[0] == -1.0f);
    REQUIRE(negated[3] == -4.0f);

    const vector<float, 4> scaled = div(mul(add(a, 1.0f), 2.0f), 2.0f);
    REQUIRE(scaled[0] == 2.0f);
    REQUIRE(scaled[3] == 5.0f);

    const vector<float, 4> operator_result = a * b + c;
    REQUIRE(operator_result[1] == result[1]);

    result.data();
}

TEST_CASE("vector exposes game aliases and helpers")
{
    using namespace arc::math;

    static_assert(std::is_same_v<vector3f, vector<float, 3>>);
    static_assert(std::is_same_v<vector4d, vector<double, 4>>);
    static_assert(std::is_same_v<vector2i, vector<int, 2>>);

    vector3f x{ 1.0f, 0.0f, 0.0f };
    vector3f y{ 0.0f, 1.0f, 0.0f };
    vector3f v{ 3.0f, 4.0f, 0.0f };

    REQUIRE(dot(x, y) == 0.0f);
    REQUIRE(length_squared(v) == 25.0f);
    REQUIRE(length(v) == Catch::Approx(5.0f));

    const vector3f unit = normalize(v);
    REQUIRE(unit[0] == Catch::Approx(0.6f));
    REQUIRE(unit[1] == Catch::Approx(0.8f));

    const vector3f z = cross(x, y);
    REQUIRE(z[2] == Catch::Approx(1.0f));
    REQUIRE(cross2(vector2f{ 1.0f, 0.0f }, vector2f{ 0.0f, 1.0f }) == Catch::Approx(1.0f));

    const vector3f halfway = lerp(x, y, 0.5f);
    REQUIRE(halfway[0] == Catch::Approx(0.5f));
    REQUIRE(halfway[1] == Catch::Approx(0.5f));

    const vector3f clamped = clamp(vector3f{ -1.0f, 2.0f, 5.0f }, vector3f{ 0.0f, 0.0f, 0.0f }, vector3f{ 1.0f, 1.0f, 4.0f });
    REQUIRE(clamped[0] == 0.0f);
    REQUIRE(clamped[1] == 1.0f);
    REQUIRE(clamped[2] == 4.0f);

    const vector3f reflected = reflect(vector3f{ 1.0f, -1.0f, 0.0f }, vector3f{ 0.0f, 1.0f, 0.0f });
    REQUIRE(reflected[1] == Catch::Approx(1.0f));

    const vector3f projected = project(vector3f{ 2.0f, 2.0f, 0.0f }, x);
    REQUIRE(projected[0] == Catch::Approx(2.0f));
    REQUIRE(projected[1] == Catch::Approx(0.0f));
}

TEST_CASE("matrix supports layout aware storage and elementwise expressions")
{
    using namespace arc::math;

    matrix<float, 2> square(1.0f);
    static_assert(decltype(square)::rows == 2);
    static_assert(decltype(square)::cols == 2);
    REQUIRE(square(1, 1) == 1.0f);

    matrix<int, 2, 3, matrix_layout::row_major> row_major;
    matrix<int, 2, 3, matrix_layout::column_major> column_major;

    row_major(0, 0) = 1;
    row_major(0, 1) = 2;
    row_major(0, 2) = 3;
    row_major(1, 0) = 4;
    row_major(1, 1) = 5;
    row_major(1, 2) = 6;

    column_major(0, 0) = 1;
    column_major(0, 1) = 2;
    column_major(0, 2) = 3;
    column_major(1, 0) = 4;
    column_major(1, 1) = 5;
    column_major(1, 2) = 6;

    REQUIRE(row_major(1, 2) == 6);
    REQUIRE(column_major(1, 2) == 6);
    REQUIRE(row_major.data()[1] == 2);
    REQUIRE(column_major.data()[1] == 4);

    matrix<int, 2, 3, matrix_layout::row_major> doubled = row_major + row_major;
    REQUIRE(doubled(0, 0) == 2);
    REQUIRE(doubled(1, 2) == 12);

    const matrix<int, 2, 3, matrix_layout::row_major> via_functions = sub(mul(doubled, 2), row_major);
    const matrix<int, 2, 3, matrix_layout::row_major> via_operators = doubled * 2 - row_major;
    static_assert(std::is_same_v<decltype(doubled + row_major), decltype(add(doubled, row_major))>);

    doubled = via_functions;
    REQUIRE(doubled(0, 1) == 6);
    REQUIRE(via_operators(0, 1) == doubled(0, 1));

    const matrix<int, 2, 3, matrix_layout::row_major> negated = neg(row_major);
    REQUIRE(negated(1, 2) == -6);

    const matrix<int, 2, 3, matrix_layout::row_major> scalar_div = div(12, row_major);
    REQUIRE(scalar_div(0, 2) == 4);
}

TEST_CASE("matrix exposes aliases and game helpers")
{
    using namespace arc::math;

    static_assert(std::is_same_v<matrix4f, matrix<float, 4>>);
    static_assert(std::is_same_v<matrix4x4d, matrix<double, 4, 4>>);

    const matrix3f ident3 = identity<float, 3>();
    REQUIRE(ident3(0, 0) == 1.0f);
    REQUIRE(ident3(0, 1) == 0.0f);

    matrix<float, 2, 3, matrix_layout::row_major> a{};
    a(0, 0) = 1.0f;
    a(0, 1) = 2.0f;
    a(0, 2) = 3.0f;
    a(1, 0) = 4.0f;
    a(1, 1) = 5.0f;
    a(1, 2) = 6.0f;

    const auto transposed = transpose(a);
    REQUIRE(transposed(2, 1) == 6.0f);

    matrix<float, 3, 2, matrix_layout::row_major> b{};
    b(0, 0) = 7.0f;
    b(0, 1) = 8.0f;
    b(1, 0) = 9.0f;
    b(1, 1) = 10.0f;
    b(2, 0) = 11.0f;
    b(2, 1) = 12.0f;

    const auto product = matmul(a, b);
    REQUIRE(product(0, 0) == Catch::Approx(58.0f));
    REQUIRE(product(1, 1) == Catch::Approx(154.0f));

    const auto moved = translation(vector3f{ 10.0f, 20.0f, 30.0f });
    const vector3f point = transform_point(moved, vector3f{ 1.0f, 2.0f, 3.0f });
    const vector3f direction = transform_vector(moved, vector3f{ 1.0f, 2.0f, 3.0f });
    REQUIRE(point[0] == Catch::Approx(11.0f));
    REQUIRE(point[2] == Catch::Approx(33.0f));
    REQUIRE(direction[0] == Catch::Approx(1.0f));

    const auto scaled = scaling(2.0f, 3.0f, 4.0f);
    REQUIRE(transform_vector(scaled, vector3f{ 1.0f, 1.0f, 1.0f })[2] == Catch::Approx(4.0f));

    const auto rotated = rotation_z<float>(3.1415926535f / 2.0f);
    const vector3f rotated_x = transform_vector(rotated, vector3f{ 1.0f, 0.0f, 0.0f });
    REQUIRE(rotated_x[0] == Catch::Approx(0.0f).margin(0.00001f));
    REQUIRE(rotated_x[1] == Catch::Approx(1.0f).margin(0.00001f));
}

TEST_CASE("quaternion supports identity and expression assignment")
{
    using namespace arc::math;

    quaternion<float> identity;
    REQUIRE(identity.x() == 0.0f);
    REQUIRE(identity.y() == 0.0f);
    REQUIRE(identity.z() == 0.0f);
    REQUIRE(identity.w() == 1.0f);

    quaternion<float> a{ 1.0f, 2.0f, 3.0f, 4.0f };
    quaternion<float> b(2.0f);
    quaternion<float> result;

    result = add(a, mul(b, 2.0f));

    REQUIRE(result[0] == 5.0f);
    REQUIRE(result[1] == 6.0f);
    REQUIRE(result[2] == 7.0f);
    REQUIRE(result[3] == 8.0f);
    REQUIRE(result.data()[3] == 8.0f);

    const quaternion<float> operator_result = a + b * 2.0f;
    static_assert(std::is_same_v<decltype(a + b), decltype(add(a, b))>);

    REQUIRE(operator_result[0] == result[0]);
    REQUIRE(operator_result[3] == result[3]);

    const quaternion<float> negated = neg(a);
    REQUIRE(negated.w() == -4.0f);

    const quaternion<float> scalar = div(mul(a, 2.0f), 4.0f);
    REQUIRE(scalar[2] == 1.5f);
}

TEST_CASE("quaternion exposes aliases and game helpers")
{
    using namespace arc::math;

    static_assert(std::is_same_v<quaternionf, quaternion<float>>);
    static_assert(std::is_same_v<quatd, quaternion<double>>);

    const quatf q{ 0.0f, 0.0f, 1.0f, 0.0f };
    const quatf c = conjugate(q);
    REQUIRE(c.z() == -1.0f);
    REQUIRE(c.w() == 0.0f);

    const quatf n = normalize(quatf{ 0.0f, 0.0f, 0.0f, 2.0f });
    REQUIRE(n.w() == Catch::Approx(1.0f));

    const quatf inv = inverse(quatf{ 0.0f, 0.0f, 0.0f, 2.0f });
    REQUIRE(inv.w() == Catch::Approx(0.5f));

    const quatf z_turn = from_axis_angle(vector3f{ 0.0f, 0.0f, 1.0f }, 3.1415926535f / 2.0f);
    const vector3f rotated = rotate(z_turn, vector3f{ 1.0f, 0.0f, 0.0f });
    REQUIRE(rotated[0] == Catch::Approx(0.0f).margin(0.00001f));
    REQUIRE(rotated[1] == Catch::Approx(1.0f).margin(0.00001f));
}
