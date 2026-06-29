#define CATCH_CONFIG_MAIN

#include <catch2/catch_all.hpp>

#include <arc/simd/simd.h>

TEST_CASE("add", "[simd]")
{
    arc::simd<float, 4> a = arc::fill<float, 4>(1.0f);
    arc::simd<float, 4> b = arc::fill<float, 4>(2.0f);
    auto c = arc::add(a, b);
    
    float result[4];
    arc::store_aligned(result, c);
    
    REQUIRE(result[0] == 3.0f);
    REQUIRE(result[1] == 3.0f);
    REQUIRE(result[2] == 3.0f);
    REQUIRE(result[3] == 3.0f);
}

TEST_CASE("sub", "[simd]")
{
    arc::simd<float, 4> a = arc::fill<float, 4>(5.0f);
    arc::simd<float, 4> b = arc::fill<float, 4>(3.0f);
    auto c = arc::sub(a, b);
    
    float result[4];
    arc::store_aligned(result, c);
    
    REQUIRE(result[0] == 2.0f);
    REQUIRE(result[1] == 2.0f);
    REQUIRE(result[2] == 2.0f);
    REQUIRE(result[3] == 2.0f);
}

TEST_CASE("mul", "[simd]")
{
    arc::simd<float, 4> a = arc::fill<float, 4>(2.0f);
    arc::simd<float, 4> b = arc::fill<float, 4>(3.0f);
    auto c = arc::mul(a, b);
    
    float result[4];
    arc::store_aligned(result, c);
    
    REQUIRE(result[0] == 6.0f);
    REQUIRE(result[1] == 6.0f);
    REQUIRE(result[2] == 6.0f);
    REQUIRE(result[3] == 6.0f);
}

TEST_CASE("div", "[simd]")
{
    arc::simd<float, 4> a = arc::fill<float, 4>(10.0f);
    arc::simd<float, 4> b = arc::fill<float, 4>(2.0f);
    auto c = arc::div(a, b);
    
    float result[4];
    arc::store_aligned(result, c);
    
    REQUIRE(result[0] == 5.0f);
    REQUIRE(result[1] == 5.0f);
    REQUIRE(result[2] == 5.0f);
    REQUIRE(result[3] == 5.0f);
}

TEST_CASE("sum", "[simd]")
{
    float data[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    auto a = arc::load_aligned<float, 4>(data);
    float result = arc::sum(a);
    
    REQUIRE(result == 10.0f);
}

TEST_CASE("sum_negative", "[simd]")
{
    float data[4] = {-1.0f, -2.0f, -3.0f, -4.0f};
    auto a = arc::load_aligned<float, 4>(data);
    float result = arc::sum(a);
    
    REQUIRE(result == -10.0f);
}

TEST_CASE("sum_mixed", "[simd]")
{
    float data[4] = {1.5f, -2.5f, 3.0f, -1.0f};
    auto a = arc::load_aligned<float, 4>(data);
    float result = arc::sum(a);
    
    REQUIRE(result == 1.0f);
}

TEST_CASE("sum_zeros", "[simd]")
{
    float data[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    auto a = arc::load_aligned<float, 4>(data);
    float result = arc::sum(a);
    
    REQUIRE(result == 0.0f);
}

TEST_CASE("dot", "[simd]")
{
    float a_data[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    float b_data[4] = {2.0f, 3.0f, 4.0f, 5.0f};
    
    auto a = arc::load_aligned<float, 4>(a_data);
    auto b = arc::load_aligned<float, 4>(b_data);
    
    float result = arc::dot(a, b);
    
    // 1*2 + 2*3 + 3*4 + 4*5 = 2 + 6 + 12 + 20 = 40
    REQUIRE(result == 40.0f);
}

TEST_CASE("dot_negative", "[simd]")
{
    float a_data[4] = {1.0f, -2.0f, 3.0f, -4.0f};
    float b_data[4] = {2.0f, 3.0f, -4.0f, 5.0f};
    
    auto a = arc::load_aligned<float, 4>(a_data);
    auto b = arc::load_aligned<float, 4>(b_data);
    
    float result = arc::dot(a, b);
    
    // 1*2 + (-2)*3 + 3*(-4) + (-4)*5 = 2 - 6 - 12 - 20 = -36
    REQUIRE(result == -36.0f);
}

TEST_CASE("dot_zero", "[simd]")
{
    float a_data[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    float b_data[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    
    auto a = arc::load_aligned<float, 4>(a_data);
    auto b = arc::load_aligned<float, 4>(b_data);
    
    float result = arc::dot(a, b);
    
    REQUIRE(result == 0.0f);
}

TEST_CASE("dot_orthogonal", "[simd]")
{
    float a_data[4] = {1.0f, 0.0f, 0.0f, 0.0f};
    float b_data[4] = {0.0f, 1.0f, 0.0f, 0.0f};
    
    auto a = arc::load_aligned<float, 4>(a_data);
    auto b = arc::load_aligned<float, 4>(b_data);
    
    float result = arc::dot(a, b);
    
    REQUIRE(result == 0.0f);
}

TEST_CASE("sum_int", "[simd]")
{
    int32_t data[4] = {1, 2, 3, 4};
    auto a = arc::load_aligned<int32_t, 4>(data);
    int32_t result = arc::sum(a);
    
    REQUIRE(result == 10);
}

TEST_CASE("dot_int", "[simd]")
{
    int32_t a_data[4] = {1, 2, 3, 4};
    int32_t b_data[4] = {2, 3, 4, 5};
    
    auto a = arc::load_aligned<int32_t, 4>(a_data);
    auto b = arc::load_aligned<int32_t, 4>(b_data);
    
    int32_t result = arc::dot(a, b);
    
    // 1*2 + 2*3 + 3*4 + 4*5 = 2 + 6 + 12 + 20 = 40
    REQUIRE(result == 40);
}

TEST_CASE("sum_double", "[simd]")
{
    double data[2] = {1.5, 2.5};
    auto a = arc::load_aligned<double, 2>(data);
    double result = arc::sum(a);
    
    REQUIRE(result == 4.0);
}

TEST_CASE("dot_double", "[simd]")
{
    double a_data[2] = {1.0, 2.0};
    double b_data[2] = {3.0, 4.0};
    
    auto a = arc::load_aligned<double, 2>(a_data);
    auto b = arc::load_aligned<double, 2>(b_data);
    
    double result = arc::dot(a, b);
    
    // 1*3 + 2*4 = 3 + 8 = 11
    REQUIRE(result == 11.0);
}

TEST_CASE("fma", "[simd]")
{
    arc::simd<float, 4> a = arc::fill<float, 4>(2.0f);
    arc::simd<float, 4> b = arc::fill<float, 4>(3.0f);
    arc::simd<float, 4> c = arc::fill<float, 4>(1.0f);
    
    auto result_vec = arc::fma(a, b, c); // 2*3 + 1 = 7
    
    float result[4];
    arc::store_aligned(result, result_vec);
    
    REQUIRE(result[0] == 7.0f);
    REQUIRE(result[1] == 7.0f);
    REQUIRE(result[2] == 7.0f);
    REQUIRE(result[3] == 7.0f);
}

TEST_CASE("fma_mixed", "[simd]")
{
    float a_data[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    float b_data[4] = {2.0f, 3.0f, 4.0f, 5.0f};
    float c_data[4] = {0.5f, 1.0f, 1.5f, 2.0f};
    
    auto a = arc::load_aligned<float, 4>(a_data);
    auto b = arc::load_aligned<float, 4>(b_data);
    auto c = arc::load_aligned<float, 4>(c_data);
    
    auto result_vec = arc::fma(a, b, c);
    
    float result[4];
    arc::store_aligned(result, result_vec);
    
    // 1*2 + 0.5 = 2.5, 2*3 + 1 = 7, 3*4 + 1.5 = 13.5, 4*5 + 2 = 22
    REQUIRE(result[0] == 2.5f);
    REQUIRE(result[1] == 7.0f);
    REQUIRE(result[2] == 13.5f);
    REQUIRE(result[3] == 22.0f);
}

TEST_CASE("fma_double", "[simd]")
{
    arc::simd<double, 2> a = arc::fill<double, 2>(2.0);
    arc::simd<double, 2> b = arc::fill<double, 2>(3.0);
    arc::simd<double, 2> c = arc::fill<double, 2>(1.0);
    
    auto result_vec = arc::fma(a, b, c); // 2*3 + 1 = 7
    
    double result[2];
    arc::store_aligned(result, result_vec);
    
    REQUIRE(result[0] == 7.0);
    REQUIRE(result[1] == 7.0);
}

TEST_CASE("fma_zero_addend", "[simd]")
{
    arc::simd<float, 4> a = arc::fill<float, 4>(5.0f);
    arc::simd<float, 4> b = arc::fill<float, 4>(2.0f);
    arc::simd<float, 4> c = arc::fill<float, 4>(0.0f);
    
    auto result_vec = arc::fma(a, b, c); // 5*2 + 0 = 10
    
    float result[4];
    arc::store_aligned(result, result_vec);
    
    REQUIRE(result[0] == 10.0f);
    REQUIRE(result[1] == 10.0f);
    REQUIRE(result[2] == 10.0f);
    REQUIRE(result[3] == 10.0f);
}

TEST_CASE("fma_negative", "[simd]")
{
    arc::simd<float, 4> a = arc::fill<float, 4>(-2.0f);
    arc::simd<float, 4> b = arc::fill<float, 4>(3.0f);
    arc::simd<float, 4> c = arc::fill<float, 4>(5.0f);
    
    auto result_vec = arc::fma(a, b, c); // -2*3 + 5 = -6 + 5 = -1
    
    float result[4];
    arc::store_aligned(result, result_vec);
    
    REQUIRE(result[0] == -1.0f);
    REQUIRE(result[1] == -1.0f);
    REQUIRE(result[2] == -1.0f);
    REQUIRE(result[3] == -1.0f);
}