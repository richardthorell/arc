#define CATCH_CONFIG_MAIN

#include <catch2/catch_all.hpp>

#include <arc/simd.h>

TEST_CASE("add", "[simd]")
{
    arc::simd<float, 4> a = arc::fill<float, 4>(1.0f);
    arc::simd<float, 4> b = arc::fill<float, 4>(2.0f);
    auto c = arc::add(a, b);
    
    float result[4];
    arc::store(result, c);
    
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
    arc::store(result, c);
    
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
    arc::store(result, c);
    
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
    arc::store(result, c);
    
    REQUIRE(result[0] == 5.0f);
    REQUIRE(result[1] == 5.0f);
    REQUIRE(result[2] == 5.0f);
    REQUIRE(result[3] == 5.0f);
}

TEST_CASE("sum", "[simd]")
{
    float data[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    auto a = arc::load<float, 4>(data);
    float result = arc::sum(a);
    
    REQUIRE(result == 10.0f);
}

TEST_CASE("dot", "[simd]")
{
    float a_data[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    float b_data[4] = {2.0f, 3.0f, 4.0f, 5.0f};
    
    auto a = arc::load<float, 4>(a_data);
    auto b = arc::load<float, 4>(b_data);
    
    float result = arc::dot(a, b);
    
    // 1*2 + 2*3 + 3*4 + 4*5 = 2 + 6 + 12 + 20 = 40
    REQUIRE(result == 40.0f);
}

TEST_CASE("sum_int", "[simd]")
{
    int32_t data[4] = {1, 2, 3, 4};
    auto a = arc::load<int32_t, 4>(data);
    int32_t result = arc::sum(a);
    
    REQUIRE(result == 10);
}

TEST_CASE("dot_int", "[simd]")
{
    int32_t a_data[4] = {1, 2, 3, 4};
    int32_t b_data[4] = {2, 3, 4, 5};
    
    auto a = arc::load<int32_t, 4>(a_data);
    auto b = arc::load<int32_t, 4>(b_data);
    
    int32_t result = arc::dot(a, b);
    
    // 1*2 + 2*3 + 3*4 + 4*5 = 2 + 6 + 12 + 20 = 40
    REQUIRE(result == 40);
}

TEST_CASE("sum_double", "[simd]")
{
    double data[2] = {1.5, 2.5};
    auto a = arc::load<double, 2>(data);
    double result = arc::sum(a);
    
    REQUIRE(result == 4.0);
}

TEST_CASE("dot_double", "[simd]")
{
    double a_data[2] = {1.0, 2.0};
    double b_data[2] = {3.0, 4.0};
    
    auto a = arc::load<double, 2>(a_data);
    auto b = arc::load<double, 2>(b_data);
    
    double result = arc::dot(a, b);
    
    // 1*3 + 2*4 = 3 + 8 = 11
    REQUIRE(result == 11.0);
}