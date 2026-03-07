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