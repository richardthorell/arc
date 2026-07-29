#include <catch2/catch_all.hpp>

#include <arc/simd/simd.h>

TEST_CASE("cmp_eq", "[simd]")
{
    arc::simd::simd<float, 4> a = arc::simd::fill<float, 4>(2.0f);
    arc::simd::simd<float, 4> b = arc::simd::fill<float, 4>(2.0f);
    auto mask = arc::simd::cmp_eq(a, b);
    
    REQUIRE(arc::simd::all(mask));
}

TEST_CASE("cmp_ne", "[simd]")
{
    arc::simd::simd<float, 4> a = arc::simd::fill<float, 4>(2.0f);
    arc::simd::simd<float, 4> b = arc::simd::fill<float, 4>(3.0f);
    auto mask = arc::simd::cmp_ne(a, b);
    
    REQUIRE(arc::simd::all(mask));
}