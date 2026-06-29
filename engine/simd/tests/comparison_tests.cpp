#include <catch2/catch_all.hpp>

#include <arc/simd/simd.h>

TEST_CASE("cmp_eq", "[simd]")
{
    arc::simd<float, 4> a = arc::fill<float, 4>(2.0f);
    arc::simd<float, 4> b = arc::fill<float, 4>(2.0f);
    auto mask = arc::cmp_eq(a, b);
    
    REQUIRE(arc::all(mask));
}

TEST_CASE("cmp_ne", "[simd]")
{
    arc::simd<float, 4> a = arc::fill<float, 4>(2.0f);
    arc::simd<float, 4> b = arc::fill<float, 4>(3.0f);
    auto mask = arc::cmp_ne(a, b);
    
    REQUIRE(arc::all(mask));
}