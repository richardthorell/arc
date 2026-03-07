#include <catch2/catch_all.hpp>

#include <arc/simd.h>

TEST_CASE("bitwise", "[simd]")
{
    arc::simd<int32_t, 4> a = arc::fill<int32_t, 4>(5); // 101
    arc::simd<int32_t, 4> b = arc::fill<int32_t, 4>(3); // 011
    
    auto and_result = arc::bitwise_and(a, b);
    auto or_result = arc::bitwise_or(a, b);
    auto xor_result = arc::bitwise_xor(a, b);
    
    int32_t and_vals[4];
    int32_t or_vals[4];
    int32_t xor_vals[4];
    
    arc::store(and_vals, and_result);
    arc::store(or_vals, or_result);
    arc::store(xor_vals, xor_result);
    
    REQUIRE(and_vals[0] == 1); // 101 & 011 = 001
    REQUIRE(or_vals[0] == 7);  // 101 | 011 = 111
    REQUIRE(xor_vals[0] == 6); // 101 ^ 011 = 110
}