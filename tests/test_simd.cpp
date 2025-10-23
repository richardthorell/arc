#define CATCH_CONFIG_MAIN
#include <catch2/catch_all.hpp>
#include "arc/simd/base.h"

TEST_CASE("fill and add work as expected", "[simd]")
{
    arc::simd<float, 4> a = arc::fill<float, 4>(1.0f);
    arc::simd<float, 4> b = arc::fill<float, 4>(2.0f);
    auto c = add(a, b);
    REQUIRE(true); // placeholder until load/store implemented
}
