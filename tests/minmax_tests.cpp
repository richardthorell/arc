#include <catch2/catch_all.hpp>

#include <arc/simd.h>

TEST_CASE("min_max", "[simd]")
{
    float a_data[4] = {1.0f, 5.0f, 3.0f, 7.0f};
    float b_data[4] = {2.0f, 4.0f, 6.0f, 1.0f};
    
    auto a = arc::load<float, 4>(a_data);
    auto b = arc::load<float, 4>(b_data);
    
    auto min_result = arc::min(a, b);
    auto max_result = arc::max(a, b);
    
    float min_vals[4];
    float max_vals[4];
    arc::store(min_vals, min_result);
    arc::store(max_vals, max_result);
    
    REQUIRE(min_vals[0] == 1.0f);
    REQUIRE(min_vals[1] == 4.0f);
    REQUIRE(min_vals[2] == 3.0f);
    REQUIRE(min_vals[3] == 1.0f);
    
    REQUIRE(max_vals[0] == 2.0f);
    REQUIRE(max_vals[1] == 5.0f);
    REQUIRE(max_vals[2] == 6.0f);
    REQUIRE(max_vals[3] == 7.0f);
}