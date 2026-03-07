#include <catch2/catch_all.hpp>

#include <arc/simd.h>

TEST_CASE("load_store", "[simd]")
{
    float data[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    auto vec = arc::load<float, 4>(data);
    
    float result[4];
    arc::store(result, vec);
    
    REQUIRE(result[0] == 1.0f);
    REQUIRE(result[1] == 2.0f);
    REQUIRE(result[2] == 3.0f);
    REQUIRE(result[3] == 4.0f);
}