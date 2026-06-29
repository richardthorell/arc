#include <catch2/catch_all.hpp>

#include <arc/simd/simd.h>

TEST_CASE("load_store", "[simd]")
{
    float data[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    auto vec = arc::load_aligned<float, 4>(data);
    
    float result[4];
    arc::store_aligned(result, vec);
    
    REQUIRE(result[0] == 1.0f);
    REQUIRE(result[1] == 2.0f);
    REQUIRE(result[2] == 3.0f);
    REQUIRE(result[3] == 4.0f);
}

TEST_CASE("load_store_unaligned", "[simd]")
{
    // Test unaligned load/store by offsetting the pointer
    alignas(16) float buffer[8] = {0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f};
    float* unaligned_ptr = buffer + 1; // Offset by 1 to make it unaligned
    
    auto vec = arc::load_unaligned<float, 4>(unaligned_ptr);
    
    float result[4];
    arc::store_unaligned(result, vec);
    
    REQUIRE(result[0] == 1.0f);
    REQUIRE(result[1] == 2.0f);
    REQUIRE(result[2] == 3.0f);
    REQUIRE(result[3] == 4.0f);
}