#include <catch2/catch_all.hpp>

#include <arc/simd.h>

TEST_CASE("extract lane 0 float", "[simd][lanes]")
{
    float data[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    auto vec = arc::load_aligned<float, 4>(data);
    
    auto extracted = arc::extract<0>(vec);
    REQUIRE(extracted == 1.0f);
}

TEST_CASE("extract lane 1 float", "[simd][lanes]")
{
    float data[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    auto vec = arc::load_aligned<float, 4>(data);
    
    auto extracted = arc::extract<1>(vec);
    REQUIRE(extracted == 2.0f);
}

TEST_CASE("extract lane 2 float", "[simd][lanes]")
{
    float data[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    auto vec = arc::load_aligned<float, 4>(data);
    
    auto extracted = arc::extract<2>(vec);
    REQUIRE(extracted == 3.0f);
}

TEST_CASE("extract lane 3 float", "[simd][lanes]")
{
    float data[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    auto vec = arc::load_aligned<float, 4>(data);
    
    auto extracted = arc::extract<3>(vec);
    REQUIRE(extracted == 4.0f);
}

TEST_CASE("extract lane 0 int32_t", "[simd][lanes]")
{
    int32_t data[4] = {10, 20, 30, 40};
    auto vec = arc::load_aligned<int32_t, 4>(data);
    
    auto extracted = arc::extract<0>(vec);
    REQUIRE(extracted == 10);
}

TEST_CASE("extract lane 1 int32_t", "[simd][lanes]")
{
    int32_t data[4] = {10, 20, 30, 40};
    auto vec = arc::load_aligned<int32_t, 4>(data);
    
    auto extracted = arc::extract<1>(vec);
    REQUIRE(extracted == 20);
}

TEST_CASE("extract lane 2 int32_t", "[simd][lanes]")
{
    int32_t data[4] = {10, 20, 30, 40};
    auto vec = arc::load_aligned<int32_t, 4>(data);
    
    auto extracted = arc::extract<2>(vec);
    REQUIRE(extracted == 30);
}

TEST_CASE("extract lane 3 int32_t", "[simd][lanes]")
{
    int32_t data[4] = {10, 20, 30, 40};
    auto vec = arc::load_aligned<int32_t, 4>(data);
    
    auto extracted = arc::extract<3>(vec);
    REQUIRE(extracted == 40);
}

TEST_CASE("insert lane 0 float", "[simd][lanes]")
{
    float data[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    auto vec = arc::load_aligned<float, 4>(data);
    
    auto modified = arc::insert<0>(vec, 99.0f);
    
    float result[4];
    arc::store_aligned(result, modified);
    
    REQUIRE(result[0] == 99.0f);
    REQUIRE(result[1] == 2.0f);
    REQUIRE(result[2] == 3.0f);
    REQUIRE(result[3] == 4.0f);
}

TEST_CASE("insert lane 1 float", "[simd][lanes]")
{
    float data[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    auto vec = arc::load_aligned<float, 4>(data);
    
    auto modified = arc::insert<1>(vec, 99.0f);
    
    float result[4];
    arc::store_aligned(result, modified);
    
    REQUIRE(result[0] == 1.0f);
    REQUIRE(result[1] == 99.0f);
    REQUIRE(result[2] == 3.0f);
    REQUIRE(result[3] == 4.0f);
}

TEST_CASE("insert lane 2 float", "[simd][lanes]")
{
    float data[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    auto vec = arc::load_aligned<float, 4>(data);
    
    auto modified = arc::insert<2>(vec, 99.0f);
    
    float result[4];
    arc::store_aligned(result, modified);
    
    REQUIRE(result[0] == 1.0f);
    REQUIRE(result[1] == 2.0f);
    REQUIRE(result[2] == 99.0f);
    REQUIRE(result[3] == 4.0f);
}

TEST_CASE("insert lane 3 float", "[simd][lanes]")
{
    float data[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    auto vec = arc::load_aligned<float, 4>(data);
    
    auto modified = arc::insert<3>(vec, 99.0f);
    
    float result[4];
    arc::store_aligned(result, modified);
    
    REQUIRE(result[0] == 1.0f);
    REQUIRE(result[1] == 2.0f);
    REQUIRE(result[2] == 3.0f);
    REQUIRE(result[3] == 99.0f);
}

TEST_CASE("insert lane 0 int32_t", "[simd][lanes]")
{
    int32_t data[4] = {10, 20, 30, 40};
    auto vec = arc::load_aligned<int32_t, 4>(data);
    
    auto modified = arc::insert<0>(vec, 99);
    
    int32_t result[4];
    arc::store_aligned(result, modified);
    
    REQUIRE(result[0] == 99);
    REQUIRE(result[1] == 20);
    REQUIRE(result[2] == 30);
    REQUIRE(result[3] == 40);
}

TEST_CASE("insert lane 1 int32_t", "[simd][lanes]")
{
    int32_t data[4] = {10, 20, 30, 40};
    auto vec = arc::load_aligned<int32_t, 4>(data);
    
    auto modified = arc::insert<1>(vec, 99);
    
    int32_t result[4];
    arc::store_aligned(result, modified);
    
    REQUIRE(result[0] == 10);
    REQUIRE(result[1] == 99);
    REQUIRE(result[2] == 30);
    REQUIRE(result[3] == 40);
}

TEST_CASE("insert lane 2 int32_t", "[simd][lanes]")
{
    int32_t data[4] = {10, 20, 30, 40};
    auto vec = arc::load_aligned<int32_t, 4>(data);
    
    auto modified = arc::insert<2>(vec, 99);
    
    int32_t result[4];
    arc::store_aligned(result, modified);
    
    REQUIRE(result[0] == 10);
    REQUIRE(result[1] == 20);
    REQUIRE(result[2] == 99);
    REQUIRE(result[3] == 40);
}

TEST_CASE("insert lane 3 int32_t", "[simd][lanes]")
{
    int32_t data[4] = {10, 20, 30, 40};
    auto vec = arc::load_aligned<int32_t, 4>(data);
    
    auto modified = arc::insert<3>(vec, 99);
    
    int32_t result[4];
    arc::store_aligned(result, modified);
    
    REQUIRE(result[0] == 10);
    REQUIRE(result[1] == 20);
    REQUIRE(result[2] == 30);
    REQUIRE(result[3] == 99);
}

TEST_CASE("broadcast lane 0 float", "[simd][lanes]")
{
    float data[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    auto vec = arc::load_aligned<float, 4>(data);
    
    auto broadcasted = arc::broadcast<0>(vec);
    
    float result[4];
    arc::store_aligned(result, broadcasted);
    
    REQUIRE(result[0] == 1.0f);
    REQUIRE(result[1] == 1.0f);
    REQUIRE(result[2] == 1.0f);
    REQUIRE(result[3] == 1.0f);
}

TEST_CASE("broadcast lane 1 float", "[simd][lanes]")
{
    float data[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    auto vec = arc::load_aligned<float, 4>(data);
    
    auto broadcasted = arc::broadcast<1>(vec);
    
    float result[4];
    arc::store_aligned(result, broadcasted);
    
    REQUIRE(result[0] == 2.0f);
    REQUIRE(result[1] == 2.0f);
    REQUIRE(result[2] == 2.0f);
    REQUIRE(result[3] == 2.0f);
}

TEST_CASE("broadcast lane 2 float", "[simd][lanes]")
{
    float data[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    auto vec = arc::load_aligned<float, 4>(data);
    
    auto broadcasted = arc::broadcast<2>(vec);
    
    float result[4];
    arc::store_aligned(result, broadcasted);
    
    REQUIRE(result[0] == 3.0f);
    REQUIRE(result[1] == 3.0f);
    REQUIRE(result[2] == 3.0f);
    REQUIRE(result[3] == 3.0f);
}

TEST_CASE("broadcast lane 3 float", "[simd][lanes]")
{
    float data[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    auto vec = arc::load_aligned<float, 4>(data);
    
    auto broadcasted = arc::broadcast<3>(vec);
    
    float result[4];
    arc::store_aligned(result, broadcasted);
    
    REQUIRE(result[0] == 4.0f);
    REQUIRE(result[1] == 4.0f);
    REQUIRE(result[2] == 4.0f);
    REQUIRE(result[3] == 4.0f);
}

TEST_CASE("broadcast lane 0 int32_t", "[simd][lanes]")
{
    int32_t data[4] = {10, 20, 30, 40};
    auto vec = arc::load_aligned<int32_t, 4>(data);
    
    auto broadcasted = arc::broadcast<0>(vec);
    
    int32_t result[4];
    arc::store_aligned(result, broadcasted);
    
    REQUIRE(result[0] == 10);
    REQUIRE(result[1] == 10);
    REQUIRE(result[2] == 10);
    REQUIRE(result[3] == 10);
}

TEST_CASE("broadcast lane 1 int32_t", "[simd][lanes]")
{
    int32_t data[4] = {10, 20, 30, 40};
    auto vec = arc::load_aligned<int32_t, 4>(data);
    
    auto broadcasted = arc::broadcast<1>(vec);
    
    int32_t result[4];
    arc::store_aligned(result, broadcasted);
    
    REQUIRE(result[0] == 20);
    REQUIRE(result[1] == 20);
    REQUIRE(result[2] == 20);
    REQUIRE(result[3] == 20);
}

TEST_CASE("broadcast lane 2 int32_t", "[simd][lanes]")
{
    int32_t data[4] = {10, 20, 30, 40};
    auto vec = arc::load_aligned<int32_t, 4>(data);
    
    auto broadcasted = arc::broadcast<2>(vec);
    
    int32_t result[4];
    arc::store_aligned(result, broadcasted);
    
    REQUIRE(result[0] == 30);
    REQUIRE(result[1] == 30);
    REQUIRE(result[2] == 30);
    REQUIRE(result[3] == 30);
}

TEST_CASE("broadcast lane 3 int32_t", "[simd][lanes]")
{
    int32_t data[4] = {10, 20, 30, 40};
    auto vec = arc::load_aligned<int32_t, 4>(data);
    
    auto broadcasted = arc::broadcast<3>(vec);
    
    int32_t result[4];
    arc::store_aligned(result, broadcasted);
    
    REQUIRE(result[0] == 40);
    REQUIRE(result[1] == 40);
    REQUIRE(result[2] == 40);
    REQUIRE(result[3] == 40);
}