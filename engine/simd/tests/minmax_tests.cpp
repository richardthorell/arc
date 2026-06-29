#include <catch2/catch_all.hpp>

#include <arc/simd/simd.h>

TEST_CASE("min_max", "[simd]")
{
    float a_data[4] = {1.0f, 5.0f, 3.0f, 7.0f};
    float b_data[4] = {2.0f, 4.0f, 6.0f, 1.0f};
    
    auto a = arc::load_aligned<float, 4>(a_data);
    auto b = arc::load_aligned<float, 4>(b_data);
    
    auto min_result = arc::min(a, b);
    auto max_result = arc::max(a, b);
    
    float min_vals[4];
    float max_vals[4];
    arc::store_aligned(min_vals, min_result);
    arc::store_aligned(max_vals, max_result);
    
    REQUIRE(min_vals[0] == 1.0f);
    REQUIRE(min_vals[1] == 4.0f);
    REQUIRE(min_vals[2] == 3.0f);
    REQUIRE(min_vals[3] == 1.0f);
    
    REQUIRE(max_vals[0] == 2.0f);
    REQUIRE(max_vals[1] == 5.0f);
    REQUIRE(max_vals[2] == 6.0f);
    REQUIRE(max_vals[3] == 7.0f);
}

TEST_CASE("min_element", "[simd]")
{
    float data[4] = {3.0f, 1.0f, 4.0f, 2.0f};
    auto a = arc::load_aligned<float, 4>(data);
    float result = arc::min_element(a);
    
    REQUIRE(result == 1.0f);
}

TEST_CASE("min_element_negative", "[simd]")
{
    float data[4] = {-1.0f, -5.0f, -3.0f, -7.0f};
    auto a = arc::load_aligned<float, 4>(data);
    float result = arc::min_element(a);
    
    REQUIRE(result == -7.0f);
}

TEST_CASE("min_element_mixed", "[simd]")
{
    float data[4] = {1.5f, -2.5f, 3.0f, -1.0f};
    auto a = arc::load_aligned<float, 4>(data);
    float result = arc::min_element(a);
    
    REQUIRE(result == -2.5f);
}

TEST_CASE("min_element_same", "[simd]")
{
    float data[4] = {2.5f, 2.5f, 2.5f, 2.5f};
    auto a = arc::load_aligned<float, 4>(data);
    float result = arc::min_element(a);
    
    REQUIRE(result == 2.5f);
}

TEST_CASE("max_element", "[simd]")
{
    float data[4] = {3.0f, 1.0f, 4.0f, 2.0f};
    auto a = arc::load_aligned<float, 4>(data);
    float result = arc::max_element(a);
    
    REQUIRE(result == 4.0f);
}

TEST_CASE("max_element_negative", "[simd]")
{
    float data[4] = {-1.0f, -5.0f, -3.0f, -7.0f};
    auto a = arc::load_aligned<float, 4>(data);
    float result = arc::max_element(a);
    
    REQUIRE(result == -1.0f);
}

TEST_CASE("max_element_mixed", "[simd]")
{
    float data[4] = {1.5f, -2.5f, 3.0f, -1.0f};
    auto a = arc::load_aligned<float, 4>(data);
    float result = arc::max_element(a);
    
    REQUIRE(result == 3.0f);
}

TEST_CASE("max_element_same", "[simd]")
{
    float data[4] = {2.5f, 2.5f, 2.5f, 2.5f};
    auto a = arc::load_aligned<float, 4>(data);
    float result = arc::max_element(a);
    
    REQUIRE(result == 2.5f);
}

TEST_CASE("min_element_int", "[simd]")
{
    int32_t data[4] = {3, 1, 4, 2};
    auto a = arc::load_aligned<int32_t, 4>(data);
    int32_t result = arc::min_element(a);
    
    REQUIRE(result == 1);
}

TEST_CASE("max_element_int", "[simd]")
{
    int32_t data[4] = {3, 1, 4, 2};
    auto a = arc::load_aligned<int32_t, 4>(data);
    int32_t result = arc::max_element(a);
    
    REQUIRE(result == 4);
}

TEST_CASE("min_element_double", "[simd]")
{
    double data[2] = {3.5, 1.5};
    auto a = arc::load_aligned<double, 2>(data);
    double result = arc::min_element(a);
    
    REQUIRE(result == 1.5);
}

TEST_CASE("max_element_double", "[simd]")
{
    double data[2] = {3.5, 1.5};
    auto a = arc::load_aligned<double, 2>(data);
    double result = arc::max_element(a);
    
    REQUIRE(result == 3.5);
}