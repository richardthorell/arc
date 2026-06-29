#include <arc/simd/simd.h>

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include <array>
#include <cstdint>

TEST_CASE("integer shifts", "[simd][game]")
{
    std::array<int32_t, 4> values{ 1, -8, 16, -32 };
    auto v = arc::load_unaligned<int32_t, 4>(values.data());

    int32_t out[4]{};
    arc::store_unaligned(out, arc::shift_left<1>(v));
    REQUIRE(out[0] == 2);
    REQUIRE(out[1] == -16);
    REQUIRE(out[2] == 32);
    REQUIRE(out[3] == -64);

    arc::store_unaligned(out, arc::shift_right_arithmetic<1>(v));
    REQUIRE(out[0] == 0);
    REQUIRE(out[1] == -4);
    REQUIRE(out[2] == 8);
    REQUIRE(out[3] == -16);
}

TEST_CASE("conversions and bit cast", "[simd][game]")
{
    std::array<float, 4> values{ 1.0f, 2.0f, 3.0f, 4.0f };
    auto f = arc::load_unaligned<float, 4>(values.data());
    auto i = arc::to_int32(f);

    int32_t int_out[4]{};
    arc::store_unaligned(int_out, i);
    REQUIRE(int_out[0] == 1);
    REQUIRE(int_out[3] == 4);

    auto bits = arc::bit_cast<uint32_t>(f);
    uint32_t bit_out[4]{};
    arc::store_unaligned(bit_out, bits);
    REQUIRE(bit_out[0] == 0x3F800000u);
}

TEST_CASE("mask bits and compression", "[simd][game]")
{
    std::array<float, 4> values{ 10.0f, 20.0f, 30.0f, 40.0f };
    auto v = arc::load_unaligned<float, 4>(values.data());
    auto mask = arc::bits_to_mask<4>(0b0101);

    REQUIRE(arc::mask_to_bits(mask) == 0b0101);
    REQUIRE(arc::popcount(mask) == 2);
    REQUIRE(arc::first_active_lane(mask) == 0);

    float out[4]{};
    arc::store_unaligned(out, arc::compress(v, mask, -1.0f));
    REQUIRE(out[0] == 10.0f);
    REQUIRE(out[1] == 30.0f);
    REQUIRE(out[2] == -1.0f);
    REQUIRE(out[3] == -1.0f);
}

TEST_CASE("gather and scatter", "[simd][game]")
{
    std::array<float, 8> source{ 0.0f, 10.0f, 20.0f, 30.0f, 40.0f, 50.0f, 60.0f, 70.0f };
    std::array<int32_t, 4> index_values{ 3, 1, 7, 0 };
    auto indices = arc::load_unaligned<int32_t, 4>(index_values.data());
    auto gathered = arc::gather<float, 4>(source.data(), indices);

    float out[4]{};
    arc::store_unaligned(out, gathered);
    REQUIRE(out[0] == 30.0f);
    REQUIRE(out[1] == 10.0f);
    REQUIRE(out[2] == 70.0f);
    REQUIRE(out[3] == 0.0f);

    std::array<float, 8> destination{};
    arc::scatter(destination.data(), indices, gathered);
    REQUIRE(destination[3] == 30.0f);
    REQUIRE(destination[1] == 10.0f);
    REQUIRE(destination[7] == 70.0f);
    REQUIRE(destination[0] == 0.0f);
}

TEST_CASE("geometry helpers", "[simd][game]")
{
    std::array<float, 4> x_values{ 1.0f, 0.0f, 0.0f, 9.0f };
    std::array<float, 4> y_values{ 0.0f, 1.0f, 0.0f, 8.0f };
    auto x = arc::load_unaligned<float, 4>(x_values.data());
    auto y = arc::load_unaligned<float, 4>(y_values.data());

    REQUIRE(arc::dot3(x, y) == 0.0f);

    float out[4]{};
    arc::store_unaligned(out, arc::cross3(x, y));
    REQUIRE(out[0] == 0.0f);
    REQUIRE(out[1] == 0.0f);
    REQUIRE(out[2] == 1.0f);
    REQUIRE(out[3] == 0.0f);

    std::array<float, 4> len_values{ 3.0f, 4.0f, 0.0f, 2.0f };
    auto len_vec = arc::load_unaligned<float, 4>(len_values.data());
    REQUIRE(arc::length3(len_vec) == Catch::Approx(5.0f));
}

TEST_CASE("transpose and transform", "[simd][game]")
{
    const float r0[4]{ 1.0f, 2.0f, 3.0f, 4.0f };
    const float r1[4]{ 5.0f, 6.0f, 7.0f, 8.0f };
    const float r2[4]{ 9.0f, 10.0f, 11.0f, 12.0f };
    const float r3[4]{ 13.0f, 14.0f, 15.0f, 16.0f };

    std::array<arc::simd<float, 4>, 4> rows{
        arc::load4<float>(r0),
        arc::load4<float>(r1),
        arc::load4<float>(r2),
        arc::load4<float>(r3)
    };

    auto columns = arc::transpose4x4(rows);
    float out[4]{};
    arc::store_unaligned(out, columns[0]);
    REQUIRE(out[0] == 1.0f);
    REQUIRE(out[1] == 5.0f);
    REQUIRE(out[2] == 9.0f);
    REQUIRE(out[3] == 13.0f);
}

TEST_CASE("transcendentals and packing", "[simd][game]")
{
    std::array<float, 4> values{ 0.0f, 0.5f, 1.0f, 2.0f };
    auto v = arc::load_unaligned<float, 4>(values.data());

    float out[4]{};
    arc::store_unaligned(out, arc::sin(v));
    REQUIRE(out[0] == Catch::Approx(0.0f));

    auto packed = arc::pack_unorm8(v);
    uint32_t packed_out[4]{};
    arc::store_unaligned(packed_out, packed);
    REQUIRE(packed_out[0] == 0u);
    REQUIRE(packed_out[1] == 128u);
    REQUIRE(packed_out[2] == 255u);
    REQUIRE(packed_out[3] == 255u);

    auto half = arc::float_to_half(v);
    auto unpacked_half = arc::half_to_float(half);
    arc::store_unaligned(out, unpacked_half);
    REQUIRE(out[0] == Catch::Approx(0.0f));
    REQUIRE(out[1] == Catch::Approx(0.5f));
    REQUIRE(out[2] == Catch::Approx(1.0f));
    REQUIRE(out[3] == Catch::Approx(2.0f));
}

TEST_CASE("integer saturation and quaternion rotation", "[simd][game]")
{
    std::array<uint32_t, 4> ua{ 0xFFFFFFFFu, 10u, 3u, 100u };
    std::array<uint32_t, 4> ub{ 1u, 20u, 5u, 50u };
    auto a = arc::load_unaligned<uint32_t, 4>(ua.data());
    auto b = arc::load_unaligned<uint32_t, 4>(ub.data());

    uint32_t uint_out[4]{};
    arc::store_unaligned(uint_out, arc::saturating_add(a, b));
    REQUIRE(uint_out[0] == 0xFFFFFFFFu);
    REQUIRE(uint_out[1] == 30u);

    arc::store_unaligned(uint_out, arc::saturating_sub(a, b));
    REQUIRE(uint_out[2] == 0u);
    REQUIRE(uint_out[3] == 50u);

    const float root_half = 0.70710678118f;
    const float q_values[4]{ 0.0f, 0.0f, root_half, root_half };
    const float v_values[4]{ 1.0f, 0.0f, 0.0f, 0.0f };
    auto q = arc::load4<float>(q_values);
    auto v = arc::load4<float>(v_values);

    float out[4]{};
    arc::store_unaligned(out, arc::quat_rotate3(q, v));
    REQUIRE(out[0] == Catch::Approx(0.0f).margin(0.00001f));
    REQUIRE(out[1] == Catch::Approx(1.0f).margin(0.00001f));
}
