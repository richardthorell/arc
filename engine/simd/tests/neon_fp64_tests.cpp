#include <catch2/catch_all.hpp>

#include <arc/simd/simd.h>

#include <array>
#include <cmath>
#include <cstdint>
#include <limits>

#if defined(ARC_SIMD_NEON_FP64)
namespace
{
using fp64_ops = arc::simd::simd_op<float64x2_t>;
}

TEST_CASE("neon_fp64_horizontal_reductions", "[simd][neon]")
{
    const std::array<double, 2> lhs{3.5, -1.5};
    const std::array<double, 2> rhs{2.0, 4.0};
    const auto a = fp64_ops::load_unaligned(lhs.data());
    const auto b = fp64_ops::load_unaligned(rhs.data());

    REQUIRE(fp64_ops::sum(a) == 2.0);
    REQUIRE(fp64_ops::dot(a, b) == 1.0);
    REQUIRE(fp64_ops::min_element(a) == -1.5);
    REQUIRE(fp64_ops::max_element(a) == 3.5);

    const auto both_negative = fp64_ops::load_unaligned(std::array<double, 2>{-9.0, -2.0}.data());
    REQUIRE(fp64_ops::min_element(both_negative) == -9.0);
    REQUIRE(fp64_ops::max_element(both_negative) == -2.0);
}

TEST_CASE("neon_fp64_bitwise_not_preserves_all_64_bits", "[simd][neon]")
{
    const std::array<double, 2> values{1.0, -2.0};
    const auto a = fp64_ops::load_unaligned(values.data());
    const auto before = vreinterpretq_u64_f64(a);
    const auto after = vreinterpretq_u64_f64(fp64_ops::bitwise_not(a));

    REQUIRE(vgetq_lane_u64(after, 0) == ~vgetq_lane_u64(before, 0));
    REQUIRE(vgetq_lane_u64(after, 1) == ~vgetq_lane_u64(before, 1));
}

TEST_CASE("neon_fp64_comparisons_fill_both_mask_halves", "[simd][neon]")
{
    const std::array<double, 2> lhs{1.0, 2.0};
    const std::array<double, 2> rhs{1.0, 3.0};
    const auto a = fp64_ops::load_unaligned(lhs.data());
    const auto b = fp64_ops::load_unaligned(rhs.data());
    const auto equal = fp64_ops::cmp_eq(a, b);
    const auto not_equal = fp64_ops::cmp_ne(a, b);

    REQUIRE(vgetq_lane_u32(equal, 0) == 0xffffffffu);
    REQUIRE(vgetq_lane_u32(equal, 1) == 0xffffffffu);
    REQUIRE(vgetq_lane_u32(equal, 2) == 0u);
    REQUIRE(vgetq_lane_u32(equal, 3) == 0u);
    REQUIRE(vgetq_lane_u32(not_equal, 0) == 0u);
    REQUIRE(vgetq_lane_u32(not_equal, 1) == 0u);
    REQUIRE(vgetq_lane_u32(not_equal, 2) == 0xffffffffu);
    REQUIRE(vgetq_lane_u32(not_equal, 3) == 0xffffffffu);

    const double nan = std::numeric_limits<double>::quiet_NaN();
    const auto invalid = fp64_ops::load_unaligned(std::array<double, 2>{nan, 2.0}.data());
    const auto self_equal = fp64_ops::cmp_eq(invalid, invalid);
    const auto self_not_equal = fp64_ops::cmp_ne(invalid, invalid);
    REQUIRE(vgetq_lane_u32(self_equal, 0) == 0u);
    REQUIRE(vgetq_lane_u32(self_equal, 1) == 0u);
    REQUIRE(vgetq_lane_u32(self_equal, 2) == 0xffffffffu);
    REQUIRE(vgetq_lane_u32(self_equal, 3) == 0xffffffffu);
    REQUIRE(vgetq_lane_u32(self_not_equal, 0) == 0xffffffffu);
    REQUIRE(vgetq_lane_u32(self_not_equal, 1) == 0xffffffffu);
    REQUIRE(vgetq_lane_u32(self_not_equal, 2) == 0u);
    REQUIRE(vgetq_lane_u32(self_not_equal, 3) == 0u);
    REQUIRE(std::isnan(fp64_ops::sum(invalid)));
}
#endif
