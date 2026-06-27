#pragma once

#include <arc/simd/core/ops/arithmetic.h>
#include <arc/simd/core/ops/math.h>
#include <arc/simd/core/ops/scalar.h>

#include <array>
#include <cmath>

namespace arc
{

/// @brief Return the dot product of the first three lanes.
template <class T, std::size_t N>
constexpr T dot3(const simd<T, N>& a, const simd<T, N>& b) noexcept
{
    static_assert(N >= 3, "dot3 requires at least 3 lanes");
    return extract<0>(a) * extract<0>(b) +
           extract<1>(a) * extract<1>(b) +
           extract<2>(a) * extract<2>(b);
}

/// @brief Return the dot product of the first four lanes.
template <class T, std::size_t N>
constexpr T dot4(const simd<T, N>& a, const simd<T, N>& b) noexcept
{
    static_assert(N >= 4, "dot4 requires at least 4 lanes");
    return extract<0>(a) * extract<0>(b) +
           extract<1>(a) * extract<1>(b) +
           extract<2>(a) * extract<2>(b) +
           extract<3>(a) * extract<3>(b);
}

/// @brief Return squared length using the first three lanes.
template <class T, std::size_t N>
constexpr T length_squared3(const simd<T, N>& value) noexcept
{
    return dot3(value, value);
}

/// @brief Return squared length using the first four lanes.
template <class T, std::size_t N>
constexpr T length_squared4(const simd<T, N>& value) noexcept
{
    return dot4(value, value);
}

/// @brief Return length using the first three lanes.
template <class T, std::size_t N>
inline T length3(const simd<T, N>& value) noexcept
{
    return static_cast<T>(std::sqrt(length_squared3(value)));
}

/// @brief Return length using the first four lanes.
template <class T, std::size_t N>
inline T length4(const simd<T, N>& value) noexcept
{
    return static_cast<T>(std::sqrt(length_squared4(value)));
}

/// @brief Normalize the first three lanes, using `fallback` for zero length.
template <class T, std::size_t N>
inline simd<T, N> normalize3(const simd<T, N>& value, T fallback = T{}) noexcept
{
    auto lanes = detail::simd_to_array(value);
    const T len = length3(value);
    if (len == T{})
    {
        lanes[0] = fallback;
        lanes[1] = fallback;
        lanes[2] = fallback;
    }
    else
    {
        lanes[0] /= len;
        lanes[1] /= len;
        lanes[2] /= len;
    }
    return detail::simd_from_array<T, N>(lanes);
}

/// @brief Normalize the first four lanes, using `fallback` for zero length.
template <class T, std::size_t N>
inline simd<T, N> normalize4(const simd<T, N>& value, T fallback = T{}) noexcept
{
    auto lanes = detail::simd_to_array(value);
    const T len = length4(value);
    if (len == T{})
    {
        lanes[0] = fallback;
        lanes[1] = fallback;
        lanes[2] = fallback;
        lanes[3] = fallback;
    }
    else
    {
        lanes[0] /= len;
        lanes[1] /= len;
        lanes[2] /= len;
        lanes[3] /= len;
    }
    return detail::simd_from_array<T, N>(lanes);
}

/// @brief Return the 3D cross product in the first three lanes.
template <class T, std::size_t N>
constexpr simd<T, N> cross3(const simd<T, N>& a, const simd<T, N>& b) noexcept
{
    static_assert(N >= 3, "cross3 requires at least 3 lanes");
    auto result = detail::simd_to_array(a);
    result[0] = extract<1>(a) * extract<2>(b) - extract<2>(a) * extract<1>(b);
    result[1] = extract<2>(a) * extract<0>(b) - extract<0>(a) * extract<2>(b);
    result[2] = extract<0>(a) * extract<1>(b) - extract<1>(a) * extract<0>(b);
    if constexpr (N >= 4)
        result[3] = T{};
    return detail::simd_from_array<T, N>(result);
}

/// @brief Linearly interpolate between two SIMD vectors.
template <class T, std::size_t N>
constexpr simd<T, N> lerp(const simd<T, N>& a, const simd<T, N>& b, T t) noexcept
{
    return a + (b - a) * fill<T, N>(t);
}

/// @brief Reflect a 3D incident vector around a 3D normal.
template <class T, std::size_t N>
constexpr simd<T, N> reflect3(const simd<T, N>& incident, const simd<T, N>& normal) noexcept
{
    return incident - normal * fill<T, N>(static_cast<T>(2) * dot3(incident, normal));
}

/// @brief Project a 3D vector onto another 3D vector.
template <class T, std::size_t N>
constexpr simd<T, N> project3(const simd<T, N>& value, const simd<T, N>& onto) noexcept
{
    const T denom = dot3(onto, onto);
    if (denom == T{})
        return fill<T, N>(T{});
    return onto * fill<T, N>(dot3(value, onto) / denom);
}

/// @brief Transpose four SIMD rows that represent a 4x4 matrix.
template <class T>
constexpr std::array<simd<T, 4>, 4> transpose4x4(const std::array<simd<T, 4>, 4>& rows) noexcept
{
    std::array<simd<T, 4>, 4> result{};
    result[0] = detail::simd_from_array<T, 4>({ extract<0>(rows[0]), extract<0>(rows[1]), extract<0>(rows[2]), extract<0>(rows[3]) });
    result[1] = detail::simd_from_array<T, 4>({ extract<1>(rows[0]), extract<1>(rows[1]), extract<1>(rows[2]), extract<1>(rows[3]) });
    result[2] = detail::simd_from_array<T, 4>({ extract<2>(rows[0]), extract<2>(rows[1]), extract<2>(rows[2]), extract<2>(rows[3]) });
    result[3] = detail::simd_from_array<T, 4>({ extract<3>(rows[0]), extract<3>(rows[1]), extract<3>(rows[2]), extract<3>(rows[3]) });
    return result;
}

/// @brief Transform a 4D SIMD vector by four SIMD matrix rows.
template <class T>
constexpr simd<T, 4> transform4(const std::array<simd<T, 4>, 4>& rows, const simd<T, 4>& value) noexcept
{
    return detail::simd_from_array<T, 4>({
        dot4(rows[0], value),
        dot4(rows[1], value),
        dot4(rows[2], value),
        dot4(rows[3], value)
    });
}

/// @brief Transform a 3D point by a 4x4 matrix represented as rows.
template <class T>
constexpr simd<T, 4> transform_point3(const std::array<simd<T, 4>, 4>& rows, const simd<T, 4>& point) noexcept
{
    auto p = detail::simd_to_array(point);
    p[3] = static_cast<T>(1);
    return transform4(rows, detail::simd_from_array<T, 4>(p));
}

/// @brief Transform a 3D direction by a 4x4 matrix represented as rows.
template <class T>
constexpr simd<T, 4> transform_direction3(const std::array<simd<T, 4>, 4>& rows, const simd<T, 4>& direction) noexcept
{
    auto d = detail::simd_to_array(direction);
    d[3] = T{};
    return transform4(rows, detail::simd_from_array<T, 4>(d));
}

} // namespace arc
