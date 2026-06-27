#pragma once

#include <arc/simd/core/ops/geometry.h>
#include <arc/simd/core/ops/scalar.h>

namespace arc
{

/// @brief Return the conjugate of a quaternion stored as `(x, y, z, w)`.
template <class T>
constexpr simd<T, 4> quat_conjugate(const simd<T, 4>& q) noexcept
{
    return detail::simd_from_array<T, 4>({
        -extract<0>(q),
        -extract<1>(q),
        -extract<2>(q),
        extract<3>(q)
    });
}

/// @brief Normalize a quaternion stored as `(x, y, z, w)`.
template <class T>
inline simd<T, 4> quat_normalize(const simd<T, 4>& q) noexcept
{
    return normalize4(q);
}

/// @brief Return the Hamilton product of two quaternions stored as `(x, y, z, w)`.
template <class T>
constexpr simd<T, 4> quat_mul(const simd<T, 4>& a, const simd<T, 4>& b) noexcept
{
    const T ax = extract<0>(a);
    const T ay = extract<1>(a);
    const T az = extract<2>(a);
    const T aw = extract<3>(a);
    const T bx = extract<0>(b);
    const T by = extract<1>(b);
    const T bz = extract<2>(b);
    const T bw = extract<3>(b);

    return detail::simd_from_array<T, 4>({
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
        aw * bw - ax * bx - ay * by - az * bz
    });
}

/// @brief Rotate a 3D vector by a quaternion stored as `(x, y, z, w)`.
template <class T>
constexpr simd<T, 4> quat_rotate3(const simd<T, 4>& q, const simd<T, 4>& vector) noexcept
{
    const auto qv = detail::simd_from_array<T, 4>({ extract<0>(q), extract<1>(q), extract<2>(q), T{} });
    const auto t = cross3(qv, vector) * fill<T, 4>(static_cast<T>(2));
    return vector + t * fill<T, 4>(extract<3>(q)) + cross3(qv, t);
}

} // namespace arc
