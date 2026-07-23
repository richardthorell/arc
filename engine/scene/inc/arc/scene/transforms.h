#pragma once

#include <arc/scene/components.h>

#include <cmath>

namespace arc::scene
{

/**
 * @brief Build a right-handed local transform matrix from TRS components.
 */
math::matrix4f local_matrix(const transform_component& transform) noexcept;

/** Decompose an affine TRS matrix. Returns false for singular transforms. */
bool decompose_trs(const math::matrix4f& matrix, transform_component& transform) noexcept;

/** Invert an affine matrix. Returns false for singular transforms. */
bool inverse_affine(const math::matrix4f& matrix, math::matrix4f& inverse) noexcept;

/** Build a view matrix from a cached world transform matrix. */
math::matrix4f world_view_matrix(const transform_component& transform) noexcept;

math::vector3f world_position(const transform_component& transform) noexcept;
math::vector3f world_forward_direction(const transform_component& transform) noexcept;
math::vector3f world_up_direction(const transform_component& transform) noexcept;

/**
 * @brief Build a view matrix for a right-handed camera looking along local -Z.
 */
math::matrix4f view_matrix(const transform_component& transform) noexcept;

/**
 * @brief Return the transform's normalized local -Z forward direction in world space.
 */
math::vector3f forward_direction(const transform_component& transform) noexcept;

/**
 * @brief Return the transform's normalized local +Y up direction in world space.
 */
math::vector3f up_direction(const transform_component& transform) noexcept;

/**
 * @brief Build a right-handed perspective projection with Vulkan 0..1 depth.
 */
math::matrix4f perspective_rh_zo(float fov_y_radians, float aspect, float near_plane, float far_plane) noexcept;

/**
 * @brief Build a right-handed orthographic projection with Vulkan 0..1 depth.
 */
math::matrix4f orthographic_rh_zo(float height, float aspect, float near_plane, float far_plane) noexcept;

/**
 * @brief Build the active camera view-projection matrix for a viewport aspect.
 */
math::matrix4f view_projection(
    const camera_component& camera,
    const transform_component& transform,
    float aspect) noexcept;

} // namespace arc::scene
