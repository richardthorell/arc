#include <arc/scene/transforms.h>

#include <algorithm>

namespace arc::scene
{
namespace
{

math::matrix4f rotation_matrix(const math::quatf& value) noexcept
{
    const auto q = math::normalize(value);
    const float x = q[0];
    const float y = q[1];
    const float z = q[2];
    const float w = q[3];

    math::matrix4f result = math::identity<float, 4>();
    result(0, 0) = 1.0f - 2.0f * y * y - 2.0f * z * z;
    result(0, 1) = 2.0f * x * y - 2.0f * z * w;
    result(0, 2) = 2.0f * x * z + 2.0f * y * w;
    result(1, 0) = 2.0f * x * y + 2.0f * z * w;
    result(1, 1) = 1.0f - 2.0f * x * x - 2.0f * z * z;
    result(1, 2) = 2.0f * y * z - 2.0f * x * w;
    result(2, 0) = 2.0f * x * z - 2.0f * y * w;
    result(2, 1) = 2.0f * y * z + 2.0f * x * w;
    result(2, 2) = 1.0f - 2.0f * x * x - 2.0f * y * y;
    return result;
}

} // namespace

math::matrix4f local_matrix(const transform_component& transform) noexcept
{
    const auto translation = math::translation(transform.position);
    const auto rotation = rotation_matrix(transform.rotation);
    const auto scale = math::scaling(transform.scale);
    return math::matmul(math::matmul(translation, rotation), scale);
}

bool inverse_affine(const math::matrix4f& value, math::matrix4f& result) noexcept
{
    const float a00 = value(0, 0), a01 = value(0, 1), a02 = value(0, 2);
    const float a10 = value(1, 0), a11 = value(1, 1), a12 = value(1, 2);
    const float a20 = value(2, 0), a21 = value(2, 1), a22 = value(2, 2);
    const float c00 = a11 * a22 - a12 * a21;
    const float c01 = a02 * a21 - a01 * a22;
    const float c02 = a01 * a12 - a02 * a11;
    const float determinant = a00 * c00 + a10 * c01 + a20 * c02;
    if (std::abs(determinant) <= 1.0e-8f)
        return false;
    const float d = 1.0f / determinant;
    result = math::identity<float, 4>();
    result(0, 0) = c00 * d;
    result(0, 1) = (a12 * a20 - a10 * a22) * d;
    result(0, 2) = (a10 * a21 - a11 * a20) * d;
    result(1, 0) = c01 * d;
    result(1, 1) = (a00 * a22 - a02 * a20) * d;
    result(1, 2) = (a02 * a10 - a00 * a12) * d;
    result(2, 0) = c02 * d;
    result(2, 1) = (a01 * a20 - a00 * a21) * d;
    result(2, 2) = (a00 * a11 - a01 * a10) * d;
    const auto inverse_translation = math::transform_vector(
        result, math::vector3f{ value(0, 3), value(1, 3), value(2, 3) });
    result(0, 3) = -inverse_translation[0];
    result(1, 3) = -inverse_translation[1];
    result(2, 3) = -inverse_translation[2];
    return true;
}

bool decompose_trs(const math::matrix4f& matrix, transform_component& transform) noexcept
{
    transform.position = { matrix(0, 3), matrix(1, 3), matrix(2, 3) };
    transform.scale = {
        std::sqrt(matrix(0, 0) * matrix(0, 0) + matrix(1, 0) * matrix(1, 0) + matrix(2, 0) * matrix(2, 0)),
        std::sqrt(matrix(0, 1) * matrix(0, 1) + matrix(1, 1) * matrix(1, 1) + matrix(2, 1) * matrix(2, 1)),
        std::sqrt(matrix(0, 2) * matrix(0, 2) + matrix(1, 2) * matrix(1, 2) + matrix(2, 2) * matrix(2, 2)) };
    if (transform.scale[0] <= 1.0e-8f || transform.scale[1] <= 1.0e-8f || transform.scale[2] <= 1.0e-8f)
        return false;
    const float determinant =
        matrix(0, 0) * (matrix(1, 1) * matrix(2, 2) - matrix(1, 2) * matrix(2, 1)) -
        matrix(0, 1) * (matrix(1, 0) * matrix(2, 2) - matrix(1, 2) * matrix(2, 0)) +
        matrix(0, 2) * (matrix(1, 0) * matrix(2, 1) - matrix(1, 1) * matrix(2, 0));
    if (determinant < 0.0f)
        transform.scale[0] = -transform.scale[0];
    math::matrix4f rotation = matrix;
    for (std::size_t column = 0; column < 3; ++column)
        for (std::size_t row = 0; row < 3; ++row)
            rotation(row, column) /= transform.scale[column];

    const float trace = rotation(0, 0) + rotation(1, 1) + rotation(2, 2);
    math::quatf quaternion{};
    if (trace > 0.0f)
    {
        const float s = std::sqrt(trace + 1.0f) * 2.0f;
        quaternion = { (rotation(2, 1) - rotation(1, 2)) / s, (rotation(0, 2) - rotation(2, 0)) / s,
            (rotation(1, 0) - rotation(0, 1)) / s, 0.25f * s };
    }
    else if (rotation(0, 0) > rotation(1, 1) && rotation(0, 0) > rotation(2, 2))
    {
        const float s = std::sqrt(1.0f + rotation(0, 0) - rotation(1, 1) - rotation(2, 2)) * 2.0f;
        quaternion = { 0.25f * s, (rotation(0, 1) + rotation(1, 0)) / s,
            (rotation(0, 2) + rotation(2, 0)) / s, (rotation(2, 1) - rotation(1, 2)) / s };
    }
    else if (rotation(1, 1) > rotation(2, 2))
    {
        const float s = std::sqrt(1.0f + rotation(1, 1) - rotation(0, 0) - rotation(2, 2)) * 2.0f;
        quaternion = { (rotation(0, 1) + rotation(1, 0)) / s, 0.25f * s,
            (rotation(1, 2) + rotation(2, 1)) / s, (rotation(0, 2) - rotation(2, 0)) / s };
    }
    else
    {
        const float s = std::sqrt(1.0f + rotation(2, 2) - rotation(0, 0) - rotation(1, 1)) * 2.0f;
        quaternion = { (rotation(0, 2) + rotation(2, 0)) / s, (rotation(1, 2) + rotation(2, 1)) / s,
            0.25f * s, (rotation(1, 0) - rotation(0, 1)) / s };
    }
    transform.rotation = math::normalize(quaternion);
    transform.mark_dirty();
    return true;
}

math::matrix4f world_view_matrix(const transform_component& transform) noexcept
{
    math::matrix4f result{};
    const auto world = transform.dirty ? local_matrix(transform) : transform.world;
    return inverse_affine(world, result) ? result : math::identity<float, 4>();
}

math::vector3f world_position(const transform_component& transform) noexcept
{
    const auto world = transform.dirty ? local_matrix(transform) : transform.world;
    return { world(0, 3), world(1, 3), world(2, 3) };
}

math::vector3f world_forward_direction(const transform_component& transform) noexcept
{
    const auto world = transform.dirty ? local_matrix(transform) : transform.world;
    return math::normalize(math::transform_vector(world, math::vector3f{ 0.0f, 0.0f, -1.0f }));
}

math::vector3f world_up_direction(const transform_component& transform) noexcept
{
    const auto world = transform.dirty ? local_matrix(transform) : transform.world;
    return math::normalize(math::transform_vector(world, math::vector3f{ 0.0f, 1.0f, 0.0f }));
}

math::matrix4f view_matrix(const transform_component& transform) noexcept
{
    const auto inverse_rotation = math::conjugate(math::normalize(transform.rotation));
    const auto rotation = rotation_matrix(inverse_rotation);
    const math::vector3f inverse_position{
        -transform.position[0],
        -transform.position[1],
        -transform.position[2]
    };
    return math::matmul(rotation, math::translation(inverse_position));
}

math::vector3f forward_direction(const transform_component& transform) noexcept
{
    const auto rotation = rotation_matrix(transform.rotation);
    const math::vector3f local_forward{ 0.0f, 0.0f, -1.0f };
    return math::normalize(math::transform_vector(rotation, local_forward));
}

math::vector3f up_direction(const transform_component& transform) noexcept
{
    const auto rotation = rotation_matrix(transform.rotation);
    const math::vector3f local_up{ 0.0f, 1.0f, 0.0f };
    return math::normalize(math::transform_vector(rotation, local_up));
}

math::matrix4f perspective_rh_zo(float fov_y_radians, float aspect, float near_plane, float far_plane) noexcept
{
    aspect = aspect <= 0.0f ? 1.0f : aspect;
    near_plane = near_plane <= 0.0f ? 0.01f : near_plane;
    far_plane = far_plane <= near_plane ? near_plane + 1.0f : far_plane;

    const float f = 1.0f / std::tan(fov_y_radians * 0.5f);
    math::matrix4f result{};
    result(0, 0) = f / aspect;
    result(1, 1) = f;
    result(2, 2) = far_plane / (near_plane - far_plane);
    result(2, 3) = (far_plane * near_plane) / (near_plane - far_plane);
    result(3, 2) = -1.0f;
    return result;
}

math::matrix4f orthographic_rh_zo(float height, float aspect, float near_plane, float far_plane) noexcept
{
    aspect = aspect <= 0.0f ? 1.0f : aspect;
    height = height <= 0.0f ? 1.0f : height;
    far_plane = far_plane <= near_plane ? near_plane + 1.0f : far_plane;

    const float width = height * aspect;
    math::matrix4f result = math::identity<float, 4>();
    result(0, 0) = 2.0f / width;
    result(1, 1) = 2.0f / height;
    result(2, 2) = 1.0f / (near_plane - far_plane);
    result(2, 3) = near_plane / (near_plane - far_plane);
    return result;
}

math::matrix4f view_projection(
    const camera_component& camera,
    const transform_component& transform,
    float aspect) noexcept
{
    const math::matrix4f projection =
        camera.projection == camera_projection::orthographic
        ? orthographic_rh_zo(camera.orthographic_height, aspect, camera.near_plane, camera.far_plane)
        : perspective_rh_zo(camera.fov_y_radians, aspect, camera.near_plane, camera.far_plane);
    return math::matmul(projection, view_matrix(transform));
}

} // namespace arc::scene
