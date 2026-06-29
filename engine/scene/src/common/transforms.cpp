#include <arc/scene/transforms.h>

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
