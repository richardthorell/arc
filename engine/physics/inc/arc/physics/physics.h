#pragma once

#include <arc/math/quaternion.h>
#include <arc/math/vector.h>

#include <cstdint>
#include <limits>
#include <variant>

namespace arc::physics
{

using body_id = std::uint64_t;
inline constexpr body_id invalid_body_id = 0;

enum class body_motion : std::uint8_t
{
    static_body,
    kinematic,
    dynamic
};

struct sphere_shape
{
    float radius = 0.5F;
};

struct box_shape
{
    math::vector<float, 3> half_extents{0.5F, 0.5F, 0.5F};
};

struct capsule_shape
{
    float radius = 0.5F;
    float half_height = 0.5F;
};

using collision_shape = std::variant<sphere_shape, box_shape, capsule_shape>;

struct body_definition
{
    body_motion motion = body_motion::static_body;
    collision_shape shape = box_shape{};
    math::vector<float, 3> position{};
    math::quaternion<float> rotation{};
    float mass = 1.0F;
    float linear_damping = 0.0F;
    float angular_damping = 0.0F;
    bool enable_gravity = true;
};

struct ray_query
{
    math::vector<float, 3> origin{};
    math::vector<float, 3> direction{0.0F, 0.0F, 1.0F};
    float max_distance = std::numeric_limits<float>::infinity();
    std::uint32_t layer_mask = ~std::uint32_t{0};
};

struct ray_hit
{
    body_id body = invalid_body_id;
    math::vector<float, 3> position{};
    math::vector<float, 3> normal{};
    float distance = 0.0F;
};

enum class validation_error : std::uint8_t
{
    none,
    invalid_shape,
    invalid_mass,
    invalid_damping,
    invalid_query
};

[[nodiscard]] validation_error validate(const body_definition& definition) noexcept;
[[nodiscard]] validation_error validate(const ray_query& query) noexcept;

} // namespace arc::physics
