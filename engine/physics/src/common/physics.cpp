#include <arc/physics/physics.h>

#include <cmath>
#include <type_traits>

namespace arc::physics
{
namespace
{

bool finite_positive(float value) noexcept
{
    return std::isfinite(value) && value > 0.0F;
}

bool valid_shape(const collision_shape& shape) noexcept
{
    return std::visit(
        [](const auto& value)
        {
            using shape_type = std::decay_t<decltype(value)>;
            if constexpr (std::is_same_v<shape_type, sphere_shape>)
                return finite_positive(value.radius);
            else if constexpr (std::is_same_v<shape_type, capsule_shape>)
                return finite_positive(value.radius) && finite_positive(value.half_height);
            else
                return finite_positive(value.half_extents[0]) && finite_positive(value.half_extents[1]) &&
                       finite_positive(value.half_extents[2]);
        },
        shape);
}

} // namespace

validation_error validate(const body_definition& definition) noexcept
{
    if (!valid_shape(definition.shape)) return validation_error::invalid_shape;
    if (definition.motion == body_motion::dynamic && !finite_positive(definition.mass))
        return validation_error::invalid_mass;
    if (!std::isfinite(definition.linear_damping) || definition.linear_damping < 0.0F ||
        !std::isfinite(definition.angular_damping) || definition.angular_damping < 0.0F)
        return validation_error::invalid_damping;
    return validation_error::none;
}

validation_error validate(const ray_query& query) noexcept
{
    const float direction_length_squared = query.direction[0] * query.direction[0] +
                                           query.direction[1] * query.direction[1] +
                                           query.direction[2] * query.direction[2];
    if (!std::isfinite(direction_length_squared) || direction_length_squared <= 0.0F ||
        std::isnan(query.max_distance) || query.max_distance <= 0.0F)
        return validation_error::invalid_query;
    return validation_error::none;
}

} // namespace arc::physics
