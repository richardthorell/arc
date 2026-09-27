#include <arc/navigation/navigation.h>

#include <cmath>

namespace arc::navigation
{
namespace
{

template <std::size_t N> [[nodiscard]] bool finite_vector(const math::vector<float, N>& value) noexcept
{
    for (std::size_t index = 0; index < N; ++index)
    {
        if (!std::isfinite(value[index]))
        {
            return false;
        }
    }
    return true;
}

} // namespace

validation_error validate(const agent_definition& agent) noexcept
{
    if (!std::isfinite(agent.radius) || agent.radius <= 0.0F || !std::isfinite(agent.height) || agent.height <= 0.0F ||
        !std::isfinite(agent.max_step_height) || agent.max_step_height < 0.0F ||
        !std::isfinite(agent.max_slope_degrees) || agent.max_slope_degrees < 0.0F || agent.max_slope_degrees >= 90.0F ||
        agent.layer_mask == 0)
    {
        return validation_error::invalid_agent;
    }

    return validation_error::none;
}

validation_error validate(const navigation_bounds& bounds) noexcept
{
    if (!finite_vector(bounds.minimum) || !finite_vector(bounds.maximum))
    {
        return validation_error::invalid_bounds;
    }

    for (std::size_t axis = 0; axis < 3; ++axis)
    {
        if (bounds.minimum[axis] > bounds.maximum[axis])
        {
            return validation_error::invalid_bounds;
        }
    }

    return validation_error::none;
}

validation_error validate(const path_query& query) noexcept
{
    if (validate(query.agent) != validation_error::none)
    {
        return validation_error::invalid_agent;
    }

    if (!finite_vector(query.start) || !finite_vector(query.goal) || std::isnan(query.max_path_length) ||
        query.max_path_length <= 0.0F)
    {
        return validation_error::invalid_query;
    }

    return validation_error::none;
}

} // namespace arc::navigation
