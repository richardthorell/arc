#include <arc/vfx/vfx.h>

#include <cmath>

namespace arc::vfx
{
namespace
{

[[nodiscard]] bool finite_vector(const math::vector<float, 3>& value) noexcept
{
    return std::isfinite(value[0]) && std::isfinite(value[1]) && std::isfinite(value[2]);
}

} // namespace

validation_error validate(const emitter_definition& definition) noexcept
{
    if (!std::isfinite(definition.spawn_rate) || definition.spawn_rate < 0.0F)
    {
        return validation_error::invalid_spawn_rate;
    }

    if (!std::isfinite(definition.lifetime_min) || !std::isfinite(definition.lifetime_max) ||
        definition.lifetime_min <= 0.0F || definition.lifetime_max < definition.lifetime_min)
    {
        return validation_error::invalid_lifetime;
    }

    if (!finite_vector(definition.initial_velocity))
    {
        return validation_error::invalid_velocity;
    }

    if (definition.capacity.max_particles == 0 || definition.capacity.spawn_batch_size == 0 ||
        definition.capacity.spawn_batch_size > definition.capacity.max_particles)
    {
        return validation_error::invalid_capacity;
    }

    return validation_error::none;
}

validation_error validate(const simulation_request& request, const simulation_capabilities& capabilities) noexcept
{
    if (request.preferred_backend == simulation_backend::gpu && !capabilities.gpu_simulation)
    {
        return validation_error::unsupported_backend;
    }

    if (request.particle_count > capabilities.max_particles)
    {
        return validation_error::capacity_exceeded;
    }

    return validation_error::none;
}

simulation_backend resolve_backend(simulation_backend preferred, const simulation_capabilities& capabilities) noexcept
{
    if (preferred == simulation_backend::gpu && !capabilities.gpu_simulation)
    {
        return simulation_backend::cpu;
    }
    return preferred;
}

} // namespace arc::vfx
