#include <arc/vfx/vfx.h>

#include <limits>

int main()
{
    using namespace arc::vfx;

    emitter_definition emitter{};
    if (validate(emitter) != validation_error::none)
    {
        return 1;
    }

    emitter.spawn_rate = -1.0F;
    if (validate(emitter) != validation_error::invalid_spawn_rate)
    {
        return 2;
    }
    emitter.spawn_rate = 10.0F;

    emitter.lifetime_max = 0.5F;
    if (validate(emitter) != validation_error::invalid_lifetime)
    {
        return 3;
    }
    emitter.lifetime_max = 1.0F;

    emitter.initial_velocity[0] = std::numeric_limits<float>::infinity();
    if (validate(emitter) != validation_error::invalid_velocity)
    {
        return 4;
    }
    emitter.initial_velocity[0] = 0.0F;

    emitter.capacity.spawn_batch_size = emitter.capacity.max_particles + 1;
    if (validate(emitter) != validation_error::invalid_capacity)
    {
        return 5;
    }

    simulation_capabilities cpu_only{false, 2048};
    simulation_request request{simulation_backend::gpu, 128};
    if (validate(request, cpu_only) != validation_error::unsupported_backend)
    {
        return 6;
    }
    if (resolve_backend(request.preferred_backend, cpu_only) != simulation_backend::cpu)
    {
        return 7;
    }

    simulation_capabilities gpu{true, 1024};
    if (validate(request, gpu) != validation_error::none)
    {
        return 8;
    }
    if (resolve_backend(request.preferred_backend, gpu) != simulation_backend::gpu)
    {
        return 9;
    }

    request.particle_count = 2048;
    if (validate(request, gpu) != validation_error::capacity_exceeded)
    {
        return 10;
    }

    return 0;
}
