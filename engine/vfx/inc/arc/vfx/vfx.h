#pragma once

#include <arc/math/vector.h>

#include <cstdint>

namespace arc::vfx
{

using effect_id = std::uint64_t;
inline constexpr effect_id invalid_effect_id = 0;

enum class simulation_space : std::uint8_t
{
    local,
    world
};

enum class simulation_backend : std::uint8_t
{
    cpu,
    gpu
};

struct particle_capacity
{
    std::uint32_t max_particles = 1024;
    std::uint32_t spawn_batch_size = 64;
};

struct emitter_definition
{
    float spawn_rate = 10.0F;
    float lifetime_min = 1.0F;
    float lifetime_max = 1.0F;
    math::vector<float, 3> initial_velocity{};
    simulation_space space = simulation_space::local;
    particle_capacity capacity{};
};

struct effect_definition
{
    emitter_definition emitter{};
    bool looping = true;
};

struct simulation_capabilities
{
    bool gpu_simulation = false;
    std::uint32_t max_particles = 0;
};

struct simulation_request
{
    simulation_backend preferred_backend = simulation_backend::gpu;
    std::uint32_t particle_count = 0;
};

enum class validation_error : std::uint8_t
{
    none,
    invalid_spawn_rate,
    invalid_lifetime,
    invalid_velocity,
    invalid_capacity,
    unsupported_backend,
    capacity_exceeded
};

[[nodiscard]] validation_error validate(const emitter_definition& definition) noexcept;
[[nodiscard]] validation_error validate(const simulation_request& request,
                                        const simulation_capabilities& capabilities) noexcept;

[[nodiscard]] simulation_backend resolve_backend(simulation_backend preferred,
                                                 const simulation_capabilities& capabilities) noexcept;

} // namespace arc::vfx
