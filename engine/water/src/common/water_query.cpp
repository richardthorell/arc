#include <arc/water/water_query.h>

#include <arc/water/ocean_simulation.h>

#include <algorithm>
#include <array>
#include <bit>
#include <chrono>
#include <cmath>
#include <complex>
#include <limits>
#include <unordered_map>
#include <utility>
#include <vector>

namespace arc::water
{
namespace
{

constexpr float pi = 3.14159265358979323846f;
constexpr float two_pi = 2.0f * pi;
constexpr float minimum_positive = 1.0e-6f;

struct query_spectral_mode
{
    std::complex<float> initial;
    std::complex<float> mirrored_conjugate;
    float wave_x{};
    float wave_z{};
    float angular_frequency{};
    float inverse_sample_count{};
};

struct query_cascade
{
    std::vector<query_spectral_mode> modes;
};

struct query_body_cache
{
    std::array<query_cascade, maximum_ocean_cascades> cascades;
    water_query_profile profile;
    std::uint32_t cascade_count{};
    std::uint64_t settings_signature{};
    float choppiness{1.0f};
};

struct query_accumulator
{
    math::vector3f displacement{};
    math::vector3f velocity{};
    float slope_x{};
    float slope_z{};
    ocean_displacement_derivatives derivatives;
};

std::uint64_t body_key(water_body_handle body) noexcept
{
    return (static_cast<std::uint64_t>(body.generation) << 32u) | body.index;
}

void hash_combine(std::uint64_t& seed, std::uint64_t value) noexcept
{
    value += 0x9e3779b97f4a7c15ull;
    value = (value ^ (value >> 30u)) * 0xbf58476d1ce4e5b9ull;
    value = (value ^ (value >> 27u)) * 0x94d049bb133111ebull;
    value ^= value >> 31u;
    seed ^= value + 0x9e3779b97f4a7c15ull + (seed << 6u) + (seed >> 2u);
}

std::uint64_t settings_signature(const water_body_descriptor& descriptor) noexcept
{
    const auto& simulation = descriptor.settings.simulation;
    const auto& foam = descriptor.settings.foam;
    std::uint64_t result{0x4152435751554552ull};
    hash_combine(result, static_cast<std::uint64_t>(descriptor.type));
    hash_combine(result, static_cast<std::uint64_t>(descriptor.settings.quality));
    hash_combine(result, std::bit_cast<std::uint32_t>(simulation.wind_speed));
    hash_combine(result, std::bit_cast<std::uint32_t>(simulation.wind_direction[0]));
    hash_combine(result, std::bit_cast<std::uint32_t>(simulation.wind_direction[1]));
    hash_combine(result, std::bit_cast<std::uint32_t>(simulation.fetch_length));
    hash_combine(result, std::bit_cast<std::uint32_t>(simulation.wave_amplitude));
    hash_combine(result, std::bit_cast<std::uint32_t>(simulation.choppiness));
    hash_combine(result, simulation.seed);
    hash_combine(result, foam.enabled ? 1u : 0u);
    hash_combine(result, std::bit_cast<std::uint32_t>(foam.threshold));
    hash_combine(result, std::bit_cast<std::uint32_t>(foam.decay));
    return result;
}

int signed_frequency(std::uint32_t coordinate, std::uint32_t resolution) noexcept
{
    const auto value = static_cast<int>(coordinate);
    const auto size = static_cast<int>(resolution);
    return value <= size / 2 ? value : value - size;
}

math::vector3f normalized_surface_normal(float slope_x, float slope_z) noexcept
{
    const float magnitude = std::sqrt(slope_x * slope_x + 1.0f + slope_z * slope_z);
    if (!(magnitude > minimum_positive) || !std::isfinite(magnitude)) return {0.0f, 1.0f, 0.0f};
    const float inverse = 1.0f / magnitude;
    return {-slope_x * inverse, inverse, -slope_z * inverse};
}

query_accumulator evaluate(const query_body_cache& cache, float world_x, float world_z, float time_seconds,
                           bool complete_surface, std::uint64_t& evaluated_modes) noexcept
{
    query_accumulator result;
    const std::complex<float> imaginary{0.0f, 1.0f};
    for (std::uint32_t cascade_index = 0u; cascade_index < cache.cascade_count; ++cascade_index)
        for (const auto& mode : cache.cascades[cascade_index].modes)
        {
            const float temporal_phase = mode.angular_frequency * time_seconds;
            const std::complex<float> temporal{std::cos(temporal_phase), std::sin(temporal_phase)};
            const auto a = mode.initial * temporal;
            const auto b = mode.mirrored_conjugate * std::conj(temporal);
            const auto height = a + b;
            const float spatial_phase = mode.wave_x * world_x + mode.wave_z * world_z;
            const std::complex<float> spatial{std::cos(spatial_phase), std::sin(spatial_phase)};
            const auto resolved_height = height * spatial * mode.inverse_sample_count;
            const float wave_number = std::sqrt(mode.wave_x * mode.wave_x + mode.wave_z * mode.wave_z);
            if (!(wave_number > minimum_positive)) continue;
            const float normalized_x = mode.wave_x / wave_number;
            const float normalized_z = mode.wave_z / wave_number;
            const auto horizontal_x = imaginary * (-normalized_x * cache.choppiness) * resolved_height;
            const auto horizontal_z = imaginary * (-normalized_z * cache.choppiness) * resolved_height;

            result.derivatives.displacement_x_dx += (imaginary * mode.wave_x * horizontal_x).real();
            result.derivatives.displacement_x_dz += (imaginary * mode.wave_z * horizontal_x).real();
            result.derivatives.displacement_z_dx += (imaginary * mode.wave_x * horizontal_z).real();
            result.derivatives.displacement_z_dz += (imaginary * mode.wave_z * horizontal_z).real();
            if (complete_surface)
            {
                const auto height_velocity =
                    imaginary * mode.angular_frequency * (a - b) * spatial * mode.inverse_sample_count;
                result.displacement[0] += horizontal_x.real();
                result.displacement[1] += resolved_height.real();
                result.displacement[2] += horizontal_z.real();
                result.slope_x += (imaginary * mode.wave_x * resolved_height).real();
                result.slope_z += (imaginary * mode.wave_z * resolved_height).real();
                result.velocity[0] += (imaginary * (-normalized_x * cache.choppiness) * height_velocity).real();
                result.velocity[1] += height_velocity.real();
                result.velocity[2] += (imaginary * (-normalized_z * cache.choppiness) * height_velocity).real();
            }
            ++evaluated_modes;
        }
    return result;
}

} // namespace

struct water_query_system::implementation
{
    std::unordered_map<std::uint64_t, query_body_cache> bodies;
    water_query_statistics statistics;
    std::vector<water_query_debug_sample> debug_samples;

    [[nodiscard]] std::optional<water_surface_sample> sample_prepared(const water_registry& registry,
                                                                      water_body_handle body,
                                                                      const math::vector3f& position,
                                                                      float simulation_time_seconds);
};

water_query_profile query_profile(water_quality quality) noexcept
{
    switch (quality)
    {
        case water_quality::low:
            return {.maximum_frequency_index = 32u,
                    .foam_history_samples = 1u,
                    .foam_history_interval_seconds = 0.4f,
                    .maximum_debug_samples = 32u};
        case water_quality::medium:
            return {.maximum_frequency_index = 24u,
                    .foam_history_samples = 2u,
                    .foam_history_interval_seconds = 0.35f,
                    .maximum_debug_samples = 48u};
        case water_quality::high:
            return {.maximum_frequency_index = 20u,
                    .foam_history_samples = 3u,
                    .foam_history_interval_seconds = 0.25f,
                    .maximum_debug_samples = 64u};
        case water_quality::ultra:
            return {.maximum_frequency_index = 20u,
                    .foam_history_samples = 4u,
                    .foam_history_interval_seconds = 0.2f,
                    .maximum_debug_samples = 96u};
    }
    return {};
}

water_query_system::water_query_system() : state_(std::make_unique<implementation>()) {}

water_query_system::~water_query_system() = default;
water_query_system::water_query_system(water_query_system&&) noexcept = default;
water_query_system& water_query_system::operator=(water_query_system&&) noexcept = default;

bool water_query_system::prepare(const water_registry& registry, water_body_handle body)
{
    const auto* descriptor = registry.get(body);
    if (!descriptor || !descriptor->enabled || !descriptor->queries_enabled) return false;
    const auto signature = settings_signature(*descriptor);
    const auto key = body_key(body);
    const auto found = state_->bodies.find(key);
    if (found != state_->bodies.end() && found->second.settings_signature == signature) return true;

    query_body_cache cache;
    cache.settings_signature = signature;
    cache.profile = query_profile(descriptor->settings.quality);
    if (descriptor->type == water_body_type::ocean)
    {
        const auto simulation_profile = ocean_profile(descriptor->settings.quality);
        const auto parameters = make_ocean_spectrum_parameters(descriptor->settings.simulation);
        cache.choppiness = parameters.choppiness;
        cache.cascade_count = simulation_profile.cascade_count;
        for (std::uint32_t cascade_index = 0u; cascade_index < cache.cascade_count; ++cascade_index)
        {
            const auto& cascade = simulation_profile.cascades[cascade_index];
            const auto initial = initialize_ocean_spectrum(parameters, cascade, cascade_index);
            if (initial.empty()) return false;
            auto& modes = cache.cascades[cascade_index].modes;
            const auto frequency_limit = std::min(cache.profile.maximum_frequency_index, cascade.resolution / 2u);
            const auto diameter = frequency_limit * 2u + 1u;
            modes.reserve(static_cast<std::size_t>(diameter) * diameter);
            const float wave_scale = two_pi / cascade.physical_length;
            const float inverse_sample_count =
                1.0f / static_cast<float>(static_cast<std::uint64_t>(cascade.resolution) * cascade.resolution);
            for (std::uint32_t y = 0u; y < cascade.resolution; ++y)
                for (std::uint32_t x = 0u; x < cascade.resolution; ++x)
                {
                    const int frequency_x = signed_frequency(x, cascade.resolution);
                    const int frequency_z = signed_frequency(y, cascade.resolution);
                    if (std::abs(frequency_x) > static_cast<int>(frequency_limit) ||
                        std::abs(frequency_z) > static_cast<int>(frequency_limit) ||
                        (frequency_x == 0 && frequency_z == 0))
                        continue;
                    const std::uint32_t mirror_x = (cascade.resolution - x) % cascade.resolution;
                    const std::uint32_t mirror_y = (cascade.resolution - y) % cascade.resolution;
                    const std::size_t index = static_cast<std::size_t>(y) * cascade.resolution + x;
                    const std::size_t mirror = static_cast<std::size_t>(mirror_y) * cascade.resolution + mirror_x;
                    const float wave_x = static_cast<float>(frequency_x) * wave_scale;
                    const float wave_z = static_cast<float>(frequency_z) * wave_scale;
                    const float wave_number = std::sqrt(wave_x * wave_x + wave_z * wave_z);
                    modes.push_back({.initial = initial[index],
                                     .mirrored_conjugate = std::conj(initial[mirror]),
                                     .wave_x = wave_x,
                                     .wave_z = wave_z,
                                     .angular_frequency = deep_water_angular_frequency(wave_number, parameters.gravity),
                                     .inverse_sample_count = inverse_sample_count});
                }
        }
    }
    state_->bodies.insert_or_assign(key, std::move(cache));
    ++state_->statistics.cache_rebuild_count;
    state_->statistics.cached_body_count = static_cast<std::uint32_t>(state_->bodies.size());
    return true;
}

std::optional<water_surface_sample> water_query_system::implementation::sample_prepared(const water_registry& registry,
                                                                                        water_body_handle body,
                                                                                        const math::vector3f& position,
                                                                                        float simulation_time_seconds)
{
    const auto* descriptor = registry.get(body);
    if (!descriptor || !descriptor->enabled || !descriptor->queries_enabled ||
        !descriptor->bounds.contains(position[0], position[2]))
        return std::nullopt;

    water_surface_sample result{.body = body, .height = descriptor->water_level};
    if (descriptor->type != water_body_type::ocean) return result;
    const auto found = bodies.find(body_key(body));
    if (found == bodies.end()) return result;
    const auto& cache = found->second;
    const float time = std::isfinite(simulation_time_seconds) ? simulation_time_seconds : 0.0f;

    float parameter_x = position[0];
    float parameter_z = position[2];
    query_accumulator surface;
    for (std::uint32_t iteration = 0u; iteration < 2u; ++iteration)
    {
        surface = evaluate(cache, parameter_x, parameter_z, time, true, statistics.spectral_mode_evaluation_count);
        parameter_x = position[0] - surface.displacement[0];
        parameter_z = position[2] - surface.displacement[2];
    }
    surface = evaluate(cache, parameter_x, parameter_z, time, true, statistics.spectral_mode_evaluation_count);
    result.height += surface.displacement[1];
    result.normal = normalized_surface_normal(surface.slope_x, surface.slope_z);
    result.velocity = surface.velocity;

    const auto& foam = descriptor->settings.foam;
    result.foam = ocean_crest_foam(ocean_displacement_jacobian(surface.derivatives), foam);
    for (std::uint32_t history = 1u; history < cache.profile.foam_history_samples; ++history)
    {
        const float age = static_cast<float>(history) * cache.profile.foam_history_interval_seconds;
        const auto previous =
            evaluate(cache, parameter_x, parameter_z, time - age, false, statistics.spectral_mode_evaluation_count);
        const float generated = ocean_crest_foam(ocean_displacement_jacobian(previous.derivatives), foam);
        result.foam = std::max(result.foam, generated * std::exp(-std::max(0.0f, foam.decay) * age));
    }
    return result;
}

std::optional<water_surface_sample> water_query_system::sample(const water_registry& registry,
                                                               const math::vector3f& world_position,
                                                               float simulation_time_seconds)
{
    const auto start = std::chrono::steady_clock::now();
    state_->debug_samples.clear();
    const auto body = registry.resolve(world_position[0], world_position[2]);
    std::optional<water_surface_sample> result;
    if (body && prepare(registry, *body))
        result = state_->sample_prepared(registry, *body, world_position, simulation_time_seconds);
    state_->debug_samples.push_back({.query = {.position = world_position}, .result = result});
    ++state_->statistics.query_count;
    ++state_->statistics.batch_count;
    state_->statistics.last_batch_query_count = 1u;
    state_->statistics.last_batch_milliseconds =
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
    return result;
}

std::optional<water_surface_sample> water_query_system::sample(const water_registry& registry, water_body_handle body,
                                                               const math::vector3f& world_position,
                                                               float simulation_time_seconds)
{
    const auto start = std::chrono::steady_clock::now();
    state_->debug_samples.clear();
    std::optional<water_surface_sample> result;
    if (prepare(registry, body))
        result = state_->sample_prepared(registry, body, world_position, simulation_time_seconds);
    state_->debug_samples.push_back({.query = {.position = world_position, .body = body}, .result = result});
    ++state_->statistics.query_count;
    ++state_->statistics.batch_count;
    state_->statistics.last_batch_query_count = 1u;
    state_->statistics.last_batch_milliseconds =
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
    return result;
}

std::size_t water_query_system::sample(const water_registry& registry, std::span<const water_query> queries,
                                       std::span<water_surface_sample> results, float simulation_time_seconds)
{
    const auto start = std::chrono::steady_clock::now();
    state_->debug_samples.clear();
    const std::size_t count = std::min(queries.size(), results.size());
    std::size_t resolved{};
    std::uint32_t debug_limit{32u};
    for (std::size_t index = 0u; index < count; ++index)
    {
        results[index] = {};
        const auto& query = queries[index];
        std::optional<water_body_handle> body;
        if (query.body.valid())
            body = query.body;
        else
            body = registry.resolve(query.position[0], query.position[2]);
        std::optional<water_surface_sample> result;
        if (body && prepare(registry, *body))
        {
            result = state_->sample_prepared(registry, *body, query.position, simulation_time_seconds);
            const auto found = state_->bodies.find(body_key(*body));
            if (found != state_->bodies.end())
                debug_limit = std::max(debug_limit, found->second.profile.maximum_debug_samples);
        }
        if (result)
        {
            results[index] = *result;
            ++resolved;
        }
        if (state_->debug_samples.size() < debug_limit)
            state_->debug_samples.push_back({.query = query, .result = result});
    }
    state_->statistics.query_count += count;
    ++state_->statistics.batch_count;
    state_->statistics.last_batch_query_count =
        static_cast<std::uint32_t>(std::min<std::size_t>(count, std::numeric_limits<std::uint32_t>::max()));
    state_->statistics.last_batch_milliseconds =
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
    return resolved;
}

const water_query_statistics& water_query_system::statistics() const noexcept
{
    return state_->statistics;
}

void water_query_system::reset_statistics() noexcept
{
    const auto cached = state_->statistics.cached_body_count;
    state_->statistics = {};
    state_->statistics.cached_body_count = cached;
}

std::span<const water_query_debug_sample> water_query_system::debug_samples() const noexcept
{
    return state_->debug_samples;
}

void water_query_system::clear() noexcept
{
    state_->bodies.clear();
    state_->debug_samples.clear();
    state_->statistics.cached_body_count = 0u;
}

} // namespace arc::water
