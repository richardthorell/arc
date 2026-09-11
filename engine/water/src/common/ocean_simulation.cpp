#include <arc/water/ocean_simulation.h>

#include <algorithm>
#include <bit>
#include <cmath>
#include <limits>

namespace arc::water
{
namespace
{

constexpr float pi = 3.14159265358979323846f;
constexpr float two_pi = 2.0f * pi;
constexpr float minimum_positive = 1.0e-6f;

std::uint64_t mix_bits(std::uint64_t value) noexcept
{
    value += 0x9e3779b97f4a7c15ull;
    value = (value ^ (value >> 30u)) * 0xbf58476d1ce4e5b9ull;
    value = (value ^ (value >> 27u)) * 0x94d049bb133111ebull;
    return value ^ (value >> 31u);
}

float unit_float(std::uint64_t value) noexcept
{
    constexpr float inverse = 1.0f / 16777216.0f;
    return (static_cast<float>((mix_bits(value) >> 40u) & 0x00ffffffu) + 0.5f) * inverse;
}

std::complex<float> deterministic_gaussian(std::uint64_t seed, std::uint32_t cascade_index, std::uint32_t x,
                                           std::uint32_t y) noexcept
{
    const std::uint64_t coordinate = (static_cast<std::uint64_t>(cascade_index) << 56u) ^
                                     (static_cast<std::uint64_t>(x) << 28u) ^ static_cast<std::uint64_t>(y);
    const float u1 = std::max(unit_float(seed ^ coordinate), minimum_positive);
    const float u2 = unit_float(seed ^ coordinate ^ 0xd1b54a32d192ed03ull);
    const float radius = std::sqrt(-2.0f * std::log(u1));
    const float angle = two_pi * u2;
    return {radius * std::cos(angle), radius * std::sin(angle)};
}

int signed_frequency(std::uint32_t coordinate, std::uint32_t resolution) noexcept
{
    const auto value = static_cast<int>(coordinate);
    const auto size = static_cast<int>(resolution);
    return value <= size / 2 ? value : value - size;
}

math::vector2f wave_vector(std::uint32_t x, std::uint32_t y, const ocean_cascade_descriptor& cascade) noexcept
{
    const float scale = two_pi / cascade.physical_length;
    return {static_cast<float>(signed_frequency(x, cascade.resolution)) * scale,
            static_cast<float>(signed_frequency(y, cascade.resolution)) * scale};
}

float length(const math::vector2f& value) noexcept
{
    return std::sqrt(value[0] * value[0] + value[1] * value[1]);
}

math::vector2f normalize_or_default(const math::vector2f& value) noexcept
{
    const float magnitude = length(value);
    if (magnitude <= minimum_positive || !std::isfinite(magnitude)) return {1.0f, 0.0f};
    return {value[0] / magnitude, value[1] / magnitude};
}

float initial_variance(const ocean_spectrum_parameters& parameters, const ocean_cascade_descriptor& cascade,
                       const math::vector2f& k) noexcept
{
    const float wave_number = length(k);
    if (wave_number <= minimum_positive) return 0.0f;
    const float wavelength = two_pi / wave_number;
    if (wavelength < cascade.minimum_wavelength || wavelength > cascade.maximum_wavelength) return 0.0f;

    const float omega = deep_water_angular_frequency(wave_number, parameters.gravity);
    const float frequency_density = jonswap_spectral_density(omega, parameters);
    if (!(frequency_density > 0.0f)) return 0.0f;

    const math::vector2f direction{k[0] / wave_number, k[1] / wave_number};
    const float directional = ocean_directional_spreading(direction, parameters);
    const float group_velocity = 0.5f * std::sqrt(parameters.gravity / wave_number);
    const float radial_jacobian = std::max(two_pi * wave_number, minimum_positive);
    const float density_2d = frequency_density * group_velocity * directional / radial_jacobian;
    const float delta_k = two_pi / cascade.physical_length;
    const float damping = std::exp(-parameters.short_wave_damping * wave_number * wave_number);
    return std::max(0.0f, density_2d * delta_k * delta_k * damping * parameters.amplitude * parameters.amplitude);
}

void inverse_fft_1d(std::span<std::complex<float>> values) noexcept
{
    const std::size_t count = values.size();
    for (std::size_t index = 1u, reversed = 0u; index < count; ++index)
    {
        std::size_t bit = count >> 1u;
        for (; (reversed & bit) != 0u; bit >>= 1u)
            reversed ^= bit;
        reversed ^= bit;
        if (index < reversed) std::swap(values[index], values[reversed]);
    }

    for (std::size_t width = 2u; width <= count; width <<= 1u)
    {
        const float angle = two_pi / static_cast<float>(width);
        const std::complex<float> step{std::cos(angle), std::sin(angle)};
        for (std::size_t offset = 0u; offset < count; offset += width)
        {
            std::complex<float> phase{1.0f, 0.0f};
            for (std::size_t lane = 0u; lane < width / 2u; ++lane)
            {
                const auto even = values[offset + lane];
                const auto odd = values[offset + lane + width / 2u] * phase;
                values[offset + lane] = even + odd;
                values[offset + lane + width / 2u] = even - odd;
                phase *= step;
            }
        }
    }
    const float inverse_count = 1.0f / static_cast<float>(count);
    for (auto& value : values)
        value *= inverse_count;
}

bool valid_cascade(const ocean_cascade_descriptor& cascade) noexcept
{
    return cascade.resolution >= 2u && std::has_single_bit(cascade.resolution) && cascade.physical_length > 0.0f &&
           cascade.minimum_wavelength > 0.0f && cascade.maximum_wavelength >= cascade.minimum_wavelength;
}

} // namespace

ocean_simulation_profile ocean_profile(water_quality quality) noexcept
{
    ocean_simulation_profile result;
    result.cascades = {{{64u, 512.0f, 32.0f, 512.0f},
                        {64u, 128.0f, 4.0f, 64.0f},
                        {64u, 32.0f, 0.5f, 16.0f},
                        {64u, 8.0f, 0.125f, 4.0f}}};
    switch (quality)
    {
        case water_quality::low:
            result.cascade_count = 1u;
            result.update_interval_frames = 2u;
            break;
        case water_quality::medium:
            result.cascade_count = 2u;
            for (auto& cascade : result.cascades)
                cascade.resolution = 128u;
            break;
        case water_quality::high:
            result.cascade_count = 3u;
            for (auto& cascade : result.cascades)
                cascade.resolution = 256u;
            break;
        case water_quality::ultra:
            result.cascade_count = 4u;
            for (auto& cascade : result.cascades)
                cascade.resolution = 512u;
            result.cascades[3].resolution = 256u;
            break;
    }
    return result;
}

ocean_spectrum_parameters make_ocean_spectrum_parameters(const water_simulation_settings& settings) noexcept
{
    ocean_spectrum_parameters result;
    result.wind_speed = std::max(0.0f, settings.wind_speed);
    result.wind_direction = normalize_or_default(settings.wind_direction);
    result.fetch_length = std::max(1.0f, settings.fetch_length);
    result.amplitude = std::max(0.0f, settings.wave_amplitude);
    result.choppiness = std::max(0.0f, settings.choppiness);
    result.seed = settings.seed;
    return result;
}

float deep_water_angular_frequency(float wave_number, float gravity) noexcept
{
    if (!(wave_number > 0.0f) || !(gravity > 0.0f) || !std::isfinite(wave_number) || !std::isfinite(gravity))
        return 0.0f;
    return std::sqrt(gravity * wave_number);
}

float jonswap_spectral_density(float angular_frequency, const ocean_spectrum_parameters& parameters) noexcept
{
    if (!(angular_frequency > minimum_positive) || !(parameters.wind_speed > minimum_positive) ||
        !(parameters.fetch_length > minimum_positive) || !(parameters.gravity > minimum_positive))
        return 0.0f;

    const float inverse_wave_age =
        parameters.gravity * parameters.fetch_length / (parameters.wind_speed * parameters.wind_speed);
    const float peak_omega =
        22.0f * std::pow(parameters.gravity * parameters.gravity / (parameters.wind_speed * parameters.fetch_length),
                         1.0f / 3.0f);
    const float alpha = 0.076f * std::pow(std::max(inverse_wave_age, minimum_positive), -0.22f);
    const float sigma = angular_frequency <= peak_omega ? 0.07f : 0.09f;
    const float normalized_delta = (angular_frequency - peak_omega) / (sigma * peak_omega);
    const float peak_shape = std::exp(-0.5f * normalized_delta * normalized_delta);
    const float peak = std::pow(std::max(parameters.peak_enhancement, 1.0f), peak_shape);
    const float cutoff = std::exp(-1.25f * std::pow(peak_omega / angular_frequency, 4.0f));
    const float density =
        alpha * parameters.gravity * parameters.gravity * cutoff * peak / std::pow(angular_frequency, 5.0f);
    return std::isfinite(density) ? std::max(0.0f, density) : 0.0f;
}

float ocean_directional_spreading(const math::vector2f& wave_direction,
                                  const ocean_spectrum_parameters& parameters) noexcept
{
    const auto direction = normalize_or_default(wave_direction);
    const auto wind = normalize_or_default(parameters.wind_direction);
    const float cosine = std::clamp(direction[0] * wind[0] + direction[1] * wind[1], -1.0f, 1.0f);
    const float half_angle_cosine = std::sqrt(std::max(0.0f, 0.5f * (1.0f + cosine)));
    return std::pow(half_angle_cosine, std::max(0.0f, parameters.directional_spread));
}

std::vector<std::complex<float>> initialize_ocean_spectrum(const ocean_spectrum_parameters& parameters,
                                                           const ocean_cascade_descriptor& cascade,
                                                           std::uint32_t cascade_index)
{
    if (!valid_cascade(cascade)) return {};
    const auto count = static_cast<std::size_t>(cascade.resolution) * cascade.resolution;
    std::vector<std::complex<float>> result(count);
    for (std::uint32_t y = 0u; y < cascade.resolution; ++y)
        for (std::uint32_t x = 0u; x < cascade.resolution; ++x)
        {
            const float variance = initial_variance(parameters, cascade, wave_vector(x, y, cascade));
            const auto gaussian = deterministic_gaussian(parameters.seed, cascade_index, x, y);
            result[static_cast<std::size_t>(y) * cascade.resolution + x] = gaussian * std::sqrt(0.5f * variance);
        }
    return result;
}

ocean_frequency_fields evolve_ocean_spectrum(std::span<const std::complex<float>> initial_spectrum,
                                             const ocean_spectrum_parameters& parameters,
                                             const ocean_cascade_descriptor& cascade, float time_seconds)
{
    ocean_frequency_fields result;
    if (!valid_cascade(cascade) ||
        initial_spectrum.size() != static_cast<std::size_t>(cascade.resolution) * cascade.resolution)
        return result;

    result.resolution = cascade.resolution;
    const auto count = initial_spectrum.size();
    result.displacement_x.resize(count);
    result.displacement_y.resize(count);
    result.displacement_z.resize(count);
    result.slope_x.resize(count);
    result.slope_z.resize(count);
    result.velocity_x.resize(count);
    result.velocity_y.resize(count);
    result.velocity_z.resize(count);

    const std::complex<float> imaginary{0.0f, 1.0f};
    for (std::uint32_t y = 0u; y < cascade.resolution; ++y)
        for (std::uint32_t x = 0u; x < cascade.resolution; ++x)
        {
            const std::size_t index = static_cast<std::size_t>(y) * cascade.resolution + x;
            const std::uint32_t mirror_x = (cascade.resolution - x) % cascade.resolution;
            const std::uint32_t mirror_y = (cascade.resolution - y) % cascade.resolution;
            const std::size_t mirror = static_cast<std::size_t>(mirror_y) * cascade.resolution + mirror_x;
            const auto k = wave_vector(x, y, cascade);
            const float wave_number = length(k);
            if (wave_number <= minimum_positive) continue;

            const float omega = deep_water_angular_frequency(wave_number, parameters.gravity);
            const float phase = omega * time_seconds;
            const std::complex<float> forward{std::cos(phase), std::sin(phase)};
            const auto backward = std::conj(forward);
            const auto a = initial_spectrum[index] * forward;
            const auto b = std::conj(initial_spectrum[mirror]) * backward;
            const auto height = a + b;
            const auto height_velocity = imaginary * omega * (a - b);
            const float normalized_x = k[0] / wave_number;
            const float normalized_z = k[1] / wave_number;
            const auto horizontal_x = imaginary * (-normalized_x * parameters.choppiness) * height;
            const auto horizontal_z = imaginary * (-normalized_z * parameters.choppiness) * height;

            result.displacement_x[index] = horizontal_x;
            result.displacement_y[index] = height;
            result.displacement_z[index] = horizontal_z;
            result.slope_x[index] = imaginary * k[0] * height;
            result.slope_z[index] = imaginary * k[1] * height;
            result.velocity_x[index] = imaginary * (-normalized_x * parameters.choppiness) * height_velocity;
            result.velocity_y[index] = height_velocity;
            result.velocity_z[index] = imaginary * (-normalized_z * parameters.choppiness) * height_velocity;
        }
    return result;
}

bool inverse_fft_2d(std::span<std::complex<float>> values, std::uint32_t resolution) noexcept
{
    if (resolution < 2u || !std::has_single_bit(resolution) ||
        values.size() != static_cast<std::size_t>(resolution) * resolution)
        return false;

    for (std::uint32_t row = 0u; row < resolution; ++row)
        inverse_fft_1d(values.subspan(static_cast<std::size_t>(row) * resolution, resolution));

    std::vector<std::complex<float>> column(resolution);
    for (std::uint32_t x = 0u; x < resolution; ++x)
    {
        for (std::uint32_t y = 0u; y < resolution; ++y)
            column[y] = values[static_cast<std::size_t>(y) * resolution + x];
        inverse_fft_1d(column);
        for (std::uint32_t y = 0u; y < resolution; ++y)
            values[static_cast<std::size_t>(y) * resolution + x] = column[y];
    }
    return true;
}

std::vector<ocean_surface_point> evaluate_ocean_reference(const ocean_spectrum_parameters& parameters,
                                                          const ocean_cascade_descriptor& cascade, float time_seconds,
                                                          std::uint32_t cascade_index)
{
    const auto initial = initialize_ocean_spectrum(parameters, cascade, cascade_index);
    auto fields = evolve_ocean_spectrum(initial, parameters, cascade, time_seconds);
    if (fields.resolution == 0u) return {};

    std::array<std::vector<std::complex<float>>*, 8u> frequency_fields{
        &fields.displacement_x, &fields.displacement_y, &fields.displacement_z, &fields.slope_x,
        &fields.slope_z,        &fields.velocity_x,     &fields.velocity_y,     &fields.velocity_z};
    for (auto* field : frequency_fields)
        if (!inverse_fft_2d(*field, fields.resolution)) return {};

    std::vector<ocean_surface_point> result(initial.size());
    for (std::size_t index = 0u; index < result.size(); ++index)
    {
        const float slope_x = fields.slope_x[index].real();
        const float slope_z = fields.slope_z[index].real();
        const float inverse_length = 1.0f / std::sqrt(slope_x * slope_x + 1.0f + slope_z * slope_z);
        result[index] = {.displacement = {fields.displacement_x[index].real(), fields.displacement_y[index].real(),
                                          fields.displacement_z[index].real()},
                         .normal = {-slope_x * inverse_length, inverse_length, -slope_z * inverse_length},
                         .velocity = {fields.velocity_x[index].real(), fields.velocity_y[index].real(),
                                      fields.velocity_z[index].real()}};
    }
    return result;
}

} // namespace arc::water
