#pragma once

#include <arc/water/water_types.h>

#include <array>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace arc::water
{

inline constexpr std::uint32_t maximum_ocean_cascades = 4u;

/** @brief One independently band-limited spectral Ocean domain. */
struct ocean_cascade_descriptor
{
    std::uint32_t resolution{256u};
    float physical_length{512.0f};
    float minimum_wavelength{32.0f};
    float maximum_wavelength{512.0f};

    friend constexpr bool operator==(const ocean_cascade_descriptor&,
                                     const ocean_cascade_descriptor&) noexcept = default;
};

/** @brief Quality-scaled simulation budget shared by all renderer backends.
 */
struct ocean_simulation_profile
{
    std::array<ocean_cascade_descriptor, maximum_ocean_cascades> cascades{};
    std::uint32_t cascade_count{3u};
    std::uint32_t update_interval_frames{1u};

    friend constexpr bool operator==(const ocean_simulation_profile&,
                                     const ocean_simulation_profile&) noexcept = default;
};

/** @brief Physical spectrum parameters resolved from artist-facing Water
 * settings. */
struct ocean_spectrum_parameters
{
    float wind_speed{12.0f};
    math::vector2f wind_direction{1.0f, 0.0f};
    float fetch_length{50000.0f};
    float amplitude{1.0f};
    float choppiness{1.0f};
    float gravity{9.81f};
    float peak_enhancement{3.3f};
    float directional_spread{6.0f};
    float short_wave_damping{0.001f};
    std::uint64_t seed{1u};
};

struct ocean_frequency_fields
{
    std::uint32_t resolution{};
    std::vector<std::complex<float>> displacement_x;
    std::vector<std::complex<float>> displacement_y;
    std::vector<std::complex<float>> displacement_z;
    std::vector<std::complex<float>> slope_x;
    std::vector<std::complex<float>> slope_z;
    std::vector<std::complex<float>> velocity_x;
    std::vector<std::complex<float>> velocity_y;
    std::vector<std::complex<float>> velocity_z;
};

struct ocean_surface_point
{
    math::vector3f displacement{};
    math::vector3f normal{0.0f, 1.0f, 0.0f};
    math::vector3f velocity{};
};

/** @brief Resolve stable cascade counts and FFT resolutions for an authored
 * quality level. */
[[nodiscard]] ocean_simulation_profile ocean_profile(water_quality quality) noexcept;

/** @brief Convert the W0 authoring contract into normalized physical spectrum
 * parameters. */
[[nodiscard]] ocean_spectrum_parameters
make_ocean_spectrum_parameters(const water_simulation_settings& settings) noexcept;

/** @brief Deep-water dispersion relation, omega(k) = sqrt(g * |k|). */
[[nodiscard]] float deep_water_angular_frequency(float wave_number, float gravity = 9.81f) noexcept;

/** @brief One-dimensional JONSWAP energy density in angular-frequency space.
 */
[[nodiscard]] float jonswap_spectral_density(float angular_frequency,
                                             const ocean_spectrum_parameters& parameters) noexcept;

/** @brief Normalized directional energy weight around the authored wind
 * direction. */
[[nodiscard]] float ocean_directional_spreading(const math::vector2f& wave_direction,
                                                const ocean_spectrum_parameters& parameters) noexcept;

/** @brief Deterministically initialize one complete complex h0 spectrum. */
[[nodiscard]] std::vector<std::complex<float>> initialize_ocean_spectrum(const ocean_spectrum_parameters& parameters,
                                                                         const ocean_cascade_descriptor& cascade,
                                                                         std::uint32_t cascade_index = 0u);

/** @brief Evolve h0 and derive displacement, slope, and velocity fields in
 * frequency space. */
[[nodiscard]] ocean_frequency_fields evolve_ocean_spectrum(std::span<const std::complex<float>> initial_spectrum,
                                                           const ocean_spectrum_parameters& parameters,
                                                           const ocean_cascade_descriptor& cascade, float time_seconds);

/** @brief In-place normalized inverse FFT for a square power-of-two complex
 * field. */
bool inverse_fft_2d(std::span<std::complex<float>> values, std::uint32_t resolution) noexcept;

/** @brief CPU reference evaluation used by deterministic tests and future
 * compatibility fallbacks. */
[[nodiscard]] std::vector<ocean_surface_point> evaluate_ocean_reference(const ocean_spectrum_parameters& parameters,
                                                                        const ocean_cascade_descriptor& cascade,
                                                                        float time_seconds,
                                                                        std::uint32_t cascade_index = 0u);

} // namespace arc::water
