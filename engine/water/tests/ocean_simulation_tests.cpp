#include <arc/water/ocean_simulation.h>

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include <algorithm>
#include <cmath>
#include <complex>
#include <numeric>

namespace
{

float displacement_energy(const std::vector<arc::water::ocean_surface_point>& points)
{
    if (points.empty()) return 0.0f;
    const float sum = std::accumulate(points.begin(), points.end(), 0.0f, [](float value, const auto& point)
                                      { return value + point.displacement[1] * point.displacement[1]; });
    return sum / static_cast<float>(points.size());
}

} // namespace

TEST_CASE("Ocean quality profiles scale cascade and FFT budgets without "
          "changing the contract")
{
    const auto low = arc::water::ocean_profile(arc::water::water_quality::low);
    const auto medium = arc::water::ocean_profile(arc::water::water_quality::medium);
    const auto high = arc::water::ocean_profile(arc::water::water_quality::high);
    const auto ultra = arc::water::ocean_profile(arc::water::water_quality::ultra);

    CHECK(low.cascade_count == 1u);
    CHECK(medium.cascade_count == 2u);
    CHECK(high.cascade_count == 3u);
    CHECK(ultra.cascade_count == 4u);
    CHECK(low.update_interval_frames == 2u);
    CHECK(high.cascades[0].resolution < ultra.cascades[0].resolution);
    CHECK(high.cascades[0].physical_length > high.cascades[1].physical_length);
    CHECK(high.cascades[1].physical_length > high.cascades[2].physical_length);
}

TEST_CASE("Ocean spectrum parameter conversion normalizes authoring input")
{
    arc::water::water_simulation_settings settings;
    settings.wind_speed = -4.0f;
    settings.wind_direction = {3.0f, 4.0f};
    settings.fetch_length = 0.0f;
    settings.wave_amplitude = -2.0f;
    settings.choppiness = -1.0f;
    settings.seed = 90210u;

    const auto parameters = arc::water::make_ocean_spectrum_parameters(settings);
    CHECK(parameters.wind_speed == 0.0f);
    CHECK(parameters.wind_direction[0] == Catch::Approx(0.6f));
    CHECK(parameters.wind_direction[1] == Catch::Approx(0.8f));
    CHECK(parameters.fetch_length == 1.0f);
    CHECK(parameters.amplitude == 0.0f);
    CHECK(parameters.choppiness == 0.0f);
    CHECK(parameters.seed == 90210u);
}

TEST_CASE("Deep-water dispersion and JONSWAP energy are finite and physical")
{
    arc::water::ocean_spectrum_parameters parameters;
    const float first = arc::water::deep_water_angular_frequency(0.25f);
    const float second = arc::water::deep_water_angular_frequency(1.0f);
    CHECK(first == Catch::Approx(std::sqrt(9.81f * 0.25f)));
    CHECK(second > first);
    CHECK(arc::water::deep_water_angular_frequency(0.0f) == 0.0f);

    const float energy = arc::water::jonswap_spectral_density(first, parameters);
    CHECK(std::isfinite(energy));
    CHECK(energy > 0.0f);
    parameters.wind_speed = 0.0f;
    CHECK(arc::water::jonswap_spectral_density(first, parameters) == 0.0f);
}

TEST_CASE("Directional spreading follows authored wind direction")
{
    arc::water::ocean_spectrum_parameters parameters;
    parameters.wind_direction = {1.0f, 0.0f};
    const float aligned = arc::water::ocean_directional_spreading({1.0f, 0.0f}, parameters);
    const float perpendicular = arc::water::ocean_directional_spreading({0.0f, 1.0f}, parameters);
    const float opposite = arc::water::ocean_directional_spreading({-1.0f, 0.0f}, parameters);
    CHECK(aligned == Catch::Approx(1.0f));
    CHECK(aligned > perpendicular);
    CHECK(perpendicular > opposite);
    CHECK(opposite == Catch::Approx(0.0f).margin(1.0e-6f));
}

TEST_CASE("Ocean initial spectra are deterministic and seed-dependent")
{
    arc::water::ocean_spectrum_parameters parameters;
    const arc::water::ocean_cascade_descriptor cascade{16u, 128.0f, 2.0f, 128.0f};
    const auto first = arc::water::initialize_ocean_spectrum(parameters, cascade, 1u);
    const auto repeated = arc::water::initialize_ocean_spectrum(parameters, cascade, 1u);
    CHECK(first == repeated);
    REQUIRE(first.size() == 256u);
    CHECK(std::any_of(first.begin(), first.end(), [](const auto value) { return std::abs(value) > 0.0f; }));

    parameters.seed += 1u;
    const auto changed = arc::water::initialize_ocean_spectrum(parameters, cascade, 1u);
    CHECK(changed != first);
}

TEST_CASE("Tessendorf evolution preserves Hermitian height symmetry")
{
    const arc::water::ocean_spectrum_parameters parameters;
    const arc::water::ocean_cascade_descriptor cascade{16u, 128.0f, 2.0f, 128.0f};
    const auto initial = arc::water::initialize_ocean_spectrum(parameters, cascade);
    const auto fields = arc::water::evolve_ocean_spectrum(initial, parameters, cascade, 7.25f);
    REQUIRE(fields.resolution == cascade.resolution);

    for (std::uint32_t y = 0u; y < cascade.resolution; ++y)
        for (std::uint32_t x = 0u; x < cascade.resolution; ++x)
        {
            const auto index = static_cast<std::size_t>(y) * cascade.resolution + x;
            const auto mirror_x = (cascade.resolution - x) % cascade.resolution;
            const auto mirror_y = (cascade.resolution - y) % cascade.resolution;
            const auto mirror = static_cast<std::size_t>(mirror_y) * cascade.resolution + mirror_x;
            CHECK(fields.displacement_y[index].real() ==
                  Catch::Approx(fields.displacement_y[mirror].real()).margin(1.0e-5f));
            CHECK(fields.displacement_y[index].imag() ==
                  Catch::Approx(-fields.displacement_y[mirror].imag()).margin(1.0e-5f));
        }
}

TEST_CASE("Normalized inverse FFT resolves a DC spectrum to a constant field")
{
    std::vector<std::complex<float>> values(64u);
    values[0] = {64.0f, 0.0f};
    REQUIRE(arc::water::inverse_fft_2d(values, 8u));
    for (const auto value : values)
    {
        CHECK(value.real() == Catch::Approx(1.0f).margin(1.0e-5f));
        CHECK(value.imag() == Catch::Approx(0.0f).margin(1.0e-5f));
    }
    CHECK_FALSE(arc::water::inverse_fft_2d(values, 7u));
}

TEST_CASE("CPU reference Ocean produces deterministic displacement normals and "
          "velocity")
{
    const arc::water::ocean_spectrum_parameters parameters;
    const arc::water::ocean_cascade_descriptor cascade{16u, 128.0f, 2.0f, 128.0f};
    const auto first = arc::water::evaluate_ocean_reference(parameters, cascade, 3.0f);
    const auto repeated = arc::water::evaluate_ocean_reference(parameters, cascade, 3.0f);
    const auto later = arc::water::evaluate_ocean_reference(parameters, cascade, 3.5f);

    REQUIRE(first.size() == 256u);
    REQUIRE(repeated.size() == first.size());
    REQUIRE(later.size() == first.size());
    for (std::size_t index = 0u; index < first.size(); ++index)
    {
        for (std::size_t axis = 0u; axis < 3u; ++axis)
        {
            CHECK(first[index].displacement[axis] == repeated[index].displacement[axis]);
            CHECK(first[index].normal[axis] == repeated[index].normal[axis]);
            CHECK(first[index].velocity[axis] == repeated[index].velocity[axis]);
        }
        const auto& normal = first[index].normal;
        CHECK(std::sqrt(normal[0] * normal[0] + normal[1] * normal[1] + normal[2] * normal[2]) ==
              Catch::Approx(1.0f).margin(1.0e-4f));
    }
    CHECK(displacement_energy(first) > 0.0f);
    bool changed{};
    for (std::size_t index = 0u; index < first.size(); ++index)
        for (std::size_t axis = 0u; axis < 3u; ++axis)
            changed = changed || first[index].displacement[axis] != later[index].displacement[axis];
    CHECK(changed);
}

TEST_CASE("Three Ocean cascades retain distinct spatial energy bands")
{
    const auto profile = arc::water::ocean_profile(arc::water::water_quality::high);
    auto parameters = arc::water::ocean_spectrum_parameters{};
    std::array<float, 3u> energies{};
    for (std::uint32_t index = 0u; index < 3u; ++index)
    {
        auto cascade = profile.cascades[index];
        cascade.resolution = 16u;
        energies[index] = displacement_energy(arc::water::evaluate_ocean_reference(parameters, cascade, 2.0f, index));
        CHECK(energies[index] > 0.0f);
    }
    CHECK(energies[0] != Catch::Approx(energies[1]));
    CHECK(energies[1] != Catch::Approx(energies[2]));
}
