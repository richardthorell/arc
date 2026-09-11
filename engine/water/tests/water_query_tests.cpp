#include <arc/water/water_query.h>

#include <arc/water/ocean_simulation.h>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <cmath>
#include <vector>

TEST_CASE("Water query quality presets bound spectral work while retaining one API")
{
    const auto low = arc::water::query_profile(arc::water::water_quality::low);
    const auto medium = arc::water::query_profile(arc::water::water_quality::medium);
    const auto high = arc::water::query_profile(arc::water::water_quality::high);
    const auto ultra = arc::water::query_profile(arc::water::water_quality::ultra);

    CHECK(low.maximum_frequency_index >= medium.maximum_frequency_index);
    CHECK(medium.maximum_frequency_index >= high.maximum_frequency_index);
    CHECK(high.maximum_frequency_index == ultra.maximum_frequency_index);
    CHECK(low.foam_history_samples < medium.foam_history_samples);
    CHECK(medium.foam_history_samples < high.foam_history_samples);
    CHECK(high.foam_history_samples < ultra.foam_history_samples);
    CHECK(low.maximum_debug_samples < ultra.maximum_debug_samples);
}

TEST_CASE("CPU Water query matches the deterministic low-quality reference spectrum")
{
    arc::water::water_registry registry;
    arc::water::water_body_descriptor descriptor;
    descriptor.water_level = 3.25f;
    descriptor.settings.quality = arc::water::water_quality::low;
    descriptor.settings.simulation.choppiness = 0.0f;
    descriptor.settings.simulation.seed = 481516u;
    const auto body = registry.add(descriptor);

    arc::water::water_query_system queries;
    REQUIRE(queries.prepare(registry, body));
    const auto profile = arc::water::ocean_profile(arc::water::water_quality::low);
    const auto parameters = arc::water::make_ocean_spectrum_parameters(descriptor.settings.simulation);
    const auto reference = arc::water::evaluate_ocean_reference(parameters, profile.cascades[0], 5.75f);
    REQUIRE(reference.size() == 64u * 64u);

    constexpr std::uint32_t grid_x = 13u;
    constexpr std::uint32_t grid_z = 27u;
    const float world_x = static_cast<float>(grid_x) * profile.cascades[0].physical_length / 64.0f;
    const float world_z = static_cast<float>(grid_z) * profile.cascades[0].physical_length / 64.0f;
    const auto sample = queries.sample(registry, body, {world_x, 0.0f, world_z}, 5.75f);
    REQUIRE(sample.has_value());
    const auto& expected = reference[grid_z * 64u + grid_x];
    CHECK(sample->height == Catch::Approx(descriptor.water_level + expected.displacement[1]).margin(2.0e-4f));
    for (std::size_t axis = 0u; axis < 3u; ++axis)
    {
        CHECK(sample->normal[axis] == Catch::Approx(expected.normal[axis]).margin(2.0e-4f));
        CHECK(sample->velocity[axis] == Catch::Approx(expected.velocity[axis]).margin(2.0e-4f));
    }
    CHECK(sample->foam == 0.0f);
}

TEST_CASE("Batched Water queries reuse spectra and retain bounded debug samples")
{
    arc::water::water_registry registry;
    arc::water::water_body_descriptor descriptor;
    descriptor.settings.quality = arc::water::water_quality::low;
    descriptor.settings.simulation.seed = 90210u;
    const auto body = registry.add(descriptor);
    arc::water::water_query_system query_system;

    std::vector<arc::water::water_query> queries(256u);
    std::vector<arc::water::water_surface_sample> samples(queries.size());
    for (std::size_t index = 0u; index < queries.size(); ++index)
        queries[index].position = {static_cast<float>(index % 32u) * 3.0f, 0.0f,
                                   static_cast<float>(index / 32u) * 5.0f};

    CHECK(query_system.sample(registry, queries, samples, 2.0f) == queries.size());
    const auto first_statistics = query_system.statistics();
    CHECK(first_statistics.query_count == queries.size());
    CHECK(first_statistics.batch_count == 1u);
    CHECK(first_statistics.cache_rebuild_count == 1u);
    CHECK(first_statistics.cached_body_count == 1u);
    CHECK(first_statistics.spectral_mode_evaluation_count <= queries.size() * 3u * 64u * 64u);
    CHECK(query_system.debug_samples().size() ==
          arc::water::query_profile(descriptor.settings.quality).maximum_debug_samples);

    for (const auto& sample : samples)
    {
        CHECK(sample.body == body);
        CHECK(std::isfinite(sample.height));
        CHECK(std::isfinite(sample.normal[1]));
        CHECK(std::isfinite(sample.velocity[1]));
        CHECK(sample.foam >= 0.0f);
        CHECK(sample.foam <= 1.0f);
    }

    CHECK(query_system.sample(registry, queries, samples, 2.25f) == queries.size());
    CHECK(query_system.statistics().cache_rebuild_count == 1u);
    CHECK(query_system.statistics().batch_count == 2u);
}

TEST_CASE("Water query routing honors body bounds and query opt-out")
{
    arc::water::water_registry registry;
    const auto lake = registry.add({.type = arc::water::water_body_type::lake,
                                    .bounds = {-10.0f, -10.0f, 10.0f, 10.0f, true},
                                    .water_level = 7.0f});
    arc::water::water_query_system queries;

    const auto inside = queries.sample(registry, {2.0f, 0.0f, -3.0f}, 1.0f);
    REQUIRE(inside.has_value());
    CHECK(inside->body == lake);
    CHECK(inside->height == 7.0f);
    CHECK_FALSE(queries.sample(registry, {20.0f, 0.0f, 20.0f}, 1.0f).has_value());

    auto disabled = *registry.get(lake);
    disabled.queries_enabled = false;
    REQUIRE(registry.update(lake, disabled));
    CHECK_FALSE(queries.sample(registry, lake, {0.0f, 0.0f, 0.0f}, 1.0f).has_value());
}
