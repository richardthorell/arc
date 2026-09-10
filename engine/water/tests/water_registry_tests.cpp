#include <arc/water/water_registry.h>

#include <catch2/catch_test_macros.hpp>

TEST_CASE("Water registry keeps handles stable and rejects stale generations")
{
    arc::water::water_registry registry;
    const auto first = registry.add({.label = "Ocean"});
    REQUIRE(registry.alive(first));
    REQUIRE(registry.remove(first));
    CHECK_FALSE(registry.alive(first));

    const auto replacement = registry.add({.label = "Replacement"});
    CHECK(replacement.index == first.index);
    CHECK(replacement.generation != first.generation);
    CHECK(registry.size() == 1);
}

TEST_CASE("Water registry resolves priority, bounded ownership, and Ocean fallback deterministically")
{
    arc::water::water_registry registry;
    const auto ocean = registry.add({.type = arc::water::water_body_type::ocean, .water_level = 1.0f, .label = "Ocean"});
    const auto lake = registry.add({.type = arc::water::water_body_type::lake,
                                    .bounds = {-20.0f, -20.0f, 20.0f, 20.0f, true},
                                    .water_level = 3.0f,
                                    .label = "Lake"});
    const auto pond = registry.add({.type = arc::water::water_body_type::lake,
                                    .bounds = {-5.0f, -5.0f, 5.0f, 5.0f, true},
                                    .water_level = 4.0f,
                                    .label = "Pond"});

    CHECK(registry.resolve(100.0f, 100.0f) == ocean);
    CHECK(registry.resolve(10.0f, 10.0f) == lake);
    CHECK(registry.resolve(0.0f, 0.0f) == pond);

    auto prioritized = *registry.get(lake);
    prioritized.priority = 10;
    REQUIRE(registry.update(lake, prioritized));
    CHECK(registry.resolve(0.0f, 0.0f) == lake);

    const auto sample = registry.sample(0.0f, 0.0f);
    REQUIRE(sample.has_value());
    CHECK(sample->height == 3.0f);
    CHECK(sample->normal[1] == 1.0f);
}
