#include <arc/render/fluid_surface.h>

#include <catch2/catch_test_macros.hpp>

TEST_CASE("Fluid Surface supports non-Water surfaces without Water simulation semantics")
{
    arc::render::fluid_surface_render_instance surface;
    surface.source_kind = arc::render::fluid_surface_source_kind::static_surface;
    surface.channels.normals = true;
    surface.channels.thickness = true;

    REQUIRE_FALSE(surface.is_water());
    REQUIRE(surface.channels.normals);
    REQUIRE(surface.channels.thickness);
    REQUIRE_FALSE(surface.channels.displacement);
    REQUIRE_FALSE(surface.channels.velocity);
    REQUIRE_FALSE(surface.channels.foam);
}

TEST_CASE("Water is an explicit Fluid Surface specialization")
{
    arc::render::fluid_surface_render_instance surface;
    surface.source_kind = arc::render::fluid_surface_source_kind::water;
    surface.channels = {
        .displacement = true,
        .normals = true,
        .velocity = true,
        .foam = true,
        .thickness = true,
    };
    surface.water.type = arc::water::water_body_type::ocean;
    surface.water.settings.simulation.wind_speed = 18.0f;
    surface.water.settings.appearance.refraction_strength = 0.06f;

    REQUIRE(surface.is_water());
    REQUIRE(surface.water.type == arc::water::water_body_type::ocean);
    REQUIRE(surface.water.settings.simulation.wind_speed == 18.0f);
    REQUIRE(surface.water.settings.appearance.refraction_strength == 0.06f);
    REQUIRE(surface.channels.displacement);
    REQUIRE(surface.channels.normals);
    REQUIRE(surface.channels.velocity);
    REQUIRE(surface.channels.foam);
    REQUIRE(surface.channels.thickness);
}
