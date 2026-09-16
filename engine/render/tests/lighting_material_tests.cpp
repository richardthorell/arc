#include <arc/render/render.h>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <atomic>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <thread>
#include <memory>
#include <vector>

#if !defined(ARC_RENDER_TEST_ASSET_ROOT)
#define ARC_RENDER_TEST_ASSET_ROOT "assets"
#endif

TEST_CASE("scene lighting data packs sorted capped light arrays")
{
    std::vector<arc::render::directional_light_event> directional;
    for (std::uint32_t index = 0; index < arc::render::max_directional_lights + 2; ++index)
    {
        directional.push_back({.direction = {0.0f, -1.0f, 0.0f},
                               .color = {1.0f, 1.0f, 1.0f},
                               .intensity = static_cast<float>(index + 1),
                               .label = "sun"});
    }

    std::vector<arc::render::point_light_event> points{
        {.object_id = {.index = 17, .generation = 3},
         .position = {1.0f, 2.0f, 3.0f},
         .color = {1.0f, 0.5f, 0.25f},
         .intensity = 80.0f,
         .range = 4.0f,
         .intensity_unit = arc::render::light_intensity_unit::lumen},
        {.position = {0.0f, 0.0f, 0.0f}, .color = {1.0f, 1.0f, 1.0f}, .intensity = 2.0f, .range = 8.0f}};
    std::vector<arc::render::spot_light_event> spots{{.position = {0.0f, 1.0f, 0.0f},
                                                      .direction = {0.0f, -1.0f, 0.0f},
                                                      .color = {0.8f, 0.9f, 1.0f},
                                                      .intensity = 3.0f,
                                                      .range = 10.0f,
                                                      .inner_angle = 0.2f,
                                                      .outer_angle = 0.7f}};

    arc::render::environment_descriptor environment;
    environment.fallback_color = {0.1f, 0.2f, 0.3f};
    environment.intensity = 1.25f;

    const auto data = arc::render::pack_scene_lighting(directional, points, spots, &environment);
    REQUIRE(data.directional_count == arc::render::max_directional_lights);
    REQUIRE(data.skipped_directional_count == 2);
    REQUIRE(data.directional_lights[0].direction_intensity[3] == Catch::Approx(6.0f));
    REQUIRE(data.point_count == 2);
    REQUIRE(data.point_lights[0].color_intensity[3] == Catch::Approx(80.0f / (4.0f * arc::math::pi<float>)));
    REQUIRE(data.point_lights[0].object_id_shadow[0] == Catch::Approx(17.0f));
    REQUIRE(data.point_lights[0].object_id_shadow[1] == Catch::Approx(3.0f));
    REQUIRE(data.point_lights[0].shadow_parameters[0] == Catch::Approx(-1.0f));
    REQUIRE(data.spot_count == 1);
    REQUIRE(data.spot_lights[0].params[0] == Catch::Approx(0.7f));
    REQUIRE(data.ambient_color_intensity[1] == Catch::Approx(0.2f));
    REQUIRE(data.local_shadow_face_count == 0);
    STATIC_REQUIRE(arc::render::max_local_shadow_faces == 144);

    environment.prefiltered = true;
    environment.diffuse_irradiance = {0.4f, 0.5f, 0.6f};
    environment.diffuse_intensity = 0.75f;
    const auto prefiltered = arc::render::pack_scene_lighting({}, {}, {}, &environment);
    REQUIRE(prefiltered.ambient_color_intensity[0] == Catch::Approx(0.4f));
    REQUIRE(prefiltered.ambient_color_intensity[2] == Catch::Approx(0.6f));
    REQUIRE(prefiltered.ambient_color_intensity[3] == Catch::Approx(0.75f));
}

TEST_CASE("light unit and temperature helpers provide stable defaults")
{
    REQUIRE(arc::render::light_intensity_scale(arc::render::light_intensity_unit::unitless, 2.0f, 4.0f) ==
            Catch::Approx(2.0f));
    REQUIRE(arc::render::light_intensity_scale(arc::render::light_intensity_unit::candela, 5.0f, 2.0f) ==
            Catch::Approx(5.0f));
    REQUIRE(arc::render::light_intensity_scale(arc::render::light_intensity_unit::lux, 3.0f, 2.0f) ==
            Catch::Approx(3.0f));
    REQUIRE(arc::render::light_intensity_scale(arc::render::light_intensity_unit::lumen, 4.0f * arc::math::pi<float>) ==
            Catch::Approx(1.0f));
    const auto warm = arc::render::color_temperature_rgb(3000.0f);
    const auto cool = arc::render::color_temperature_rgb(9000.0f);
    REQUIRE(warm[0] >= warm[2]);
    REQUIRE(cool[2] >= cool[0]);
}

TEST_CASE("PBR color transfer and material texture semantics are explicit")
{
    const arc::math::vector3f srgb{0.0f, 0.5f, 1.0f};
    const auto linear = arc::render::srgb_to_linear(srgb);
    const auto round_trip = arc::render::linear_to_srgb(linear);
    REQUIRE(round_trip[0] == Catch::Approx(srgb[0]).margin(1.0e-6f));
    REQUIRE(round_trip[1] == Catch::Approx(srgb[1]).margin(1.0e-5f));
    REQUIRE(round_trip[2] == Catch::Approx(srgb[2]).margin(1.0e-6f));
    REQUIRE(arc::render::texture_semantic_accepts(arc::render::texture_semantic::base_color,
                                                  arc::render::texture_color_space::srgb));
    REQUIRE_FALSE(arc::render::texture_semantic_accepts(arc::render::texture_semantic::normal,
                                                        arc::render::texture_color_space::srgb));
}

TEST_CASE("PBR reference functions stay finite and preserve physical limits")
{
    for (const float roughness : {0.04f, 0.25f, 0.6f, 1.0f})
    {
        const float distribution = arc::render::ggx_distribution(0.75f, roughness);
        const float visibility = arc::render::smith_ggx_correlated(0.6f, 0.7f, roughness);
        REQUIRE(std::isfinite(distribution));
        REQUIRE(std::isfinite(visibility));
        REQUIRE(distribution >= 0.0f);
        REQUIRE(visibility >= 0.0f);
    }
    const auto fresnel = arc::render::fresnel_schlick(0.0f, {0.04f, 0.04f, 0.04f});
    REQUIRE(fresnel[0] == Catch::Approx(1.0f));
    const auto absorption = arc::render::beer_lambert_attenuation({0.5f, 0.25f, 1.0f}, 2.0f, 2.0f);
    REQUIRE(absorption[0] == Catch::Approx(0.5f));
    REQUIRE(absorption[1] == Catch::Approx(0.25f));
    REQUIRE(absorption[2] == Catch::Approx(1.0f));
}

TEST_CASE("physical attenuation exposure and area light packing are stable")
{
    REQUIRE(arc::render::inverse_square_attenuation(2.0f, 0.0f) == Catch::Approx(0.25f));
    REQUIRE(arc::render::inverse_square_attenuation(10.0f, 5.0f) == Catch::Approx(0.0f));
    REQUIRE(arc::render::cone_solid_angle(arc::math::pi<float> * 0.5f) == Catch::Approx(2.0f * arc::math::pi<float>));

    arc::render::exposure_settings settings;
    settings.mode = arc::render::exposure_mode::automatic;
    settings.brighten_speed = 4.0f;
    settings.darken_speed = 2.0f;
    auto state = arc::render::adapt_exposure({}, settings, 5.0f, 1.0f / 60.0f, true);
    REQUIRE(state.valid);
    REQUIRE(state.ev100 == Catch::Approx(5.0f));
    const auto adapted = arc::render::adapt_exposure(state, settings, 8.0f, 1.0f, false);
    REQUIRE(adapted.ev100 > state.ev100);
    REQUIRE(adapted.ev100 < 8.0f);

    std::vector<arc::render::area_light_event> areas{{.intensity = 1000.0f,
                                                      .width = 2.0f,
                                                      .height = 1.0f,
                                                      .shape = arc::render::area_light_shape::rectangle,
                                                      .intensity_unit = arc::render::light_intensity_unit::lumen}};
    const auto lighting = arc::render::pack_scene_lighting({}, {}, {}, nullptr, 0, 0, areas);
    REQUIRE(lighting.area_count == 1);
    REQUIRE(lighting.area_lights[0].color_intensity[3] == Catch::Approx(1000.0f / (2.0f * arc::math::pi<float>)));
}
