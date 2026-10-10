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
                               .label = "sun",
                               .source_angle = 0.00465f});
    }

    std::vector<arc::render::point_light_event> points{
        {.object_id = {.index = 17, .generation = 3},
         .position = {1.0f, 2.0f, 3.0f},
         .color = {1.0f, 0.5f, 0.25f},
         .intensity = 80.0f,
         .range = 4.0f,
         .intensity_unit = arc::render::light_intensity_unit::lumen,
         .source_radius = 0.2f,
         .source_length = 0.5f},
        {.position = {0.0f, 0.0f, 0.0f}, .color = {1.0f, 1.0f, 1.0f}, .intensity = 2.0f, .range = 8.0f}};
    std::vector<arc::render::spot_light_event> spots{{.position = {0.0f, 1.0f, 0.0f},
                                                      .direction = {0.0f, -1.0f, 0.0f},
                                                      .color = {0.8f, 0.9f, 1.0f},
                                                      .intensity = 3.0f,
                                                      .range = 10.0f,
                                                      .inner_angle = 0.2f,
                                                      .outer_angle = 0.7f,
                                                      .source_radius = 0.15f,
                                                      .source_length = 0.4f}};

    arc::render::environment_descriptor environment;
    environment.fallback_color = {0.1f, 0.2f, 0.3f};
    environment.intensity = 1.25f;

    const auto data = arc::render::pack_scene_lighting(directional, points, spots, &environment);
    REQUIRE(data.directional_count == arc::render::max_directional_lights);
    REQUIRE(data.skipped_directional_count == 2);
    REQUIRE(data.directional_lights[0].direction_intensity[3] == Catch::Approx(6.0f));
    REQUIRE(data.directional_lights[0].source_shape[0] == Catch::Approx(0.00465f));
    REQUIRE(data.point_count == 2);
    REQUIRE(data.point_lights[0].color_intensity[3] == Catch::Approx(80.0f / (4.0f * arc::math::pi<float>)));
    REQUIRE(data.point_lights[0].object_id_shadow[0] == Catch::Approx(17.0f));
    REQUIRE(data.point_lights[0].object_id_shadow[1] == Catch::Approx(3.0f));
    REQUIRE(data.point_lights[0].shadow_parameters[0] == Catch::Approx(-1.0f));
    REQUIRE(data.point_lights[0].source_shape[0] == Catch::Approx(0.2f));
    REQUIRE(data.point_lights[0].source_shape[1] == Catch::Approx(0.5f));
    REQUIRE(data.spot_count == 1);
    REQUIRE(data.spot_lights[0].params[0] == Catch::Approx(0.7f));
    REQUIRE(data.spot_lights[0].source_shape[0] == Catch::Approx(0.15f));
    REQUIRE(data.spot_lights[0].source_shape[1] == Catch::Approx(0.4f));
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

TEST_CASE("clustered local-light grid is screen/depth aware and bounded")
{
    std::vector<arc::render::point_light_event> points{
        {.position = {0.0f, 0.0f, -5.0f}, .intensity = 10.0f, .range = 1.0f},
        {.position = {50.0f, 0.0f, -5.0f}, .intensity = 10.0f, .range = 1.0f}};
    const auto lighting = arc::render::pack_scene_lighting({}, points, {});

    arc::render::clustered_light_grid_view view{};
    view.viewport_width = 128u;
    view.viewport_height = 64u;
    view.near_plane = 0.1f;
    view.far_plane = 100.0f;

    arc::render::clustered_light_grid_config config{};
    config.tile_size_pixels = 32u;
    config.depth_slices = 4u;
    config.maximum_lights_per_cluster = 1u;
    const auto grid = arc::render::build_clustered_light_grid(lighting, view, config);

    REQUIRE(grid.tiles_x == 4u);
    REQUIRE(grid.tiles_y == 2u);
    REQUIRE(grid.cluster_count == 32u);
    REQUIRE(grid.point_light_references > 0u);
    REQUIRE(grid.point_light_references < grid.cluster_count * lighting.point_count);
    REQUIRE(grid.gpu_words.size() ==
            arc::render::clustered_light_header_words + grid.cluster_count * (1u + config.maximum_lights_per_cluster));

    const std::uint32_t record_words = 1u + config.maximum_lights_per_cluster;
    for (std::uint32_t cluster = 0u; cluster < grid.cluster_count; ++cluster)
    {
        const auto base = arc::render::clustered_light_header_words + cluster * record_words;
        REQUIRE(grid.gpu_words[base] <= config.maximum_lights_per_cluster);
    }
}

TEST_CASE("IES photometric profiles parse, normalize, and sample deterministically")
{
    constexpr std::string_view ies = R"(IESNA:LM-63-2002
[TEST] ARC deterministic fixture
TILT=NONE
1 1000 2 3 2 1 1 0.1 0.1 0.1 1 0 10
0 45 90
0 180
10 20 40
5 10 20
)";

    arc::render::photometric_profile profile;
    std::string error;
    REQUIRE(arc::render::parse_ies_profile(ies, profile, error));
    REQUIRE(error.empty());
    REQUIRE(profile.photometric_type == 1u);
    REQUIRE(profile.vertical_angles_degrees.size() == 3u);
    REQUIRE(profile.horizontal_angles_degrees.size() == 2u);
    REQUIRE(profile.normalized_candela.size() == 6u);
    REQUIRE(profile.peak_candela == Catch::Approx(80.0f));
    REQUIRE(profile.declared_lumens == Catch::Approx(1000.0f));
    REQUIRE(profile.normalized_candela[2] == Catch::Approx(1.0f));
    REQUIRE(profile.normalized_candela[5] == Catch::Approx(0.5f));

    const float axis =
        arc::render::sample_photometric_profile(profile, 0.0f, 0.0f);
    const float vertical_mid =
        arc::render::sample_photometric_profile(profile, arc::math::pi<float> * 0.25f, 0.0f);
    const float horizontal_mid =
        arc::render::sample_photometric_profile(profile, arc::math::pi<float> * 0.5f,
                                                arc::math::pi<float> * 0.5f);
    REQUIRE(axis == Catch::Approx(0.25f));
    REQUIRE(vertical_mid == Catch::Approx(0.5f));
    REQUIRE(horizontal_mid == Catch::Approx(0.75f));

    // 0..180 photometry mirrors deterministically into the rear hemisphere.
    REQUIRE(arc::render::sample_photometric_profile(profile, arc::math::pi<float> * 0.5f,
                                                    arc::math::pi<float> * 1.5f) ==
            Catch::Approx(horizontal_mid));

    arc::render::photometric_profile empty;
    REQUIRE(arc::render::sample_photometric_profile(empty, 0.5f, 0.5f) == Catch::Approx(1.0f));
}

TEST_CASE("IES parser rejects unsupported tilt and malformed distributions")
{
    arc::render::photometric_profile profile;
    std::string error;
    REQUIRE_FALSE(arc::render::parse_ies_profile("IESNA:LM-63-2002\nTILT=INCLUDE\n", profile, error));
    REQUIRE(error.find("TILT") != std::string::npos);

    REQUIRE_FALSE(arc::render::parse_ies_profile(
        "IESNA:LM-63-2002\nTILT=NONE\n1 1000 1 2 1 1 1 1 1 1 1 0 10\n0 90\n0\n0 0\n",
        profile, error));
    REQUIRE(error.find("positive intensity") != std::string::npos);
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
