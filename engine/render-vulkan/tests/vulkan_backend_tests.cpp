#include <arc/render/vulkan/vulkan_backend.h>
#include <arc/render/primitives.h>
#include <arc/render/renderer.h>
#include <arc/math/constants.h>

#include "vulkan_sky_constants.h"
#include "vulkan_pick_utils.h"

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <bit>
#include <limits>

namespace
{

void require_terrain_frame_submission(arc::render::renderer& renderer)
{
    renderer.resize_viewport(64u, 64u);
    const auto mesh = renderer.create_mesh(arc::render::make_plane_mesh(16.0f));
    arc::render::material_desc terrain;
    terrain.name = "Vulkan terrain sampler smoke";
    terrain.domain = arc::render::material_domain::terrain;
    const auto material = renderer.create_material(std::move(terrain));

    arc::render::render_event_buffer events;
    arc::render::render_event_writer writer(events);
    const auto identity = arc::math::identity<float, 4>();
    writer.draw_mesh(mesh, material, identity, identity, "terrain sampler smoke");
    renderer.frame_queue().submit(std::move(events));

    const auto submitted = renderer.render_frame(
        1u, arc::render::make_scene_draw_graph("vulkan terrain smoke", arc::render::render_path::deferred));
    REQUIRE(submitted.submitted);
}

} // namespace

TEST_CASE("Vulkan picking maps output pixels to dynamic-resolution ObjectID pixels")
{
    using arc::render::vulkan::detail::map_output_pixel_to_render_pixel;

    STATIC_REQUIRE(map_output_pixel_to_render_pixel(0, 1920, 1286) == 0);
    STATIC_REQUIRE(map_output_pixel_to_render_pixel(960, 1920, 1286) == 643);
    STATIC_REQUIRE(map_output_pixel_to_render_pixel(1919, 1920, 1286) == 1285);
    STATIC_REQUIRE(map_output_pixel_to_render_pixel(799, 800, 800) == 799);
    STATIC_REQUIRE(map_output_pixel_to_render_pixel(10, 0, 800) == 0);
    STATIC_REQUIRE(map_output_pixel_to_render_pixel(10, 800, 0) == 0);
}

TEST_CASE("Vulkan sky settings use the portable 128-byte push constant layout")
{
    using namespace arc::render;
    using namespace arc::render::vulkan::detail;

    STATIC_REQUIRE(sizeof(sky_push_constants) == vulkan_minimum_push_constant_bytes);

    const auto sun = pack_sun_settings_bits(32.0f, 5.0f);
    REQUIRE((sun & 0xffffu) == 0xffffu);
    REQUIRE(((sun >> 16u) & 0xffffu) == 0xffffu);

    const auto moon = pack_moon_settings_bits(1.0f, 8.0f, 5.0f, true);
    REQUIRE((moon & 0x3ffu) == 0x3ffu);
    REQUIRE(((moon >> 10u) & 0x3ffu) == 0x3ffu);
    REQUIRE(((moon >> 20u) & 0x3ffu) == 0x3ffu);
    REQUIRE((moon & (1u << 30u)) != 0u);

    const auto sky = pack_sky_settings_bits(sky_source_mode::solid_color, 1.0f, 16.0f);
    REQUIRE((sky & 0x3u) == static_cast<std::uint32_t>(sky_source_mode::solid_color));
    REQUIRE(((sky >> 2u) & 0x3ffu) == 0x3ffu);
    REQUIRE(((sky >> 12u) & 0xfffu) == 0xfffu);
}

TEST_CASE("Vulkan sky settings clamp invalid values and build camera constants")
{
    using namespace arc::render;
    using namespace arc::render::vulkan::detail;

    REQUIRE(pack_sun_settings_bits(-1.0f, std::numeric_limits<float>::infinity()) == 0u);
    REQUIRE((pack_moon_settings_bits(-1.0f, -1.0f, -1.0f, false) & 0x7fffffffu) == 0u);

    world_environment_data environment;
    environment.source = sky_source_mode::hdri;
    environment.hdri_rotation_degrees = 180.0f;
    environment.clouds.enabled = true;
    environment.clouds.cumulus.enabled = true;
    environment.clouds.cumulus.coverage = 2.0f;
    environment.clouds.cumulus.density = -1.0f;
    render_camera camera;
    camera.projection = arc::math::identity<float, 4>();
    const auto constants = build_sky_push_constants(environment, camera, 1920, 1080, false);
    REQUIRE(constants.camera_up_aspect[3] == Catch::Approx(1920.0f / 1080.0f));
    REQUIRE(constants.atmosphere[0] == Catch::Approx(arc::math::pi<float>));
    REQUIRE(constants.cumulus[0] == 1.0f);
    REQUIRE(constants.cumulus[1] == 0.0f);
    REQUIRE(constants.cirrus[0] == 0.0f);
}

TEST_CASE("Vulkan loader availability can be queried")
{
    (void)arc::render::vulkan::vulkan_loader_available();
    SUCCEED("Vulkan loader query completed");
}

TEST_CASE("Vulkan backend creation either succeeds or reports a reason")
{
    auto result = arc::render::vulkan::create_vulkan_backend();
    if (result.succeeded())
    {
        arc::render::renderer renderer;
        REQUIRE(result.backend->type() == arc::render::render_backend_type::vulkan);
        const auto capabilities = result.backend->capabilities();
        REQUIRE(capabilities.api_major >= 1);
        REQUIRE((capabilities.api_major > 1 || capabilities.api_minor >= 2));
        REQUIRE_FALSE(capabilities.adapter_name.empty());
        REQUIRE(capabilities.graphics_queue);
        REQUIRE(capabilities.compute_queue);
        REQUIRE(capabilities.dynamic_rendering);
        REQUIRE(capabilities.max_color_attachments >= 5);
        renderer.set_backend(std::move(result.backend));
        require_terrain_frame_submission(renderer);
    }
    else
    {
        REQUIRE_FALSE(result.message.empty());
    }
}

TEST_CASE("Vulkan compatibility override leaves optional device paths disabled")
{
    arc::render::vulkan::vulkan_backend_config config{};
    config.force_disable_optional_features = true;
    auto result = arc::render::vulkan::create_vulkan_backend(config);
    if (result.succeeded())
    {
        arc::render::renderer renderer({ .force_disable_optional_features = true });
        renderer.set_backend(std::move(result.backend));
        REQUIRE_FALSE(renderer.resolved_config().features.synchronization2);
        REQUIRE_FALSE(renderer.resolved_config().features.timeline_semaphores);
        REQUIRE_FALSE(renderer.resolved_config().features.descriptor_indexing);
        require_terrain_frame_submission(renderer);
    }
    else
    {
        REQUIRE_FALSE(result.message.empty());
    }
}
