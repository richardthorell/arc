#include <arc/scene/scene.h>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

TEST_CASE("registry creates destroys and rejects stale entities")
{
    arc::scene::registry scene;
    const auto first = scene.create();

    REQUIRE(scene.alive(first));
    REQUIRE(scene.live_count() == 1);
    REQUIRE(scene.destroy(first));
    REQUIRE_FALSE(scene.alive(first));
    REQUIRE(scene.live_count() == 0);

    const auto second = scene.create();
    REQUIRE(second.index == first.index);
    REQUIRE(second.generation != first.generation);
    REQUIRE(scene.alive(second));
    REQUIRE_FALSE(scene.destroy(first));
}

TEST_CASE("registry stores removes and clears components on destroy")
{
    arc::scene::registry scene;
    const auto entity = scene.create();

    auto& name = scene.emplace<arc::scene::name_component>(entity, "Camera");
    REQUIRE(name.value == "Camera");
    REQUIRE(scene.has<arc::scene::name_component>(entity));
    REQUIRE(scene.try_get<arc::scene::name_component>(entity) != nullptr);

    scene.remove<arc::scene::name_component>(entity);
    REQUIRE_FALSE(scene.has<arc::scene::name_component>(entity));

    scene.emplace<arc::scene::transform_component>(entity);
    REQUIRE(scene.has<arc::scene::transform_component>(entity));
    REQUIRE(scene.destroy(entity));
    REQUIRE_FALSE(scene.has<arc::scene::transform_component>(entity));
}

TEST_CASE("registry view returns entities with all requested components")
{
    arc::scene::registry scene;
    const auto camera = scene.create();
    const auto mesh = scene.create();

    scene.emplace<arc::scene::transform_component>(camera);
    scene.emplace<arc::scene::camera_component>(camera);
    scene.emplace<arc::scene::transform_component>(mesh);
    scene.emplace<arc::scene::mesh_renderer_component>(mesh);

    std::size_t count{};
    scene.view<arc::scene::transform_component, arc::scene::camera_component>().each(
        [&](arc::scene::entity value, const auto&, const auto&) {
            REQUIRE(value == camera);
            ++count;
        });
    REQUIRE(count == 1);
}

TEST_CASE("transform and camera helpers use right handed minus z forward")
{
    arc::scene::transform_component transform;
    transform.position = arc::math::vector3f{ 1.0f, 2.0f, 3.0f };
    transform.scale = arc::math::vector3f{ 2.0f, 3.0f, 4.0f };
    transform.dirty = false;
    transform.set_position({ 1.0f, 2.0f, 3.0f });
    REQUIRE(transform.dirty);

    const auto local = arc::scene::local_matrix(transform);
    REQUIRE(local(0, 3) == Catch::Approx(1.0f));
    REQUIRE(local(1, 3) == Catch::Approx(2.0f));
    REQUIRE(local(2, 3) == Catch::Approx(3.0f));
    REQUIRE(local(0, 0) == Catch::Approx(2.0f));
    REQUIRE(local(1, 1) == Catch::Approx(3.0f));
    REQUIRE(local(2, 2) == Catch::Approx(4.0f));

    arc::scene::camera_component camera;
    const auto projection = arc::scene::perspective_rh_zo(camera.fov_y_radians, 16.0f / 9.0f, 0.1f, 100.0f);
    REQUIRE(projection(3, 2) == Catch::Approx(-1.0f));

    arc::scene::transform_component camera_transform;
    camera_transform.position = arc::math::vector3f{ 0.0f, 0.0f, 5.0f };
    const auto view = arc::scene::view_matrix(camera_transform);
    const auto camera_space = arc::math::transform_point(view, arc::math::vector3f{ 0.0f, 0.0f, 0.0f });
    REQUIRE(camera_space[2] == Catch::Approx(-5.0f));

    const auto forward = arc::scene::forward_direction(camera_transform);
    REQUIRE(forward[0] == Catch::Approx(0.0f));
    REQUIRE(forward[1] == Catch::Approx(0.0f));
    REQUIRE(forward[2] == Catch::Approx(-1.0f));

    const auto up = arc::scene::up_direction(camera_transform);
    REQUIRE(up[0] == Catch::Approx(0.0f));
    REQUIRE(up[1] == Catch::Approx(1.0f));
    REQUIRE(up[2] == Catch::Approx(0.0f));
}

TEST_CASE("render scene extracts visible mesh draw events from active camera")
{
    arc::scene::registry scene;
    arc::render::renderer renderer;
    const arc::render::mesh_handle mesh{ .index = 2, .generation = 1 };

    const auto camera_entity = scene.create();
    arc::scene::transform_component camera_transform;
    camera_transform.position = arc::math::vector3f{ 0.0f, 0.0f, 5.0f };
    scene.emplace<arc::scene::transform_component>(camera_entity, camera_transform);
    scene.emplace<arc::scene::camera_component>(camera_entity);

    const auto mesh_entity = scene.create();
    scene.emplace<arc::scene::transform_component>(mesh_entity);
    scene.emplace<arc::scene::selection_component>(mesh_entity, true);
    scene.emplace<arc::scene::mesh_renderer_component>(
        mesh_entity,
        mesh,
        arc::render::material_handle{},
        true,
        arc::math::vector4f{ 0.2f, 0.4f, 0.6f, 1.0f });

    const auto result = arc::scene::render_scene(
        scene,
        renderer,
        1280,
        720,
        arc::render::render_mode::wireframe,
        arc::render::mesh_visualization_mode::standard,
        arc::render::editor_overlay_mode::selected_wireframe);
    REQUIRE(result.camera_found);
    REQUIRE(result.renderable_count == 1);
    REQUIRE(result.submitted_draw_count == 1);
    REQUIRE(result.selected_count == 1);

    const auto packet = renderer.frame_queue().commit(1);
    REQUIRE(packet.events.size() == 1);
    REQUIRE(packet.events[0].type() == arc::render::render_event_type::render_world);
    const auto& world_event = std::get<arc::render::render_world_event>(packet.events[0].payload);
    REQUIRE(world_event.packet);
    REQUIRE(world_event.packet->visible_items.size() == 1);
    const auto& item = world_event.packet->items[world_event.packet->visible_items[0]];
    REQUIRE(item.mesh == mesh);
    REQUIRE(world_event.packet->mode == arc::render::render_mode::wireframe);
    REQUIRE(item.selected);
    REQUIRE(item.base_color_tint[2] == Catch::Approx(0.6f));
    REQUIRE(world_event.packet->shadows_enabled);
}

TEST_CASE("render scene culling uses transformed dirty local bounds")
{
    arc::scene::registry scene;
    arc::render::renderer renderer;
    const arc::render::mesh_handle mesh{ .index = 2, .generation = 1 };

    const auto camera_entity = scene.create();
    arc::scene::transform_component camera_transform;
    camera_transform.position = arc::math::vector3f{ 0.0f, 0.0f, 5.0f };
    scene.emplace<arc::scene::transform_component>(camera_entity, camera_transform);
    scene.emplace<arc::scene::camera_component>(camera_entity);

    const auto mesh_entity = scene.create();
    arc::scene::transform_component transform;
    transform.position = arc::math::vector3f{ 20.0f, 0.0f, 0.0f };
    scene.emplace<arc::scene::transform_component>(mesh_entity, transform);
    scene.emplace<arc::scene::bounds_component>(
        mesh_entity,
        arc::geometric::box3f{
            arc::geometric::point3f{ arc::math::vector3f{ -22.0f, -1.0f, -6.0f } },
            arc::geometric::point3f{ arc::math::vector3f{ -18.0f, 1.0f, -4.0f } } },
        arc::geometric::box3f{},
        true);
    scene.emplace<arc::scene::mesh_renderer_component>(mesh_entity, mesh, arc::render::material_handle{}, true);

    const auto result = arc::scene::render_scene(scene, renderer, 1280, 720);
    REQUIRE(result.renderable_count == 1);
    REQUIRE(result.submitted_draw_count == 1);
    REQUIRE(result.culled_count == 0);

    const auto packet = renderer.frame_queue().commit(1);
    REQUIRE(packet.events.size() == 1);
    const auto& world_event = std::get<arc::render::render_world_event>(packet.events[0].payload);
    REQUIRE(world_event.packet);
    REQUIRE(world_event.packet->visible_items.size() == 1);
}

TEST_CASE("render scene can request wireframe overlay for every draw")
{
    arc::scene::registry scene;
    arc::render::renderer renderer;
    const arc::render::mesh_handle mesh{ .index = 2, .generation = 1 };

    const auto camera_entity = scene.create();
    arc::scene::transform_component camera_transform;
    camera_transform.position = arc::math::vector3f{ 0.0f, 0.0f, 5.0f };
    scene.emplace<arc::scene::transform_component>(camera_entity, camera_transform);
    scene.emplace<arc::scene::camera_component>(camera_entity);

    const auto mesh_entity = scene.create();
    scene.emplace<arc::scene::transform_component>(mesh_entity);
    scene.emplace<arc::scene::mesh_renderer_component>(mesh_entity, mesh, arc::render::material_handle{}, true);

    const auto result = arc::scene::render_scene(
        scene,
        renderer,
        1280,
        720,
        arc::render::render_mode::shaded,
        arc::render::mesh_visualization_mode::albedo,
        arc::render::editor_overlay_mode::all_wireframe);
    REQUIRE(result.submitted_draw_count == 1);

    const auto packet = renderer.frame_queue().commit(1);
    REQUIRE(packet.events.size() == 1);
    const auto& world_event = std::get<arc::render::render_world_event>(packet.events[0].payload);
    REQUIRE(world_event.packet);
    REQUIRE(world_event.packet->mode == arc::render::render_mode::shaded);
    REQUIRE(world_event.packet->visualization == arc::render::mesh_visualization_mode::albedo);
    REQUIRE(world_event.packet->overlay == arc::render::editor_overlay_mode::all_wireframe);
}

TEST_CASE("render scene skips extraction without active camera")
{
    arc::scene::registry scene;
    arc::render::renderer renderer;

    const auto mesh_entity = scene.create();
    scene.emplace<arc::scene::transform_component>(mesh_entity);
    scene.emplace<arc::scene::mesh_renderer_component>(mesh_entity);

    const auto result = arc::scene::render_scene(scene, renderer, 1280, 720);
    REQUIRE_FALSE(result.camera_found);
    REQUIRE(result.submitted_draw_count == 0);
    REQUIRE(renderer.frame_queue().commit(1).events.empty());
}

TEST_CASE("render scene extracts active lights and skips inactive renderers")
{
    arc::scene::registry scene;
    arc::render::renderer renderer;
    const arc::render::mesh_handle mesh{ .index = 3, .generation = 1 };

    const auto camera_entity = scene.create();
    arc::scene::transform_component camera_transform;
    camera_transform.position = arc::math::vector3f{ 0.0f, 0.0f, 5.0f };
    scene.emplace<arc::scene::active_component>(camera_entity);
    scene.emplace<arc::scene::transform_component>(camera_entity, camera_transform);
    scene.emplace<arc::scene::camera_component>(camera_entity);

    const auto mesh_entity = scene.create();
    scene.emplace<arc::scene::active_component>(mesh_entity, false);
    scene.emplace<arc::scene::transform_component>(mesh_entity);
    scene.emplace<arc::scene::mesh_renderer_component>(mesh_entity, mesh, arc::render::material_handle{}, true);

    const auto sun = scene.create();
    scene.emplace<arc::scene::name_component>(sun, "Sun");
    scene.emplace<arc::scene::active_component>(sun);
    scene.emplace<arc::scene::transform_component>(sun);
    scene.emplace<arc::scene::directional_light_component>(
        sun,
        arc::math::vector3f{ 1.0f, 0.9f, 0.7f },
        2.0f,
        true);
    auto& sun_light = scene.get<arc::scene::directional_light_component>(sun);
    sun_light.use_color_temperature = true;
    sun_light.temperature_kelvin = 3000.0f;
    sun_light.intensity_unit = arc::render::light_intensity_unit::lux;
    sun_light.shadow.resolution = 4096;
    sun_light.shadow.filter = arc::render::shadow_filter::pcf_5x5;

    const auto disabled_point = scene.create();
    scene.emplace<arc::scene::active_component>(disabled_point);
    scene.emplace<arc::scene::transform_component>(disabled_point);
    scene.emplace<arc::scene::point_light_component>(
        disabled_point,
        arc::math::vector3f{ 0.2f, 0.3f, 1.0f },
        10.0f,
        5.0f,
        false,
        false);

    const auto reflection_probe = scene.create();
    scene.emplace<arc::scene::active_component>(reflection_probe);
    scene.emplace<arc::scene::transform_component>(reflection_probe);
    scene.emplace<arc::scene::reflection_probe_component>(reflection_probe, 8.0f, 1.5f, true);

    const auto result = arc::scene::render_scene(scene, renderer, 640, 360);
    REQUIRE(result.camera_found);
    REQUIRE(result.renderable_count == 0);
    REQUIRE(result.submitted_draw_count == 0);
    REQUIRE(result.directional_light_count == 1);
    REQUIRE(result.point_light_count == 0);
    REQUIRE(result.reflection_probe_count == 1);

    const auto packet = renderer.frame_queue().commit(2);
    REQUIRE(packet.events.size() == 1);
    REQUIRE(packet.events[0].type() == arc::render::render_event_type::render_world);
    const auto& world_event = std::get<arc::render::render_world_event>(packet.events[0].payload);
    REQUIRE(world_event.packet);
    REQUIRE(world_event.packet->directional_lights.size() == 1);
    const auto& light = world_event.packet->directional_lights[0];
    REQUIRE(light.label == "Sun");
    REQUIRE(light.intensity == Catch::Approx(2.0f));
    REQUIRE(light.casts_shadows);
    REQUIRE(light.use_color_temperature);
    REQUIRE(light.temperature_kelvin == Catch::Approx(3000.0f));
    REQUIRE(light.intensity_unit == arc::render::light_intensity_unit::lux);
    REQUIRE(light.shadow.enabled);
    REQUIRE(light.shadow.resolution == 4096);
    REQUIRE(light.shadow.filter == arc::render::shadow_filter::pcf_5x5);
    REQUIRE(light.color[0] >= light.color[2]);
    REQUIRE(world_event.packet->reflection_probes.size() == 1);
    REQUIRE(world_event.packet->reflection_probes[0].radius == Catch::Approx(8.0f));
}

TEST_CASE("render scene extracts skinned and instanced render items")
{
    arc::scene::registry scene;
    arc::render::renderer renderer;

    const auto camera_entity = scene.create();
    arc::scene::transform_component camera_transform;
    camera_transform.position = arc::math::vector3f{ 0.0f, 0.0f, 5.0f };
    scene.emplace<arc::scene::transform_component>(camera_entity, camera_transform);
    scene.emplace<arc::scene::camera_component>(camera_entity);

    const auto skinned = scene.create();
    scene.emplace<arc::scene::transform_component>(skinned);
    scene.emplace<arc::scene::skinned_mesh_renderer_component>(
        skinned,
        arc::render::mesh_handle{ .index = 10, .generation = 1 },
        arc::render::material_handle{ .index = 2, .generation = 1 },
        arc::render::buffer_handle{ .index = 4, .generation = 1 },
        64u,
        true);

    const auto instanced = scene.create();
    scene.emplace<arc::scene::transform_component>(instanced);
    scene.emplace<arc::scene::instance_group_component>(
        instanced,
        arc::render::mesh_handle{ .index = 11, .generation = 1 },
        arc::render::material_handle{ .index = 3, .generation = 1 },
        12u,
        true);

    const auto result = arc::scene::render_scene(scene, renderer, 1280, 720);
    REQUIRE(result.camera_found);
    REQUIRE(result.renderable_count == 2);
    REQUIRE(result.submitted_draw_count == 2);
    REQUIRE(result.indirect_draw_count >= 1);

    const auto packet = renderer.frame_queue().commit(3);
    REQUIRE(packet.events.size() == 1);
    const auto& world_event = std::get<arc::render::render_world_event>(packet.events[0].payload);
    REQUIRE(world_event.packet);
    REQUIRE(world_event.packet->items.size() == 2);
    REQUIRE(world_event.packet->items[0].skin_joint_count == 64);
    REQUIRE(world_event.packet->items[1].instance_count == 12);
}

TEST_CASE("render scene applies first valid LOD mesh")
{
    arc::scene::registry scene;
    arc::render::renderer renderer;
    const arc::render::mesh_handle base_mesh{ .index = 1, .generation = 1 };
    const arc::render::mesh_handle lod_mesh{ .index = 9, .generation = 1 };

    const auto camera_entity = scene.create();
    arc::scene::transform_component camera_transform;
    camera_transform.position = arc::math::vector3f{ 0.0f, 0.0f, 5.0f };
    scene.emplace<arc::scene::transform_component>(camera_entity, camera_transform);
    scene.emplace<arc::scene::camera_component>(camera_entity);

    const auto mesh_entity = scene.create();
    scene.emplace<arc::scene::transform_component>(mesh_entity);
    scene.emplace<arc::scene::mesh_renderer_component>(mesh_entity, base_mesh, arc::render::material_handle{}, true);
    scene.emplace<arc::scene::lod_component>(
        mesh_entity,
        std::vector<arc::scene::lod_level>{ { .screen_coverage = 1.0f, .mesh = lod_mesh } },
        true);

    const auto result = arc::scene::render_scene(scene, renderer, 1280, 720);
    REQUIRE(result.submitted_draw_count == 1);

    const auto packet = renderer.frame_queue().commit(4);
    const auto& world_event = std::get<arc::render::render_world_event>(packet.events[0].payload);
    REQUIRE(world_event.packet->items[0].mesh == lod_mesh);
}
