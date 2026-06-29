#include <arc/scene.h>

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
    scene.emplace<arc::scene::mesh_renderer_component>(mesh_entity, mesh, arc::render::material_handle{}, true);

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
    REQUIRE(packet.events[0].type() == arc::render::render_event_type::draw);
    const auto& draw = std::get<arc::render::draw_mesh_event>(packet.events[0].payload);
    REQUIRE(draw.mesh == mesh);
    REQUIRE(draw.mode == arc::render::render_mode::wireframe);
    REQUIRE(draw.selected);
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
    const auto& draw = std::get<arc::render::draw_mesh_event>(packet.events[0].payload);
    REQUIRE(draw.mode == arc::render::render_mode::shaded);
    REQUIRE(draw.visualization == arc::render::mesh_visualization_mode::albedo);
    REQUIRE(draw.selected);
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

    const auto result = arc::scene::render_scene(scene, renderer, 640, 360);
    REQUIRE(result.camera_found);
    REQUIRE(result.renderable_count == 0);
    REQUIRE(result.submitted_draw_count == 0);
    REQUIRE(result.directional_light_count == 1);

    const auto packet = renderer.frame_queue().commit(2);
    REQUIRE(packet.events.size() == 1);
    REQUIRE(packet.events[0].type() == arc::render::render_event_type::directional_light);
    const auto& light = std::get<arc::render::directional_light_event>(packet.events[0].payload);
    REQUIRE(light.label == "Sun");
    REQUIRE(light.intensity == Catch::Approx(2.0f));
    REQUIRE(light.casts_shadows);
}
