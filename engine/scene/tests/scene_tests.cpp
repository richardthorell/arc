#include <arc/scene/scene.h>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <cmath>
#include <functional>
#include <limits>
#include <numeric>
#include <vector>

TEST_CASE("registry creates destroys and rejects stale entities")
{
    arc::ecs::world scene;
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
    arc::ecs::world scene;
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
    arc::ecs::world scene;
    const auto camera = scene.create();
    const auto mesh = scene.create();

    scene.emplace<arc::scene::transform_component>(camera);
    scene.emplace<arc::scene::camera_component>(camera);
    scene.emplace<arc::scene::transform_component>(mesh);
    scene.emplace<arc::scene::mesh_renderer_component>(mesh);

    std::size_t count{};
    scene.view<arc::scene::transform_component, arc::scene::camera_component>().each(
        [&](arc::ecs::entity value, const auto&, const auto&)
        {
            REQUIRE(value == camera);
            ++count;
        });
    REQUIRE(count == 1);
}

TEST_CASE("registry copies component pools without sharing storage")
{
    arc::ecs::world original;
    const auto value = original.create();
    original.emplace<arc::scene::name_component>(value, "Original");
    arc::ecs::world copy = original;
    copy.get<arc::scene::name_component>(value).value = "Copy";
    REQUIRE(original.get<arc::scene::name_component>(value).value == "Original");
    REQUIRE(copy.get<arc::scene::name_component>(value).value == "Copy");
}

TEST_CASE("paged component pools keep live component addresses stable while growing")
{
    arc::memory::memory_system memory(
        {.physical_memory_override = 16 * 1024 * 1024, .track_live_allocations = true, .capture_call_stacks = false});
    arc::ecs::world scene(memory, 99);
    const auto first = scene.create();
    auto* stable = &scene.emplace<arc::scene::name_component>(first, "Stable");

    for (std::size_t index = 0; index < 2048; ++index)
    {
        const auto entity = scene.create();
        scene.emplace<arc::scene::name_component>(entity, "Additional");
    }

    REQUIRE(&scene.get<arc::scene::name_component>(first) == stable);
    REQUIRE(stable->value == "Stable");
    REQUIRE(scene.memory().world_id() == 99);
    const auto snapshot = memory.snapshot();
    const auto component_domain = std::find_if(snapshot.domains.begin(), snapshot.domains.end(), [](const auto& value)
                                               { return value.domain == arc::memory::memory_domain::components; });
    REQUIRE(component_domain != snapshot.domains.end());
    REQUIRE(component_domain->stats.bytes_outstanding > 0);
}

TEST_CASE("paged component removal only invalidates the removed component")
{
    arc::ecs::world scene;
    const auto first = scene.create();
    const auto second = scene.create();
    auto* first_component = &scene.emplace<arc::scene::name_component>(first, "First");
    auto* second_component = &scene.emplace<arc::scene::name_component>(second, "Second");

    scene.remove<arc::scene::name_component>(first);
    REQUIRE_FALSE(scene.has<arc::scene::name_component>(first));
    REQUIRE(&scene.get<arc::scene::name_component>(second) == second_component);
    REQUIRE(second_component->value == "Second");

    auto& replacement = scene.emplace<arc::scene::name_component>(first, "Replacement");
    REQUIRE(&replacement == first_component);
}

TEST_CASE("entity GUIDs round trip and reject malformed values")
{
    const auto guid = arc::ecs::generate_entity_guid();
    REQUIRE(guid.valid());
    const auto text = arc::ecs::to_string(guid);
    REQUIRE(text.size() == 36);
    REQUIRE(arc::ecs::parse_entity_guid(text) == guid);
    REQUIRE_FALSE(arc::ecs::parse_entity_guid("not-a-guid").has_value());
    REQUIRE_FALSE(arc::ecs::parse_entity_guid("01234567-89ab-cdef-0123-456789abcdeg").has_value());
    REQUIRE_FALSE(arc::ecs::parse_entity_guid("00000000-0000-0000-0000-000000000000").has_value());
}

TEST_CASE("scene hierarchy reparents orders and preserves world transforms")
{
    arc::ecs::world scene;
    const auto parent = scene.create();
    const auto first = scene.create();
    const auto second = scene.create();
    scene.emplace<arc::scene::transform_component>(parent).position = {10.0f, 0.0f, 0.0f};
    scene.emplace<arc::scene::transform_component>(first).position = {2.0f, 0.0f, 0.0f};
    scene.emplace<arc::scene::transform_component>(second).position = {4.0f, 0.0f, 0.0f};

    REQUIRE(arc::scene::reparent(scene, first, parent));
    REQUIRE(arc::scene::reparent(scene, second, parent, first));
    const auto ordered = arc::scene::children(scene, parent);
    REQUIRE(ordered == std::vector<arc::ecs::entity>{second, first});
    REQUIRE(scene.get<arc::scene::hierarchy_component>(parent).child_count == 2);
    arc::scene::update_world_transforms(scene);
    REQUIRE(arc::scene::world_position(scene.get<arc::scene::transform_component>(first))[0] == Catch::Approx(2.0f));
    REQUIRE(scene.get<arc::scene::transform_component>(first).position[0] == Catch::Approx(-8.0f));

    REQUIRE_FALSE(arc::scene::reparent(scene, parent, first));
    REQUIRE(arc::scene::reorder(scene, first, second));
    REQUIRE(arc::scene::children(scene, parent).front() == first);

    const auto root_a = scene.create();
    const auto root_b = scene.create();
    scene.emplace<arc::scene::hierarchy_component>(root_a);
    scene.emplace<arc::scene::hierarchy_component>(root_b);
    REQUIRE(arc::scene::reorder(scene, root_b, root_a));
    const auto roots = arc::scene::roots(scene);
    const auto root_a_position = std::find(roots.begin(), roots.end(), root_a);
    const auto root_b_position = std::find(roots.begin(), roots.end(), root_b);
    REQUIRE(root_b_position < root_a_position);
    REQUIRE(scene.get<arc::scene::hierarchy_component>(root_b).next_sibling == root_a);
}

TEST_CASE("scene hierarchy propagates transforms and destroys complete subtrees")
{
    arc::ecs::world scene;
    const auto logical_root = scene.create();
    const auto root = scene.create();
    const auto child = scene.create();
    const auto grandchild = scene.create();
    scene.emplace<arc::scene::transform_component>(root).position = {3.0f, 0.0f, 0.0f};
    scene.emplace<arc::scene::transform_component>(child).position = {2.0f, 0.0f, 0.0f};
    scene.emplace<arc::scene::transform_component>(grandchild).position = {1.0f, 0.0f, 0.0f};
    REQUIRE(arc::scene::reparent(scene, root, logical_root, {}, arc::scene::reparent_transform_policy::preserve_local));
    REQUIRE(arc::scene::reparent(scene, child, root, {}, arc::scene::reparent_transform_policy::preserve_local));
    REQUIRE(arc::scene::reparent(scene, grandchild, child, {}, arc::scene::reparent_transform_policy::preserve_local));
    arc::scene::update_world_transforms(scene);
    REQUIRE(arc::scene::world_position(scene.get<arc::scene::transform_component>(grandchild))[0] ==
            Catch::Approx(6.0f));
    REQUIRE(arc::scene::destroy_subtree(scene, child));
    REQUIRE(scene.alive(root));
    REQUIRE_FALSE(scene.alive(child));
    REQUIRE_FALSE(scene.alive(grandchild));
    REQUIRE(scene.get<arc::scene::hierarchy_component>(root).child_count == 0);
}

TEST_CASE("transform and camera helpers use right handed minus z forward")
{
    arc::scene::transform_component transform;
    transform.position = arc::math::vector3f{1.0f, 2.0f, 3.0f};
    transform.scale = arc::math::vector3f{2.0f, 3.0f, 4.0f};
    transform.dirty = false;
    transform.set_position({1.0f, 2.0f, 3.0f});
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
    camera_transform.position = arc::math::vector3f{0.0f, 0.0f, 5.0f};
    const auto view = arc::scene::view_matrix(camera_transform);
    const auto camera_space = arc::math::transform_point(view, arc::math::vector3f::zero);
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

TEST_CASE("affine inverse matches the generic inverse for rotated transforms")
{
    arc::scene::transform_component transform;
    transform.position = {13.0f, -7.5f, 21.0f};
    transform.rotation =
        arc::math::from_axis_angle(arc::math::normalize(arc::math::vector3f{1.0f, 2.0f, -3.0f}), 0.73f);
    transform.scale = {1.5f, 0.75f, 2.25f};

    const auto affine = arc::scene::local_matrix(transform);
    arc::math::matrix4f fast_inverse;
    arc::math::matrix4f reference_inverse;
    REQUIRE(arc::scene::inverse_affine(affine, fast_inverse));
    REQUIRE(arc::math::try_inverse(affine, reference_inverse));

    const auto identity = arc::math::matmul(affine, fast_inverse);
    for (std::size_t row = 0; row < 4; ++row)
    {
        for (std::size_t column = 0; column < 4; ++column)
        {
            const float expected = row == column ? 1.0f : 0.0f;
            REQUIRE(identity(row, column) == Catch::Approx(expected).margin(0.0001f));
            REQUIRE(fast_inverse(row, column) == Catch::Approx(reference_inverse(row, column)).margin(0.0001f));
        }
    }
}

TEST_CASE("camera world view agrees with the quaternion view matrix")
{
    arc::scene::transform_component camera;
    camera.position = {26.0f, 38.0f, 42.0f};
    camera.rotation = arc::math::from_axis_angle(arc::math::normalize(arc::math::vector3f{0.3f, 1.0f, -0.2f}), -0.91f);
    camera.scale = arc::math::vector3f::one;

    const auto world_view = arc::scene::world_view_matrix(camera);
    const auto quaternion_view = arc::scene::view_matrix(camera);
    for (std::size_t row = 0; row < 4; ++row)
        for (std::size_t column = 0; column < 4; ++column)
            REQUIRE(world_view(row, column) == Catch::Approx(quaternion_view(row, column)).margin(0.0001f));
}

TEST_CASE("render scene extracts visible mesh draw events from active camera")
{
    arc::ecs::world scene;
    arc::render::renderer renderer;
    const arc::render::mesh_handle mesh{.index = 2, .generation = 1};

    const auto camera_entity = scene.create();
    arc::scene::transform_component camera_transform;
    camera_transform.position = arc::math::vector3f{0.0f, 0.0f, 5.0f};
    scene.emplace<arc::scene::transform_component>(camera_entity, camera_transform);
    scene.emplace<arc::scene::camera_component>(camera_entity);

    const auto mesh_entity = scene.create();
    scene.emplace<arc::scene::transform_component>(mesh_entity);
    scene.emplace<arc::scene::selection_component>(mesh_entity, true);
    arc::scene::mesh_renderer_component mesh_renderer;
    mesh_renderer.mesh = mesh;
    mesh_renderer.base_color_tint = arc::math::vector4f{0.2f, 0.4f, 0.6f, 1.0f};
    scene.emplace<arc::scene::mesh_renderer_component>(mesh_entity, mesh_renderer);

    const auto result = arc::scene::render_scene(scene, renderer, 1280, 720, arc::render::render_mode::wireframe,
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

TEST_CASE("render scene honors an explicitly selected editor camera")
{
    arc::ecs::world scene;
    arc::render::renderer renderer;

    const auto game_camera = scene.create();
    arc::scene::transform_component game_transform;
    game_transform.position = {40.0f, 0.0f, 0.0f};
    scene.emplace<arc::scene::transform_component>(game_camera, game_transform);
    scene.emplace<arc::scene::camera_component>(game_camera);

    const auto editor_camera = scene.create();
    arc::scene::transform_component editor_transform;
    editor_transform.position = {0.0f, 12.0f, 7.0f};
    scene.emplace<arc::scene::transform_component>(editor_camera, editor_transform);
    scene.emplace<arc::scene::camera_component>(editor_camera);

    const auto result = arc::scene::render_scene(
        scene, renderer, 1280, 720, arc::render::render_mode::shaded, arc::render::mesh_visualization_mode::standard,
        arc::render::editor_overlay_mode::none, true, {}, 0.0f, {}, editor_camera);
    REQUIRE(result.camera_found);

    const auto packet = renderer.frame_queue().commit(1);
    REQUIRE(packet.events.size() == 1);
    const auto& world = *std::get<arc::render::render_world_event>(packet.events[0].payload).packet;
    REQUIRE(world.camera.position[0] == Catch::Approx(0.0f));
    REQUIRE(world.camera.position[1] == Catch::Approx(12.0f));
    REQUIRE(world.camera.position[2] == Catch::Approx(7.0f));
}

TEST_CASE("world environment solar clock and validation are deterministic")
{
    REQUIRE(arc::scene::is_valid_gregorian_date(2024, 2, 29));
    REQUIRE_FALSE(arc::scene::is_valid_gregorian_date(2023, 2, 29));

    const auto noon = arc::scene::calculate_solar_position(0.0f, 0.0f, 0.0f, 2026, 3, 20, 12.0f, 0.0f);
    REQUIRE(noon.elevation_degrees > 85.0f);
    REQUIRE(std::isfinite(noon.light_direction[0]));
    REQUIRE(arc::math::length(noon.light_direction) == Catch::Approx(1.0f).margin(0.001f));

    const auto phase_a = arc::scene::calculate_moon_phase(2026, 7, 14, 12.0f, 0.0f);
    const auto phase_b = arc::scene::calculate_moon_phase(2026, 7, 14, 12.0f, 0.0f);
    REQUIRE(phase_a == Catch::Approx(phase_b));
    REQUIRE(phase_a >= 0.0f);
    REQUIRE(phase_a < 1.0f);

    arc::scene::world_environment_settings settings;
    arc::scene::apply_world_environment_preset(arc::scene::world_environment_preset::indoor_neutral, settings);
    REQUIRE_FALSE(settings.world.sky_visible);
    REQUIRE(settings.world.affect_lighting);
    REQUIRE(settings.lighting.source == arc::scene::environment_lighting_source::constant_color);

    settings.atmosphere.atmosphere_radius = settings.atmosphere.planet_radius;
    const auto invalid = arc::scene::validate_world_environment(settings);
    REQUIRE_FALSE(invalid.valid);
    REQUIRE_FALSE(invalid.errors.empty());
}

TEST_CASE("world environment validation rejects every authored range family")
{
    using settings = arc::scene::world_environment_settings;
    using mutation = std::function<void(settings&)>;
    const float nan = std::numeric_limits<float>::quiet_NaN();
    const std::vector<mutation> invalid_mutations{
        [=](settings& value) { value.world.radiance_intensity = nan; },
        [](settings& value) { value.atmosphere.planet_radius = 0.0f; },
        [](settings& value) { value.atmosphere.atmosphere_radius = value.atmosphere.planet_radius; },
        [](settings& value) { value.atmosphere.mie_anisotropy = 1.0f; },
        [](settings& value) { value.atmosphere.rayleigh_strength = -0.1f; },
        [](settings& value) { value.atmosphere.rayleigh_scale_height = 0.0f; },
        [](settings& value)
        {
            value.celestial.year = 2023;
            value.celestial.month = 2;
            value.celestial.day = 29;
        },
        [](settings& value) { value.celestial.latitude_degrees = 90.1f; },
        [](settings& value) { value.celestial.longitude_degrees = -180.1f; },
        [](settings& value) { value.celestial.utc_offset_hours = 14.1f; },
        [](settings& value) { value.celestial.local_time_hours = 24.0f; },
        [](settings& value) { value.celestial.moon_phase = -0.01f; },
        [](settings& value) { value.celestial.star_density = 1.01f; },
        [](settings& value) { value.clouds.cumulus.coverage = 1.01f; },
        [](settings& value) { value.clouds.cirrus.density = -0.01f; },
        [](settings& value) { value.clouds.cumulus.scale = 0.0f; },
        [](settings& value) { value.clouds.cirrus.wind_speed = -1.0f; },
        [](settings& value) { value.fog.max_opacity = 1.01f; },
        [](settings& value) { value.fog.density = -0.01f; },
        [](settings& value) { value.lighting.diffuse_intensity = -0.01f; }};

    for (const auto& mutate : invalid_mutations)
    {
        settings value;
        mutate(value);
        INFO("invalid mutation index " << (&mutate - invalid_mutations.data()));
        REQUIRE_FALSE(arc::scene::validate_world_environment(value).valid);
    }
}

TEST_CASE("world environment presets preserve runtime identity and aggregate writes are atomic")
{
    arc::scene::world_environment_settings settings;
    settings.world.hdri_texture = {.index = 7, .generation = 2};
    settings.celestial.sun_light = {.index = 9, .generation = 3};
    settings.celestial.animation_time_seconds = 123.0f;
    settings.lighting.environment = {.index = 11, .generation = 4};
    settings.lighting.hdri_texture = {.index = 13, .generation = 5};

    for (const auto preset :
         {arc::scene::world_environment_preset::clear_day, arc::scene::world_environment_preset::alpine_late_morning,
          arc::scene::world_environment_preset::golden_hour, arc::scene::world_environment_preset::overcast,
          arc::scene::world_environment_preset::night, arc::scene::world_environment_preset::indoor_neutral})
    {
        arc::scene::apply_world_environment_preset(preset, settings);
        REQUIRE(settings.world.hdri_texture.index == 7);
        REQUIRE(settings.celestial.sun_light.index == 9);
        REQUIRE(settings.celestial.animation_time_seconds == Catch::Approx(123.0f));
        REQUIRE(settings.lighting.environment.index == 11);
        REQUIRE(settings.lighting.hdri_texture.index == 13);
        REQUIRE(arc::scene::validate_world_environment(settings).valid);
    }

    arc::ecs::world registry;
    const auto entity = registry.create();
    REQUIRE(arc::scene::set_world_environment_settings(registry, entity, settings));
    const auto stored = arc::scene::read_world_environment_settings(registry, entity);
    REQUIRE(stored.has_value());
    REQUIRE(stored->celestial.sun_light.index == 9);

    auto invalid = settings;
    invalid.atmosphere.atmosphere_radius = invalid.atmosphere.planet_radius;
    REQUIRE_FALSE(arc::scene::set_world_environment_settings(registry, entity, invalid));
    REQUIRE(registry.get<arc::scene::sky_atmosphere_component>(entity).atmosphere_radius >
            registry.get<arc::scene::sky_atmosphere_component>(entity).planet_radius);

    registry.remove<arc::scene::height_fog_component>(entity);
    REQUIRE_FALSE(arc::scene::read_world_environment_settings(registry, entity).has_value());
}

TEST_CASE("simulated geographic environment advances time and drives its linked sun")
{
    arc::ecs::world registry;
    const auto sun = registry.create();
    registry.emplace<arc::scene::transform_component>(sun);
    registry.emplace<arc::scene::directional_light_component>(sun);

    const auto environment = registry.create();
    registry.emplace<arc::scene::world_environment_component>(environment);
    arc::scene::celestial_sky_component celestial;
    celestial.sun_light = sun;
    celestial.sun_mode = arc::scene::sun_position_mode::geographic;
    celestial.time_mode = arc::scene::celestial_time_mode::simulated;
    celestial.playing = true;
    celestial.loop_day = false;
    celestial.year = 2024;
    celestial.month = 2;
    celestial.day = 28;
    celestial.local_time_hours = 23.5f;
    celestial.time_scale = 3600.0f;
    registry.emplace<arc::scene::celestial_sky_component>(environment, celestial);

    const auto original_rotation = registry.get<arc::scene::transform_component>(sun).rotation;
    arc::scene::update_world_environments(registry, 1.0f);
    const auto& advanced = registry.get<arc::scene::celestial_sky_component>(environment);
    REQUIRE(advanced.year == 2024);
    REQUIRE(advanced.month == 2);
    REQUIRE(advanced.day == 29);
    REQUIRE(advanced.local_time_hours == Catch::Approx(0.5f));
    const auto& updated_rotation = registry.get<arc::scene::transform_component>(sun).rotation;
    REQUIRE((updated_rotation[0] != original_rotation[0] || updated_rotation[1] != original_rotation[1] ||
             updated_rotation[2] != original_rotation[2] || updated_rotation[3] != original_rotation[3]));
    const auto& updated_light = registry.get<arc::scene::directional_light_component>(sun);
    REQUIRE(updated_light.use_color_temperature);
    REQUIRE(updated_light.temperature_kelvin >= 1000.0f);
    REQUIRE(updated_light.temperature_kelvin <= 40000.0f);
}

TEST_CASE("automatic geographic sun writes physically meaningful lux")
{
    arc::ecs::world registry;
    const auto sun = registry.create();
    registry.emplace<arc::scene::transform_component>(sun);
    registry.emplace<arc::scene::directional_light_component>(sun);

    const auto environment = registry.create();
    registry.emplace<arc::scene::world_environment_component>(environment);
    arc::scene::celestial_sky_component celestial;
    celestial.sun_light = sun;
    celestial.sun_mode = arc::scene::sun_position_mode::geographic;
    celestial.time_mode = arc::scene::celestial_time_mode::fixed;
    celestial.latitude_degrees = 0.0f;
    celestial.longitude_degrees = 0.0f;
    celestial.year = 2024;
    celestial.month = 3;
    celestial.day = 20;
    celestial.local_time_hours = 12.0f;
    celestial.utc_offset_hours = 0.0f;
    registry.emplace<arc::scene::celestial_sky_component>(environment, celestial);

    arc::scene::update_world_environments(registry, 0.0f);
    const auto& light = registry.get<arc::scene::directional_light_component>(sun);
    REQUIRE(light.enabled);
    REQUIRE(light.intensity_unit == arc::render::light_intensity_unit::lux);
    REQUIRE(light.intensity > 50000.0f);
}

TEST_CASE("world environment clocks handle reverse midnight leap years and fixed time")
{
    arc::ecs::world registry;
    const auto environment = registry.create();
    registry.emplace<arc::scene::world_environment_component>(environment);
    arc::scene::celestial_sky_component celestial;
    celestial.time_mode = arc::scene::celestial_time_mode::simulated;
    celestial.playing = true;
    celestial.loop_day = false;
    celestial.year = 2024;
    celestial.month = 3;
    celestial.day = 1;
    celestial.local_time_hours = 0.25f;
    celestial.time_scale = 3600.0f;
    registry.emplace<arc::scene::celestial_sky_component>(environment, celestial);

    arc::scene::update_world_environments(registry, -1.0f);
    auto& reversed = registry.get<arc::scene::celestial_sky_component>(environment);
    REQUIRE(reversed.year == 2024);
    REQUIRE(reversed.month == 2);
    REQUIRE(reversed.day == 29);
    REQUIRE(reversed.local_time_hours == Catch::Approx(23.25f));

    reversed.time_mode = arc::scene::celestial_time_mode::fixed;
    reversed.local_time_hours = 7.5f;
    arc::scene::update_world_environments(registry, 100.0f);
    REQUIRE(reversed.local_time_hours == Catch::Approx(7.5f));

    reversed.time_mode = arc::scene::celestial_time_mode::system_clock;
    reversed.utc_offset_hours = -8.0f;
    arc::scene::update_world_environments(registry, 0.0f);
    REQUIRE(arc::scene::is_valid_gregorian_date(reversed.year, reversed.month, reversed.day));
    REQUIRE(reversed.local_time_hours >= 0.0f);
    REQUIRE(reversed.local_time_hours < 24.0f);

    const auto north = arc::scene::calculate_solar_position(45.0f, 0.0f, 90.0f, 2026, 6, 21, 12.0f, 0.0f);
    const auto south = arc::scene::calculate_solar_position(45.0f, 0.0f, 0.0f, 2026, 6, 21, 12.0f, 0.0f);
    REQUIRE(std::abs(arc::math::dot(north.light_direction, south.light_direction)) < 0.9f);
}

TEST_CASE("render scene extracts one authoritative world environment snapshot")
{
    arc::ecs::world registry;
    arc::render::renderer renderer;
    const auto camera = registry.create();
    registry.emplace<arc::scene::transform_component>(camera);
    registry.emplace<arc::scene::camera_component>(camera);

    const auto environment = registry.create();
    auto& world = registry.emplace<arc::scene::world_environment_component>(environment);
    world.sky_visible = false;
    world.affect_lighting = true;
    registry.emplace<arc::scene::sky_atmosphere_component>(environment);
    registry.emplace<arc::scene::celestial_sky_component>(environment);
    registry.emplace<arc::scene::cloud_layers_component>(environment);
    registry.emplace<arc::scene::height_fog_component>(environment);
    registry.emplace<arc::scene::environment_lighting_component>(environment);

    REQUIRE(arc::scene::render_scene(registry, renderer, 640, 360).camera_found);
    auto packet = renderer.frame_queue().commit(1);
    const auto& first = *std::get<arc::render::render_world_event>(packet.events[0].payload).packet;
    REQUIRE(first.environment.enabled);
    REQUIRE_FALSE(first.environment.sky_visible);
    REQUIRE(first.environment.affect_lighting);
    REQUIRE(first.environment.fog.enabled);

    world.enabled = false;
    REQUIRE(arc::scene::render_scene(registry, renderer, 640, 360).camera_found);
    packet = renderer.frame_queue().commit(2);
    const auto& second = *std::get<arc::render::render_world_event>(packet.events[0].payload).packet;
    REQUIRE_FALSE(second.environment.enabled);
    REQUIRE_FALSE(second.environment.fog.enabled);
}

TEST_CASE("render scene rejects legacy and incomplete environments and selects duplicates deterministically")
{
    arc::ecs::world registry;
    arc::render::renderer renderer;
    const auto camera = registry.create();
    registry.emplace<arc::scene::transform_component>(camera);
    registry.emplace<arc::scene::camera_component>(camera);

    const auto legacy_sky = registry.create();
    registry.emplace<arc::scene::sky_atmosphere_component>(legacy_sky);
    auto result = arc::scene::render_scene(registry, renderer, 320, 180);
    REQUIRE(result.world_environment_count == 0);
    REQUIRE_FALSE(result.environment.enabled);
    renderer.frame_queue().commit(1);

    const auto incomplete = registry.create();
    registry.emplace<arc::scene::world_environment_component>(incomplete);
    result = arc::scene::render_scene(registry, renderer, 320, 180);
    REQUIRE(result.world_environment_count == 1);
    REQUIRE_FALSE(result.environment.fallback_reason.empty());
    renderer.frame_queue().commit(2);
    registry.emplace<arc::scene::active_component>(incomplete, false);

    arc::scene::world_environment_settings first;
    first.world.source = arc::scene::sky_source::solid_color;
    first.world.solid_color = {0.2f, 0.3f, 0.4f};
    const auto first_entity = registry.create();
    REQUIRE(arc::scene::set_world_environment_settings(registry, first_entity, first));
    arc::scene::world_environment_settings second;
    second.world.source = arc::scene::sky_source::hdri;
    const auto second_entity = registry.create();
    REQUIRE(arc::scene::set_world_environment_settings(registry, second_entity, second));

    arc::scene::scene_render_visibility visibility;
    visibility.sky = false;
    visibility.fog = false;
    result = arc::scene::render_scene(registry, renderer, 320, 180, arc::render::render_mode::shaded,
                                      arc::render::mesh_visualization_mode::standard,
                                      arc::render::editor_overlay_mode::selected_wireframe, true, visibility);
    REQUIRE(result.world_environment_count == 2);
    REQUIRE(result.environment.source == arc::render::sky_source_mode::solid_color);
    REQUIRE_FALSE(result.environment.sky_visible);
    REQUIRE_FALSE(result.environment.fog.enabled);
    REQUIRE_FALSE(result.environment.fallback_reason.empty());
}

TEST_CASE("render scene culling uses transformed dirty local bounds")
{
    arc::ecs::world scene;
    arc::render::renderer renderer;
    const arc::render::mesh_handle mesh{.index = 2, .generation = 1};

    const auto camera_entity = scene.create();
    arc::scene::transform_component camera_transform;
    camera_transform.position = arc::math::vector3f{0.0f, 0.0f, 5.0f};
    scene.emplace<arc::scene::transform_component>(camera_entity, camera_transform);
    scene.emplace<arc::scene::camera_component>(camera_entity);

    const auto mesh_entity = scene.create();
    arc::scene::transform_component transform;
    transform.position = arc::math::vector3f{20.0f, 0.0f, 0.0f};
    scene.emplace<arc::scene::transform_component>(mesh_entity, transform);
    scene.emplace<arc::scene::bounds_component>(
        mesh_entity,
        arc::geometric::box3f{arc::geometric::point3f{arc::math::vector3f{-22.0f, -1.0f, -6.0f}},
                              arc::geometric::point3f{arc::math::vector3f{-18.0f, 1.0f, -4.0f}}},
        arc::geometric::box3f{}, true);
    scene.emplace<arc::scene::mesh_renderer_component>(
        mesh_entity, arc::scene::mesh_renderer_component{.mesh = arc::render::geometry_resource_handle{mesh}});

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

TEST_CASE("mesh renderer keeps conventional fallback when virtual geometry is unavailable")
{
    arc::ecs::world scene;
    arc::render::renderer renderer;

    const auto camera_entity = scene.create();
    arc::scene::transform_component camera_transform;
    camera_transform.position = arc::math::vector3f{0.0f, 0.0f, 5.0f};
    scene.emplace<arc::scene::transform_component>(camera_entity, camera_transform);
    scene.emplace<arc::scene::camera_component>(camera_entity);

    arc::render::mesh_data source;
    source.vertices.resize(6);
    source.vertices[0].position[0] = -0.5f;
    source.vertices[0].position[1] = -0.5f;
    source.vertices[0].position[2] = 0.0f;
    source.vertices[1].position[0] = 0.5f;
    source.vertices[1].position[1] = -0.5f;
    source.vertices[1].position[2] = 0.0f;
    source.vertices[2].position[0] = 0.0f;
    source.vertices[2].position[1] = 0.5f;
    source.vertices[2].position[2] = 0.0f;
    source.vertices[3].position[0] = 100.0f;
    source.vertices[3].position[1] = -0.5f;
    source.vertices[3].position[2] = 0.0f;
    source.vertices[4].position[0] = 101.0f;
    source.vertices[4].position[1] = -0.5f;
    source.vertices[4].position[2] = 0.0f;
    source.vertices[5].position[0] = 100.5f;
    source.vertices[5].position[1] = 0.5f;
    source.vertices[5].position[2] = 0.0f;
    source.indices = {0, 1, 2, 3, 4, 5};
    const auto conventional_mesh = renderer.create_mesh(source);
    const auto virtual_mesh =
        renderer.create_virtual_mesh(arc::render::build_virtual_mesh(source, {.max_triangles_per_cluster = 1}));

    const auto mesh_entity = scene.create();
    scene.emplace<arc::scene::transform_component>(mesh_entity);
    scene.emplace<arc::scene::selection_component>(mesh_entity, true);
    arc::scene::mesh_renderer_component mesh_renderer;
    mesh_renderer.mesh = conventional_mesh;
    mesh_renderer.mesh.virtualized = virtual_mesh;
    mesh_renderer.representation = arc::render::geometry_representation_policy::auto_select;
    mesh_renderer.base_color_tint = arc::math::vector4f{0.8f, 0.2f, 0.4f, 1.0f};
    scene.emplace<arc::scene::mesh_renderer_component>(mesh_entity, mesh_renderer);

    const auto result = arc::scene::render_scene(scene, renderer, 1280, 720);
    REQUIRE(result.camera_found);
    REQUIRE(result.renderable_count == 1);
    REQUIRE(result.selected_count == 1);
    REQUIRE(result.submitted_draw_count == 1);
    REQUIRE(result.culled_count == 0);
    REQUIRE(result.culled_virtual_cluster_count == 0);

    const auto packet = renderer.frame_queue().commit(1);
    REQUIRE(packet.events.size() == 4);
    REQUIRE(packet.events[0].type() == arc::render::render_event_type::mesh_upload);
    REQUIRE(packet.events[1].type() == arc::render::render_event_type::lighting_geometry_upload);
    REQUIRE(packet.events[2].type() == arc::render::render_event_type::virtual_mesh_upload);
    REQUIRE(packet.events[3].type() == arc::render::render_event_type::render_world);
    const auto& world_event = std::get<arc::render::render_world_event>(packet.events[3].payload);
    REQUIRE(world_event.packet);
    REQUIRE(world_event.packet->visible_items.size() == 1);
    REQUIRE(world_event.packet->virtual_items.empty());
    const auto& item = world_event.packet->items[world_event.packet->visible_items[0]];
    REQUIRE(item.mesh == conventional_mesh);
    REQUIRE(item.object_id.index == mesh_entity.index);
    REQUIRE(item.object_id.generation == mesh_entity.generation);
    REQUIRE(item.selected);
    REQUIRE(item.base_color_tint[0] == Catch::Approx(0.8f));
}

TEST_CASE("render scene can request wireframe overlay for every draw")
{
    arc::ecs::world scene;
    arc::render::renderer renderer;
    const arc::render::mesh_handle mesh{.index = 2, .generation = 1};

    const auto camera_entity = scene.create();
    arc::scene::transform_component camera_transform;
    camera_transform.position = arc::math::vector3f{0.0f, 0.0f, 5.0f};
    scene.emplace<arc::scene::transform_component>(camera_entity, camera_transform);
    scene.emplace<arc::scene::camera_component>(camera_entity);

    const auto mesh_entity = scene.create();
    scene.emplace<arc::scene::transform_component>(mesh_entity);
    scene.emplace<arc::scene::mesh_renderer_component>(
        mesh_entity, arc::scene::mesh_renderer_component{.mesh = arc::render::geometry_resource_handle{mesh}});

    const auto result = arc::scene::render_scene(scene, renderer, 1280, 720, arc::render::render_mode::shaded,
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
    arc::ecs::world scene;
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
    arc::ecs::world scene;
    arc::render::renderer renderer;
    const arc::render::mesh_handle mesh{.index = 3, .generation = 1};

    const auto camera_entity = scene.create();
    arc::scene::transform_component camera_transform;
    camera_transform.position = arc::math::vector3f{0.0f, 0.0f, 5.0f};
    scene.emplace<arc::scene::active_component>(camera_entity);
    scene.emplace<arc::scene::transform_component>(camera_entity, camera_transform);
    scene.emplace<arc::scene::camera_component>(camera_entity);

    const auto mesh_entity = scene.create();
    scene.emplace<arc::scene::active_component>(mesh_entity, false);
    scene.emplace<arc::scene::transform_component>(mesh_entity);
    scene.emplace<arc::scene::mesh_renderer_component>(
        mesh_entity, arc::scene::mesh_renderer_component{.mesh = arc::render::geometry_resource_handle{mesh}});

    const auto sun = scene.create();
    scene.emplace<arc::scene::name_component>(sun, "Sun");
    scene.emplace<arc::scene::active_component>(sun);
    scene.emplace<arc::scene::transform_component>(sun);
    scene.emplace<arc::scene::directional_light_component>(sun, arc::math::vector3f{1.0f, 0.9f, 0.7f}, 2.0f, true);
    auto& sun_light = scene.get<arc::scene::directional_light_component>(sun);
    sun_light.use_color_temperature = true;
    sun_light.temperature_kelvin = 3000.0f;
    sun_light.intensity_unit = arc::render::light_intensity_unit::lux;
    sun_light.shadow.resolution = 4096;
    sun_light.shadow.filter = arc::render::shadow_filter::pcf_5x5;
    sun_light.shadow.priority = 700;
    sun_light.shadow.contact_shadows = true;
    sun_light.shadow.contact_shadow_length = 0.8f;
    sun_light.shadow.map_method = arc::render::shadow_map_method::virtualized;
    sun_light.cascades.cascade_count = 3;
    sun_light.cascades.maximum_distance = 175.0f;
    sun_light.cascades.split_lambda = 0.72f;
    sun_light.cascades.blend_fraction = 0.12f;
    scene.emplace<arc::scene::mobility_component>(sun, arc::render::render_mobility::stationary);

    const auto disabled_point = scene.create();
    scene.emplace<arc::scene::active_component>(disabled_point);
    scene.emplace<arc::scene::transform_component>(disabled_point);
    scene.emplace<arc::scene::point_light_component>(disabled_point, arc::math::vector3f{0.2f, 0.3f, 1.0f}, 10.0f, 5.0f,
                                                     false, false);

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
    REQUIRE(light.shadow.priority == 700);
    REQUIRE(light.shadow.contact_shadows);
    REQUIRE(light.shadow.contact_shadow_length == Catch::Approx(0.8f));
    REQUIRE(light.shadow.map_method == arc::render::shadow_map_method::virtualized);
    REQUIRE(light.cascades.cascade_count == 3);
    REQUIRE(light.cascades.maximum_distance == Catch::Approx(175.0f));
    REQUIRE(light.cascades.split_lambda == Catch::Approx(0.72f));
    REQUIRE(light.cascades.blend_fraction == Catch::Approx(0.12f));
    REQUIRE(light.mobility == arc::render::render_mobility::stationary);
    REQUIRE(light.color[0] >= light.color[2]);
    REQUIRE(world_event.packet->reflection_probes.size() == 1);
    REQUIRE(world_event.packet->reflection_probes[0].radius == Catch::Approx(8.0f));
}

TEST_CASE("render scene extracts skinned and instanced render items")
{
    arc::ecs::world scene;
    arc::render::renderer renderer;

    const auto camera_entity = scene.create();
    arc::scene::transform_component camera_transform;
    camera_transform.position = arc::math::vector3f{0.0f, 0.0f, 5.0f};
    scene.emplace<arc::scene::transform_component>(camera_entity, camera_transform);
    scene.emplace<arc::scene::camera_component>(camera_entity);

    const auto skinned = scene.create();
    scene.emplace<arc::scene::transform_component>(skinned);
    scene.emplace<arc::scene::skinned_mesh_renderer_component>(
        skinned, arc::render::mesh_handle{.index = 10, .generation = 1},
        arc::render::material_handle{.index = 2, .generation = 1},
        arc::render::buffer_handle{.index = 4, .generation = 1}, 64u, true);

    const auto instanced = scene.create();
    scene.emplace<arc::scene::transform_component>(instanced);
    scene.emplace<arc::scene::instance_group_component>(
        instanced, arc::render::mesh_handle{.index = 11, .generation = 1},
        arc::render::material_handle{.index = 3, .generation = 1}, 12u, true);

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
    arc::ecs::world scene;
    arc::render::renderer renderer;
    const arc::render::mesh_handle base_mesh{.index = 1, .generation = 1};
    const arc::render::mesh_handle lod_mesh{.index = 9, .generation = 1};

    const auto camera_entity = scene.create();
    arc::scene::transform_component camera_transform;
    camera_transform.position = arc::math::vector3f{0.0f, 0.0f, 5.0f};
    scene.emplace<arc::scene::transform_component>(camera_entity, camera_transform);
    scene.emplace<arc::scene::camera_component>(camera_entity);

    const auto mesh_entity = scene.create();
    scene.emplace<arc::scene::transform_component>(mesh_entity);
    scene.emplace<arc::scene::mesh_renderer_component>(
        mesh_entity, arc::scene::mesh_renderer_component{.mesh = arc::render::geometry_resource_handle{base_mesh}});
    scene.emplace<arc::scene::lod_component>(
        mesh_entity, std::vector<arc::scene::lod_level>{{.screen_coverage = 1.0f, .mesh = lod_mesh}}, true);

    const auto result = arc::scene::render_scene(scene, renderer, 1280, 720);
    REQUIRE(result.submitted_draw_count == 1);

    const auto packet = renderer.frame_queue().commit(4);
    const auto& world_event = std::get<arc::render::render_world_event>(packet.events[0].payload);
    REQUIRE(world_event.packet->items[0].mesh == lod_mesh);
}

TEST_CASE("terrain heightfields generate deterministic normalized resource data")
{
    arc::scene::terrain_component first;
    first.size = 180.0f;
    first.subdivisions = 256;
    first.chunk_quads = 128;
    first.height_scale = 28.0f;
    auto second = first;

    arc::scene::generate_terrain_heightfield(first);
    arc::scene::generate_terrain_heightfield(second);

    REQUIRE(arc::scene::terrain_heightfield_valid(first));
    REQUIRE(first.heights == second.heights);
    REQUIRE(first.layer_weights == second.layer_weights);
    REQUIRE(first.heights.size() == 257u * 257u);
    for (const auto& weights : first.layer_weights)
        REQUIRE(static_cast<unsigned>(weights[0]) + weights[1] + weights[2] + weights[3] == 255u);

    const auto hierarchy = arc::render::build_terrain_hierarchy(first.heights, 257u, first.size, first.size,
                                                                {.patch_quads = first.patch_quads});
    REQUIRE(hierarchy.root < hierarchy.nodes.size());
    REQUIRE(hierarchy.leaf_count == 64u);
    REQUIRE(hierarchy.nodes[hierarchy.root].samples.max_x == 256u);
    REQUIRE(hierarchy.nodes[hierarchy.root].samples.max_z == 256u);
}

TEST_CASE("terrain brushes sculpt smooth flatten and paint normalized layers")
{
    arc::scene::terrain_component terrain;
    terrain.size = 32.0f;
    terrain.subdivisions = 32;
    terrain.chunk_quads = 16;
    terrain.height_scale = 4.0f;
    arc::scene::generate_terrain_heightfield(terrain);
    const auto center_index = 16u * 33u + 16u;
    const float original = terrain.heights[center_index];

    arc::scene::terrain_brush_settings brush;
    brush.radius = 4.0f;
    brush.strength = 1.0f;
    auto dirty = arc::scene::apply_terrain_brush(terrain, {0.0f, original, 0.0f}, brush, 1.0f / 60.0f);
    REQUIRE(dirty.valid);
    REQUIRE(terrain.heights[center_index] > original);

    brush.tool = arc::scene::terrain_brush_tool::flatten;
    brush.flatten_height = 2.0f;
    arc::scene::apply_terrain_brush(terrain, {0.0f, 0.0f, 0.0f}, brush, 1.0f);
    REQUIRE(terrain.heights[center_index] == Catch::Approx(2.0f));

    brush.tool = arc::scene::terrain_brush_tool::paint;
    brush.active_layer = 3;
    const auto before = terrain.layer_weights[center_index][3];
    arc::scene::apply_terrain_brush(terrain, {0.0f, 0.0f, 0.0f}, brush, 1.0f);
    const auto weights = terrain.layer_weights[center_index];
    REQUIRE(weights[3] >= before);
    REQUIRE(static_cast<unsigned>(weights[0]) + weights[1] + weights[2] + weights[3] == 255u);
}

TEST_CASE("terrain raycasts return the actual heightfield surface")
{
    arc::scene::terrain_component terrain;
    terrain.size = 8.0f;
    terrain.subdivisions = 8;
    terrain.chunk_quads = 4;
    terrain.height_scale = 1.0f;
    arc::scene::generate_terrain_heightfield(terrain);
    const auto hit = arc::scene::raycast_terrain(terrain, {0.0f, 20.0f, 0.0f}, {0.0f, -1.0f, 0.0f});
    REQUIRE(hit.hit);
    REQUIRE(hit.position[1] == Catch::Approx(arc::scene::sample_terrain_height(terrain, 0.0f, 0.0f)).margin(0.001f));
}

TEST_CASE("terrain authoring validates supported creation profiles and budgets")
{
    arc::scene::terrain_creation_descriptor descriptor;
    REQUIRE(arc::scene::validate_terrain_creation(descriptor).succeeded);
    descriptor.sample_resolution = 4096;
    REQUIRE_FALSE(arc::scene::validate_terrain_creation(descriptor).succeeded);
    descriptor.sample_resolution = 1025;
    descriptor.patch_quads = 17;
    REQUIRE_FALSE(arc::scene::validate_terrain_creation(descriptor).succeeded);
    const auto estimate = arc::scene::estimate_terrain_memory(2049);
    REQUIRE(estimate.cpu_bytes > 0);
    REQUIRE(estimate.history_bytes > estimate.cpu_bytes);
}

TEST_CASE("terrain generators are deterministic and flat creation remains explicit")
{
    arc::scene::terrain_component first;
    first.size = 64.0f;
    first.subdivisions = 256;
    arc::scene::terrain_component second = first;
    const arc::scene::terrain_generation_descriptor generated{.generator_id = "arc.terrain.domain_warped.v1",
                                                              .seed = 42,
                                                              .minimum_elevation = -4.0f,
                                                              .maximum_elevation = 36.0f};
    REQUIRE(arc::scene::generate_terrain(first, generated).succeeded);
    REQUIRE(arc::scene::generate_terrain(second, generated).succeeded);
    REQUIRE(first.heights == second.heights);
    REQUIRE(first.layer_weights == second.layer_weights);
    REQUIRE(arc::scene::generate_terrain(
                first, {.generator_id = "arc.terrain.flat.v1", .minimum_elevation = 3.0f, .maximum_elevation = 9.0f})
                .succeeded);
    REQUIRE(std::all_of(first.heights.begin(), first.heights.end(), [](float value) { return value == 3.0f; }));
}

TEST_CASE("terrain heightmap import preserves 16 bit range and resamples normalized weights")
{
    arc::scene::terrain_heightmap heightmap;
    heightmap.width = 3;
    heightmap.height = 2;
    heightmap.samples = {0, 16384, 32768, 32768, 49152, 65535};
    arc::scene::terrain_component terrain;
    REQUIRE(arc::scene::import_terrain_heightmap(terrain, heightmap,
                                                 {.target_resolution = 257,
                                                  .physical_size = 80.0f,
                                                  .minimum_elevation = -10.0f,
                                                  .maximum_elevation = 30.0f,
                                                  .normalize_source_range = false})
                .succeeded);
    REQUIRE(terrain.heights.size() == 257u * 257u);
    REQUIRE(*std::min_element(terrain.heights.begin(), terrain.heights.end()) == Catch::Approx(-10.0f).margin(0.02f));
    REQUIRE(*std::max_element(terrain.heights.begin(), terrain.heights.end()) == Catch::Approx(30.0f).margin(0.02f));
    REQUIRE(arc::scene::resample_terrain(terrain, {.sample_resolution = 513, .physical_size = 160.0f}).succeeded);
    REQUIRE(terrain.subdivisions == 512);
    REQUIRE(terrain.size == 160.0f);
    for (const auto& weights : terrain.layer_weights)
        REQUIRE(std::accumulate(weights.begin(), weights.end(), 0u) == 255u);
    const auto exported =
        arc::scene::export_terrain_heightmap(terrain, {.minimum_elevation = -10.0f, .maximum_elevation = 30.0f});
    REQUIRE(exported.width == 513);
    REQUIRE(exported.samples.size() == terrain.heights.size());
}

TEST_CASE("terrain hierarchy raycast matches exact source raycast")
{
    arc::scene::terrain_component terrain;
    terrain.size = 32.0f;
    terrain.subdivisions = 256;
    REQUIRE(arc::scene::generate_terrain(terrain, {.generator_id = "arc.terrain.domain_warped.v1",
                                                   .seed = 7,
                                                   .minimum_elevation = 0.0f,
                                                   .maximum_elevation = 12.0f})
                .succeeded);
    const auto hierarchy =
        arc::render::build_terrain_hierarchy(terrain.heights, 257, terrain.size, terrain.size, {.patch_quads = 32});
    const auto exact = arc::scene::raycast_terrain(terrain, {3.0f, 50.0f, -2.0f}, {0.0f, -1.0f, 0.0f});
    const auto accelerated = arc::scene::raycast_terrain(terrain, hierarchy, {3.0f, 50.0f, -2.0f}, {0.0f, -1.0f, 0.0f});
    REQUIRE(accelerated.hit == exact.hit);
    REQUIRE(accelerated.distance == Catch::Approx(exact.distance).margin(0.001f));
}

TEST_CASE("generated scene metadata provides ordinary persistence field codecs")
{
    arc::persistence::component_persistence_registry registry;
    REQUIRE(arc::scene::register_persistence_components(registry));

    arc::scene::transform_component source;
    source.position = {3.0f, -2.0f, 7.5f};
    source.scale = {2.0f, 3.0f, 4.0f};
    arc::persistence::archive_component_record encoded;
    const auto type = arc::ecs::component_metadata<arc::scene::transform_component>().id;
    REQUIRE(registry.encode(type, &source, encoded));
    REQUIRE(encoded.fields.size() == 3);

    arc::scene::transform_component destination;
    REQUIRE(registry.decode(type, &destination, encoded));
    for (std::size_t component = 0; component < 3; ++component)
    {
        REQUIRE(destination.position[component] == Catch::Approx(source.position[component]));
        REQUIRE(destination.scale[component] == Catch::Approx(source.scale[component]));
    }
    for (std::size_t component = 0; component < 4; ++component)
        REQUIRE(destination.rotation[component] == Catch::Approx(source.rotation[component]));
}

TEST_CASE("camera persistence migrations cover every schema version")
{
    arc::persistence::schema_migration_registry migrations;
    REQUIRE(arc::scene::register_persistence_migrations(migrations));

    arc::persistence::archive_component_record camera;
    camera.type = arc::ecs::component_metadata<arc::scene::camera_component>().id;
    camera.schema_version = 1;

    const auto target_version = arc::ecs::component_metadata<arc::scene::camera_component>().schema_version;
    REQUIRE(target_version == 3);
    REQUIRE(migrations.migrate(camera, target_version));
    REQUIRE(camera.schema_version == target_version);
}

TEST_CASE("world environment owns validated indirect lighting settings")
{
    arc::ecs::world world;
    const auto environment = world.create();
    arc::scene::world_environment_settings settings;
    settings.indirect_lighting.method = arc::render::indirect_lighting_method::software;
    settings.indirect_lighting.diffuse_intensity = 1.25f;
    settings.indirect_lighting.reflection_intensity = 0.75f;
    settings.indirect_lighting.maximum_trace_distance = 180.0f;
    REQUIRE(arc::scene::set_world_environment_settings(world, environment, settings));

    const auto restored = arc::scene::read_world_environment_settings(world, environment);
    REQUIRE(restored.has_value());
    REQUIRE(restored->indirect_lighting.method == arc::render::indirect_lighting_method::software);
    REQUIRE(restored->indirect_lighting.diffuse_intensity == Catch::Approx(1.25f));
    REQUIRE(world.has<arc::scene::indirect_lighting_component>(environment));

    settings.indirect_lighting.maximum_trace_distance = 0.0f;
    REQUIRE_FALSE(arc::scene::validate_world_environment(settings).valid);
    REQUIRE_FALSE(arc::scene::set_world_environment_settings(world, environment, settings));
    REQUIRE(world.get<arc::scene::indirect_lighting_component>(environment).maximum_trace_distance ==
            Catch::Approx(180.0f));
}
