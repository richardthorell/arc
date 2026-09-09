#include <arc/scene/scene.h>

#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <array>
#include <cstdint>
#include <memory>

namespace
{

arc::scene::terrain_component make_test_terrain()
{
    arc::scene::terrain_component terrain;
    terrain.size = 8.0f;
    terrain.subdivisions = 2u;
    terrain.content_revision = 1u;
    terrain.heights = {-0.5f, 0.0f, 0.5f, -0.25f, 0.25f, 0.75f, 0.0f, 0.5f, 1.0f};
    terrain.layer_weights.assign(terrain.heights.size(), std::array<std::uint8_t, 4>{255u, 0u, 0u, 0u});
    return terrain;
}

class virtual_geometry_test_backend final : public arc::render::render_backend
{
public:
    virtual_geometry_test_backend()
    {
        capabilities_.compute_shaders = true;
        capabilities_.storage_buffers = true;
        capabilities_.shader_draw_parameters = true;
        capabilities_.draw_indirect = true;
        capabilities_.gpu_scene_indirect = true;
        capabilities_.hzb_occlusion = true;
        capabilities_.descriptor_indexing = true;
        capabilities_.virtual_geometry_streaming = true;
        capabilities_.bindless_sampled_images = true;
        capabilities_.bindless_samplers = true;
        capabilities_.bindless_material_tables = true;
        capabilities_.virtual_geometry_compute = true;
    }

    arc::render::render_backend_type type() const noexcept override
    {
        return arc::render::render_backend_type::vulkan;
    }

    const arc::render::render_capabilities& capabilities() const noexcept override
    {
        return capabilities_;
    }

    void configure(const arc::render::resolved_render_config&) override {}

    arc::render::render_submit_result submit(const arc::render::render_frame_packet&,
                                             const arc::render::compiled_render_graph&) override
    {
        return arc::render::render_submit_result::success();
    }

    void resize_viewport(std::uint32_t, std::uint32_t) override {}

    arc::render::render_viewport_texture viewport_texture() const noexcept override
    {
        return {};
    }

    arc::render::render_backend_frame_profile last_frame_profile() const override
    {
        return {};
    }

private:
    arc::render::render_capabilities capabilities_{};
};

} // namespace

TEST_CASE("terrain geometry proxy realizes generic resources and rebuilds on content revisions")
{
    arc::render::renderer renderer;
    arc::scene::terrain_render_proxy_cache cache;
    auto terrain = make_test_terrain();
    const auto guid = arc::ecs::generate_entity_guid();

    REQUIRE(cache.synchronize_geometry(guid, terrain, renderer));
    const auto* first = cache.find(guid);
    REQUIRE(first != nullptr);
    REQUIRE(first->geometry.valid());
    REQUIRE(renderer.mesh_alive(first->geometry.conventional));
    REQUIRE(first->geometry.virtualized.valid());
    REQUIRE(renderer.virtual_mesh_alive(first->geometry.virtualized));
    REQUIRE(first->surface_attribute_texture.valid());
    REQUIRE(renderer.texture_alive(first->surface_attribute_texture));
    REQUIRE_FALSE(renderer.terrain_alive(first->handle));
    CHECK(first->synchronized_revision == terrain.content_revision);

    const auto original_geometry = first->geometry;
    const auto original_attributes = first->surface_attribute_texture;
    terrain.material = {.index = 17u, .generation = 3u};
    REQUIRE(cache.synchronize_geometry(guid, terrain, renderer));
    const auto* material_only = cache.find(guid);
    REQUIRE(material_only != nullptr);
    CHECK(material_only->geometry.conventional == original_geometry.conventional);
    CHECK(material_only->geometry.virtualized == original_geometry.virtualized);
    CHECK(material_only->surface_attribute_texture == original_attributes);
    CHECK(material_only->material == terrain.material);

    terrain.layer_weights[4] = {0u, 255u, 0u, 0u};
    ++terrain.content_revision;
    const arc::scene::terrain_dirty_region paint_dirty{
        .min_x = 1u, .min_z = 1u, .max_x = 1u, .max_z = 1u, .valid = true, .weights_changed = true};
    REQUIRE(cache.synchronize_geometry(guid, terrain, renderer, &paint_dirty));
    const auto* repainted = cache.find(guid);
    REQUIRE(repainted != nullptr);
    CHECK(repainted->geometry.conventional == original_geometry.conventional);
    CHECK(repainted->geometry.virtualized == original_geometry.virtualized);
    CHECK(repainted->surface_attribute_texture == original_attributes);
    REQUIRE(renderer.texture_alive(repainted->surface_attribute_texture));
    CHECK(repainted->synchronized_revision == terrain.content_revision);

    terrain.heights[4] += 2.0f;
    ++terrain.content_revision;
    const arc::scene::terrain_dirty_region dirty{
        .min_x = 1u, .min_z = 1u, .max_x = 1u, .max_z = 1u, .valid = true, .heights_changed = true};
    REQUIRE(cache.synchronize_geometry(guid, terrain, renderer, &dirty));
    const auto* rebuilt = cache.find(guid);
    REQUIRE(rebuilt != nullptr);
    REQUIRE(rebuilt->geometry.valid());
    CHECK(rebuilt->geometry.conventional != original_geometry.conventional);
    CHECK(rebuilt->geometry.virtualized != original_geometry.virtualized);
    REQUIRE_FALSE(renderer.mesh_alive(original_geometry.conventional));
    REQUIRE_FALSE(renderer.virtual_mesh_alive(original_geometry.virtualized));
    CHECK(rebuilt->synchronized_revision == terrain.content_revision);

    const auto rebuilt_geometry = rebuilt->geometry;
    REQUIRE(cache.erase_geometry(guid, renderer));
    REQUIRE(cache.find(guid) == nullptr);
    REQUIRE_FALSE(renderer.mesh_alive(rebuilt_geometry.conventional));
    REQUIRE_FALSE(renderer.virtual_mesh_alive(rebuilt_geometry.virtualized));
}

TEST_CASE("render scene submits terrain as a conventional mesh item without dedicated terrain packets")
{
    arc::ecs::world scene;
    arc::render::renderer renderer;
    arc::scene::terrain_render_proxy_cache terrain_proxies;

    const auto camera = scene.create();
    arc::scene::transform_component camera_transform;
    camera_transform.position = {0.0f, 0.0f, 6.0f};
    scene.emplace<arc::scene::transform_component>(camera, camera_transform);
    scene.emplace<arc::scene::camera_component>(camera);

    const auto terrain_entity = scene.create();
    scene.emplace<arc::scene::transform_component>(terrain_entity);
    scene.emplace<arc::scene::selection_component>(terrain_entity, true);
    const auto guid = arc::ecs::generate_entity_guid();
    auto& persistent = scene.emplace<arc::ecs::persistent_id_component>(terrain_entity);
    persistent.value = guid;
    auto terrain = make_test_terrain();
    terrain.material = {.index = 9u, .generation = 2u};
    terrain.cast_shadows = false;
    terrain.receive_shadows = true;
    scene.emplace<arc::scene::terrain_component>(terrain_entity, terrain);

    const auto result = arc::scene::render_scene(
        scene, renderer, 1280u, 720u, arc::render::render_mode::shaded, arc::render::mesh_visualization_mode::standard,
        arc::render::editor_overlay_mode::selected_wireframe, true, {}, 0.0f, {}, {}, &terrain_proxies);
    REQUIRE(result.camera_found);
    REQUIRE(result.terrain_count == 1u);
    REQUIRE(result.renderable_count == 1u);
    REQUIRE(result.selected_count == 1u);

    const auto* proxy = terrain_proxies.find(guid);
    REQUIRE(proxy != nullptr);
    REQUIRE(proxy->geometry.valid());
    REQUIRE(proxy->surface_attribute_texture.valid());
    REQUIRE(renderer.texture_alive(proxy->surface_attribute_texture));
    REQUIRE_FALSE(renderer.terrain_alive(proxy->handle));

    const auto frame = renderer.frame_queue().commit(1u);
    const auto world_event = std::find_if(frame.events.begin(), frame.events.end(), [](const auto& event)
                                          { return event.type() == arc::render::render_event_type::render_world; });
    REQUIRE(world_event != frame.events.end());
    const auto& world = *std::get<arc::render::render_world_event>(world_event->payload).packet;
    REQUIRE(world.terrains.empty());
    REQUIRE(world.items.size() == 1u);
    REQUIRE(world.virtual_items.empty());

    const auto& item = world.items.front();
    REQUIRE(renderer.mesh_alive(item.mesh));
    CHECK(item.material == terrain.material);
    CHECK(item.material_attribute_texture == proxy->surface_attribute_texture);
    REQUIRE(renderer.texture_alive(item.material_attribute_texture));
    CHECK(item.selected);
    CHECK_FALSE(item.casts_shadows);
    CHECK(item.receives_shadows);

    const auto lod_end = proxy->geometry.conventional_lods.begin() + proxy->geometry.conventional_lod_count;
    CHECK(std::find(proxy->geometry.conventional_lods.begin(), lod_end, item.mesh) != lod_end);
}

TEST_CASE("render scene submits terrain through generic virtual geometry when available")
{
    arc::ecs::world scene;
    arc::render::renderer renderer({.quality = arc::render::render_quality_tier::ultra});
    renderer.set_backend(std::make_unique<virtual_geometry_test_backend>());
    REQUIRE(renderer.resolved_config().features.virtual_geometry);
    arc::scene::terrain_render_proxy_cache terrain_proxies;

    const auto camera = scene.create();
    arc::scene::transform_component camera_transform;
    camera_transform.position = {0.0f, 0.0f, 6.0f};
    scene.emplace<arc::scene::transform_component>(camera, camera_transform);
    scene.emplace<arc::scene::camera_component>(camera);

    const auto terrain_entity = scene.create();
    scene.emplace<arc::scene::transform_component>(terrain_entity);
    scene.emplace<arc::scene::selection_component>(terrain_entity, true);
    const auto guid = arc::ecs::generate_entity_guid();
    auto& persistent = scene.emplace<arc::ecs::persistent_id_component>(terrain_entity);
    persistent.value = guid;
    auto terrain = make_test_terrain();
    terrain.material = {.index = 12u, .generation = 4u};
    terrain.cast_shadows = true;
    terrain.receive_shadows = false;
    scene.emplace<arc::scene::terrain_component>(terrain_entity, terrain);

    const auto result = arc::scene::render_scene(
        scene, renderer, 1280u, 720u, arc::render::render_mode::shaded, arc::render::mesh_visualization_mode::standard,
        arc::render::editor_overlay_mode::selected_wireframe, true, {}, 0.0f, {}, {}, &terrain_proxies);
    REQUIRE(result.camera_found);
    REQUIRE(result.terrain_count == 1u);
    REQUIRE(result.renderable_count == 1u);
    REQUIRE(result.selected_count == 1u);

    const auto* proxy = terrain_proxies.find(guid);
    REQUIRE(proxy != nullptr);
    REQUIRE(renderer.virtual_mesh_alive(proxy->geometry.virtualized));
    REQUIRE(renderer.texture_alive(proxy->surface_attribute_texture));

    const auto frame = renderer.frame_queue().commit(2u);
    const auto world_event = std::find_if(frame.events.begin(), frame.events.end(), [](const auto& event)
                                          { return event.type() == arc::render::render_event_type::render_world; });
    REQUIRE(world_event != frame.events.end());
    const auto& world = *std::get<arc::render::render_world_event>(world_event->payload).packet;
    REQUIRE(world.terrains.empty());
    REQUIRE(world.items.empty());
    REQUIRE(world.virtual_items.size() == 1u);

    const auto& item = world.virtual_items.front();
    CHECK(item.mesh == proxy->geometry.virtualized);
    CHECK(item.material == terrain.material);
    CHECK(item.material_attribute_texture == proxy->surface_attribute_texture);
    REQUIRE(item.gpu_scene_instance.valid());
    CHECK(item.selected);
    CHECK(item.casts_shadows);
    CHECK_FALSE(item.receives_shadows);
}
