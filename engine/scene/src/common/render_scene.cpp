#include <arc/scene/render_scene.h>

#include <arc/diagnostics/log.h>
#include <arc/render/render_world.h>
#include <arc/scene/transforms.h>

#include <memory>

namespace arc::scene
{
namespace
{

bool entity_is_active(const registry& scene, entity value)
{
    const auto* active = scene.try_get<active_component>(value);
    return !active || active->active;
}

std::string entity_label(const registry& scene, entity value)
{
    const auto* name = scene.try_get<name_component>(value);
    return name ? name->value : std::string{};
}

std::uint32_t render_layer_mask(const registry& scene, entity value)
{
    const auto* layer = scene.try_get<render_layer_component>(value);
    return layer ? layer->mask : 1u;
}

geometric::box3f world_bounds_for(const registry& scene, entity value, const transform_component& transform)
{
    const auto* bounds = scene.try_get<bounds_component>(value);
    if (bounds)
        return bounds->dirty ? geometric::box3f{
            geometric::point3f{ transform.position - math::vector3f{ 0.5f, 0.5f, 0.5f } },
            geometric::point3f{ transform.position + math::vector3f{ 0.5f, 0.5f, 0.5f } } } : bounds->world_bounds;
    return geometric::box3f{
        geometric::point3f{ transform.position - math::vector3f{ 0.5f, 0.5f, 0.5f } },
        geometric::point3f{ transform.position + math::vector3f{ 0.5f, 0.5f, 0.5f } }
    };
}

bool entity_selected(const registry& scene, entity value)
{
    const auto* selection = scene.try_get<selection_component>(value);
    return selection && selection->selected;
}

void append_mesh_item(
    registry& scene,
    render::render_world_packet& packet,
    render_scene_result& result,
    entity value,
    const transform_component& transform,
    render::mesh_handle mesh,
    render::material_handle material,
    bool visible,
    bool transparent = false,
    render::buffer_handle skin = {},
    std::uint32_t joint_count = 0,
    std::uint32_t instance_count = 1)
{
    if (!entity_is_active(scene, value))
        return;

    ++result.renderable_count;
    const bool selected = entity_selected(scene, value);
    if (selected)
        ++result.selected_count;

    const math::matrix4f world = transform.dirty ? local_matrix(transform) : transform.world;
    packet.items.push_back({
        .mesh = mesh,
        .material = material,
        .model = world,
        .world_bounds = world_bounds_for(scene, value, transform),
        .render_layer_mask = render_layer_mask(scene, value),
        .object_id = render::make_render_object_id(value.index, value.generation),
        .skin_matrices = skin,
        .skin_joint_count = joint_count,
        .instance_count = instance_count,
        .visible = visible,
        .selected = selected,
        .transparent = transparent,
        .label = entity_label(scene, value)
    });
}

void apply_lod(
    const registry& scene,
    entity value,
    render::mesh_handle& mesh,
    render::material_handle& material)
{
    const auto* lod = scene.try_get<lod_component>(value);
    if (!lod || !lod->enabled || lod->levels.empty())
        return;

    for (const auto& level : lod->levels)
    {
        if (level.mesh.valid())
        {
            mesh = level.mesh;
            material = level.material.valid() ? level.material : material;
            return;
        }
    }
}

} // namespace

render_scene_result render_scene(
    registry& scene,
    render::renderer& renderer,
    std::uint32_t viewport_width,
    std::uint32_t viewport_height,
    render::render_mode mode,
    render::mesh_visualization_mode visualization,
    render::editor_overlay_mode overlay)
{
    render_scene_result result{};

    const transform_component* camera_transform{};
    const camera_component* camera{};
    scene.view<transform_component, camera_component>().each(
        [&](entity value, const transform_component& transform, const camera_component& candidate) {
            if (!camera && candidate.active && entity_is_active(scene, value))
            {
                camera_transform = &transform;
                camera = &candidate;
            }
        });

    if (!camera || !camera_transform)
    {
        static bool warned{};
        if (!warned)
        {
            arc::warn("scene", "No active camera found for render extraction");
            warned = true;
        }
        return result;
    }

    result.camera_found = true;
    const float aspect = viewport_height == 0 ? 1.0f : static_cast<float>(viewport_width) / static_cast<float>(viewport_height);
    const math::matrix4f vp = view_projection(*camera, *camera_transform, aspect);

    render::render_world_packet world_packet;
    world_packet.camera.view_projection = vp;
    world_packet.camera.position = camera_transform->position;
    world_packet.camera.clear_color = camera->clear_color;
    world_packet.camera.near_plane = camera->near_plane;
    world_packet.camera.far_plane = camera->far_plane;
    world_packet.mode = mode;
    world_packet.visualization = visualization;
    world_packet.overlay = overlay;
    world_packet.viewport_width = viewport_width;
    world_packet.viewport_height = viewport_height;

    scene.view<transform_component, mesh_renderer_component>().each(
        [&](entity value, const transform_component& transform, const mesh_renderer_component& mesh_renderer) {
            auto mesh = mesh_renderer.mesh;
            auto material = mesh_renderer.material;
            apply_lod(scene, value, mesh, material);
            append_mesh_item(scene, world_packet, result, value, transform, mesh, material, mesh_renderer.visible);
        });

    scene.view<transform_component, skinned_mesh_renderer_component>().each(
        [&](entity value, const transform_component& transform, const skinned_mesh_renderer_component& mesh_renderer) {
            append_mesh_item(
                scene,
                world_packet,
                result,
                value,
                transform,
                mesh_renderer.mesh,
                mesh_renderer.material,
                mesh_renderer.visible,
                false,
                mesh_renderer.skin_matrices,
                mesh_renderer.joint_count);
        });

    scene.view<transform_component, instance_group_component>().each(
        [&](entity value, const transform_component& transform, const instance_group_component& instances) {
            append_mesh_item(
                scene,
                world_packet,
                result,
                value,
                transform,
                instances.mesh,
                instances.material,
                instances.visible,
                false,
                {},
                0,
                instances.instance_count);
        });

    scene.view<transform_component, directional_light_component>().each(
        [&](entity value, const transform_component& transform, const directional_light_component& light) {
            if (!entity_is_active(scene, value))
                return;
            world_packet.directional_lights.push_back({
                .direction = forward_direction(transform),
                .color = light.color,
                .intensity = light.intensity,
                .casts_shadows = light.casts_shadows,
                .label = entity_label(scene, value)
            });
            ++result.directional_light_count;
        });

    scene.view<transform_component, point_light_component>().each(
        [&](entity value, const transform_component& transform, const point_light_component& light) {
            if (!entity_is_active(scene, value))
                return;
            world_packet.point_lights.push_back({
                .position = transform.position,
                .color = light.color,
                .intensity = light.intensity,
                .range = light.range,
                .casts_shadows = light.casts_shadows,
                .label = entity_label(scene, value)
            });
            ++result.point_light_count;
        });

    scene.view<transform_component, spot_light_component>().each(
        [&](entity value, const transform_component& transform, const spot_light_component& light) {
            if (!entity_is_active(scene, value))
                return;
            world_packet.spot_lights.push_back({
                .position = transform.position,
                .direction = forward_direction(transform),
                .color = light.color,
                .intensity = light.intensity,
                .range = light.range,
                .inner_angle = light.inner_angle,
                .outer_angle = light.outer_angle,
                .casts_shadows = light.casts_shadows,
                .label = entity_label(scene, value)
            });
            ++result.spot_light_count;
        });

    render::prepare_render_world(world_packet);
    result.submitted_draw_count = world_packet.visible_items.size();
    result.culled_count = world_packet.culled_item_count;
    result.instance_batch_count = world_packet.instance_batches.size();
    result.indirect_draw_count = world_packet.indirect_draws.size();

    render::render_event_buffer buffer;
    render::render_event_writer writer(buffer);
    writer.render_world(std::make_shared<render::render_world_packet>(std::move(world_packet)), "scene");
    renderer.frame_queue().submit(std::move(buffer));
    return result;
}

} // namespace arc::scene
