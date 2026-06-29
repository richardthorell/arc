#include <arc/scene/render_scene.h>

#include <arc/log.h>
#include <arc/scene/transforms.h>

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

    render::render_event_buffer buffer;
    render::render_event_writer writer(buffer);
    scene.view<transform_component, mesh_renderer_component>().each(
        [&](entity value, const transform_component& transform, const mesh_renderer_component& mesh_renderer) {
            if (!entity_is_active(scene, value))
                return;

            ++result.renderable_count;
            if (!mesh_renderer.visible || !mesh_renderer.mesh.valid())
                return;

            const bool selected = [&] {
                const auto* selection = scene.try_get<selection_component>(value);
                return selection && selection->selected;
            }();
            if (selected)
                ++result.selected_count;

            const math::matrix4f world = transform.dirty ? local_matrix(transform) : transform.world;
            writer.draw_mesh(
                mesh_renderer.mesh,
                mesh_renderer.material,
                world,
                vp,
                mode,
                visualization,
                overlay == render::editor_overlay_mode::all_wireframe ||
                    (overlay == render::editor_overlay_mode::selected_wireframe && selected),
                math::vector4f{ 0.28f, 0.62f, 1.0f, 1.0f },
                entity_label(scene, value));
            ++result.submitted_draw_count;
        });

    scene.view<transform_component, directional_light_component>().each(
        [&](entity value, const transform_component& transform, const directional_light_component& light) {
            if (!entity_is_active(scene, value))
                return;
            writer.directional_light(
                forward_direction(transform),
                light.color,
                light.intensity,
                light.casts_shadows,
                entity_label(scene, value));
            ++result.directional_light_count;
        });

    scene.view<transform_component, point_light_component>().each(
        [&](entity value, const transform_component& transform, const point_light_component& light) {
            if (!entity_is_active(scene, value))
                return;
            writer.point_light(
                transform.position,
                light.color,
                light.intensity,
                light.range,
                light.casts_shadows,
                entity_label(scene, value));
            ++result.point_light_count;
        });

    scene.view<transform_component, spot_light_component>().each(
        [&](entity value, const transform_component& transform, const spot_light_component& light) {
            if (!entity_is_active(scene, value))
                return;
            writer.spot_light(
                transform.position,
                forward_direction(transform),
                light.color,
                light.intensity,
                light.range,
                light.inner_angle,
                light.outer_angle,
                light.casts_shadows,
                entity_label(scene, value));
            ++result.spot_light_count;
        });

    renderer.frame_queue().submit(std::move(buffer));
    return result;
}

} // namespace arc::scene
