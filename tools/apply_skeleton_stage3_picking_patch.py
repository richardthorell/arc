#!/usr/bin/env python3
from pathlib import Path

def replace(path, old, new, label):
    p=Path(path); text=p.read_text()
    if old not in text: raise SystemExit(f'{label} marker not found in {path}')
    p.write_text(text.replace(old,new,1))

# Public hit-test API.
replace('editor/native/inc/arc/editor/editor_gizmo.h',
'''void append_editor_skeleton_overlay(render::debug_overlay_stream& stream, const ecs::world& registry,\n                                    ecs::entity entity, const render::skeleton_asset& skeleton,\n                                    ecs::entity camera_entity, std::uint32_t viewport_height,\n                                    std::uint32_t selected_joint = render::skeleton_asset::invalid_joint);\n''',
'''void append_editor_skeleton_overlay(render::debug_overlay_stream& stream, const ecs::world& registry,\n                                    ecs::entity entity, const render::skeleton_asset& skeleton,\n                                    ecs::entity camera_entity, std::uint32_t viewport_height,\n                                    std::uint32_t selected_joint = render::skeleton_asset::invalid_joint);\n\n/** Hit-test the editor-only skeleton overlay in output pixels. Returns invalid_joint on a miss. */\nstd::uint32_t hit_test_editor_skeleton_joint(const ecs::world& registry, ecs::entity entity,\n                                             const render::skeleton_asset& skeleton, ecs::entity camera_entity,\n                                             const editor_gizmo_context& context, float screen_x,\n                                             float screen_y) noexcept;\n''','header skeleton hit test')

# Add hit-test implementation before overlay implementation.
p=Path('editor/native/src/editor_gizmo.cpp'); text=p.read_text()
marker='''void append_editor_skeleton_overlay(render::debug_overlay_stream& stream, const ecs::world& registry,\n                                    ecs::entity entity, const render::skeleton_asset& skeleton,\n'''
if marker not in text: raise SystemExit('skeleton overlay impl marker not found')
impl=r'''std::uint32_t hit_test_editor_skeleton_joint(const ecs::world& registry, ecs::entity entity,
                                             const render::skeleton_asset& skeleton, ecs::entity camera_entity,
                                             const editor_gizmo_context& context, float screen_x,
                                             float screen_y) noexcept
{
    if (!skeleton.valid() || !registry.alive(entity) || context.viewport_width == 0u || context.viewport_height == 0u)
        return render::skeleton_asset::invalid_joint;
    const auto* entity_transform = registry.try_get<scene::transform_component>(entity);
    const auto* camera = registry.try_get<scene::camera_component>(camera_entity);
    const auto* camera_transform = registry.try_get<scene::transform_component>(camera_entity);
    if (!entity_transform || !camera || !camera_transform) return render::skeleton_asset::invalid_joint;

    const auto owner_world = entity_transform->dirty ? scene::local_matrix(*entity_transform) : entity_transform->world;
    std::vector<math::matrix4f> joint_world(skeleton.joints.size(), math::identity<float, 4>());
    std::vector<std::uint8_t> state(skeleton.joints.size());
    std::function<bool(std::size_t)> evaluate = [&](std::size_t index)
    {
        if (index >= skeleton.joints.size()) return false;
        if (state[index] == 2u) return true;
        if (state[index] == 1u) return false;
        state[index] = 1u;
        const auto& joint = skeleton.joints[index];
        scene::transform_component local;
        local.position = joint.bind_position;
        local.rotation = joint.bind_rotation;
        local.scale = joint.bind_scale;
        auto world = scene::local_matrix(local);
        if (joint.parent >= 0)
        {
            const auto parent = static_cast<std::size_t>(joint.parent);
            if (parent >= skeleton.joints.size() || !evaluate(parent)) return false;
            world = math::matmul(joint_world[parent], world);
        }
        joint_world[index] = world;
        state[index] = 2u;
        return true;
    };
    for (std::size_t joint = 0; joint < skeleton.joints.size(); ++joint)
        if (!evaluate(joint)) return render::skeleton_asset::invalid_joint;

    const auto view_projection = gizmo_view_projection(*camera, *camera_transform, context);
    const math::vector2f pointer{screen_x, screen_y};
    constexpr float joint_radius = 12.0f;
    constexpr float bone_radius = 8.0f;
    float nearest_joint = joint_radius;
    float nearest_bone = bone_radius;
    std::uint32_t joint_hit = render::skeleton_asset::invalid_joint;
    std::uint32_t bone_hit = render::skeleton_asset::invalid_joint;
    std::vector<math::vector2f> projected(skeleton.joints.size());
    std::vector<bool> visible(skeleton.joints.size());

    for (std::size_t joint = 0; joint < skeleton.joints.size(); ++joint)
    {
        const auto position = math::transform_point(
            owner_world, math::transform_point(joint_world[joint], math::vector3f::zero));
        visible[joint] = project_to_screen(view_projection, position, context.viewport_width, context.viewport_height,
                                           projected[joint]);
        if (!visible[joint]) continue;
        const float distance = math::length(math::sub(pointer, projected[joint]));
        if (distance <= nearest_joint)
        {
            nearest_joint = distance;
            joint_hit = static_cast<std::uint32_t>(joint);
        }
    }
    if (joint_hit != render::skeleton_asset::invalid_joint) return joint_hit;

    for (std::size_t joint = 0; joint < skeleton.joints.size(); ++joint)
    {
        const auto parent_value = skeleton.joints[joint].parent;
        if (parent_value < 0) continue;
        const auto parent = static_cast<std::size_t>(parent_value);
        if (parent >= skeleton.joints.size() || !visible[joint] || !visible[parent]) continue;
        const float distance = distance_to_segment(pointer, projected[parent], projected[joint]);
        if (distance <= nearest_bone)
        {
            nearest_bone = distance;
            bone_hit = static_cast<std::uint32_t>(joint);
        }
    }
    return bone_hit;
}

'''
p.write_text(text.replace(marker, impl+marker,1))

# Intercept ordinary viewport picking before ECS ObjectID picking when the selected entity owns a skeleton.
p=Path('editor/native/src/arc_host.cpp'); text=p.read_text()
marker='''    activate_viewport_surface(*state_, *viewport_surface);\n\n    const auto* material_command = std::get_if<host_set_entity_material_command>(&command.payload);\n'''
if marker not in text: raise SystemExit('arc host viewport activation marker not found')
addition=r'''    activate_viewport_surface(*state_, *viewport_surface);

    if (const auto* pick = std::get_if<host_viewport_pick_command>(&command.payload);
        pick && viewport_surface->preview_kind == asset_preview_kind::none &&
        state_->scene.scene.alive(state_->scene.selected_entity))
    {
        if (const auto* skeleton = find_imported_skeleton(state_->scene, state_->scene.selected_entity))
        {
            editor_gizmo_context context;
            context.viewport_width = viewport_surface->options.width;
            context.viewport_height = viewport_surface->options.height;
            const auto joint = hit_test_editor_skeleton_joint(
                state_->scene.scene, state_->scene.selected_entity, *skeleton, state_->scene.camera_entity, context,
                static_cast<float>(pick->x), static_cast<float>(pick->y));
            if (joint != render::skeleton_asset::invalid_joint)
            {
                state_->viewport_options.selected_skeleton_joint = joint;
                viewport_surface->options.selected_skeleton_joint = joint;
                push_event(state_->events, state_->event_sequence, host_event_type::entity_selected,
                           "Skeleton bone selected", state_->scene.selected_entity);
                host_response response{.request_id = command.request_id,
                                       .succeeded = true,
                                       .payload_json = "{\\\"jointIndex\\\":" + std::to_string(joint) + '}'};
                response.scene_revision = state_->scene_revision;
                response.world_epoch = state_->world_epoch;
                response.frame_revision = state_->viewport_frame_index;
                return response;
            }
        }
    }

    const auto* material_command = std::get_if<host_set_entity_material_command>(&command.payload);
'''
p.write_text(text.replace(marker,addition,1))

# Ensure a click (rather than a camera drag) sends the standard viewport.pick command on both transports.
p=Path('editor/src/renderer/src/viewport/ViewportPanel.tsx'); text=p.read_text()
text=text.replace('''  const dragRef = useRef<DragState | null>(null);\n''','''  const dragRef = useRef<DragState | null>(null);\n  const clickRef = useRef<{ pointerId: number; x: number; y: number } | null>(null);\n''',1)
text=text.replace('''    event.currentTarget.setPointerCapture(event.pointerId);\n    if (event.button === 2) {\n''','''    event.currentTarget.setPointerCapture(event.pointerId);\n    if (event.button === 0) clickRef.current = { pointerId: event.pointerId, x: event.clientX, y: event.clientY };\n    if (event.button === 2) {\n''',1)
text=text.replace('''  const onPointerMove = (event: PointerEvent<HTMLDivElement>) => {\n    if (streamedAvailable) {\n''','''  const onPointerMove = (event: PointerEvent<HTMLDivElement>) => {\n    const click = clickRef.current;\n    if (click?.pointerId === event.pointerId && Math.hypot(event.clientX - click.x, event.clientY - click.y) > 4)\n      clickRef.current = null;\n    if (streamedAvailable) {\n''',1)
text=text.replace('''  const onPointerUp = (event: PointerEvent<HTMLDivElement>) => {\n    if (streamedAvailable) sendPointer(event, 'up');\n    if (dragRef.current?.pointerId === event.pointerId) {\n''','''  const onPointerUp = (event: PointerEvent<HTMLDivElement>) => {\n    if (streamedAvailable) sendPointer(event, 'up');\n    const click = clickRef.current;\n    if (event.button === 0 && click?.pointerId === event.pointerId && viewportActive) {\n      const position = pointerCoordinates(event.clientX, event.clientY);\n      void window.arc.host.command('viewport.pick', { viewportId, ...position }).catch((error) => {\n        setViewportError(error instanceof Error ? error.message : String(error));\n      });\n    }\n    if (click?.pointerId === event.pointerId) clickRef.current = null;\n    if (dragRef.current?.pointerId === event.pointerId) {\n''',1)
p.write_text(text)
