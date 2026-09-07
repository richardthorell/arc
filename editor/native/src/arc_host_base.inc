        focus_selected_entity(state_->scene.scene, state_->scene.selected_entity, state_->camera_controller);
        if (auto* camera_transform =
                state_->scene.scene.try_get<scene::transform_component>(state_->scene.camera_entity))
            state_->camera_controller.apply_to(*camera_transform);
        state_->scene.focus_imported_scene_requested = false;
    }

    auto debug_overlay = state_->scene.scene_name == "Asset Preview: model"
                             ? render::debug_overlay_stream{}
                             : build_editor_gizmo_overlay(
                                   state_->scene.scene, state_->scene.selected_entity, state_->scene.camera_entity,
                                   editor_gizmo_context{
                                       .tool = to_editor_tool(state_->viewport_tool.tool),
                                       .coordinate_space =
                                           state_->viewport_tool.coordinate_space == host_coordinate_space::local
                                               ? gizmo_coordinate_space::local
                                               : gizmo_coordinate_space::world,
                                       .highlighted_axis = state_->gizmo_highlight,
                                       .viewport_width = request.width,
                                       .viewport_height = request.height});
    const auto append_skeleton = [&](ecs::entity entity, std::uint32_t selected_joint)
    {
        if (const auto* skeleton = find_imported_skeleton(state_->scene, entity))
            append_editor_skeleton_overlay(debug_overlay, state_->scene.scene, entity, *skeleton,
                                           state_->scene.camera_entity, request.height, selected_joint);
    };
    if (state_->viewport_options.skeletons)