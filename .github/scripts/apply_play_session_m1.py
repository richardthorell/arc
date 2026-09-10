from pathlib import Path


def replace_once(text: str, old: str, new: str, label: str) -> str:
    if old not in text:
        raise SystemExit(f"{label}: expected source block not found")
    return text.replace(old, new, 1)


def replace_between(text: str, start: str, end: str, replacement: str, label: str) -> str:
    begin = text.find(start)
    if begin < 0:
        raise SystemExit(f"{label}: start marker not found")
    finish = text.find(end, begin)
    if finish < 0:
        raise SystemExit(f"{label}: end marker not found")
    return text[:begin] + replacement + text[finish:]


# Allow a runtime world to safely release a borrowed ECS world at a phase boundary.
world_h = Path("engine/framework/inc/arc/framework/runtime_world.h")
text = world_h.read_text()
text = replace_once(
    text,
    "    bool attach_entities(ecs::world& entities) noexcept;\n",
    "    bool attach_entities(ecs::world& entities) noexcept;\n    bool detach_entities() noexcept;\n",
    "runtime world detach declaration",
)
world_h.write_text(text)

world_cpp = Path("engine/framework/src/common/runtime_world.cpp")
text = world_cpp.read_text()
attach_end = """bool runtime_world::attach_entities(ecs::world& entities) noexcept
{
    if (state_ == runtime_world_state::starting || state_ == runtime_world_state::running ||
        state_ == runtime_world_state::stopping)
        return false;
    entities_ = &entities;
    ++epoch_;
    return true;
}
"""
detach = attach_end + """
bool runtime_world::detach_entities() noexcept
{
    if (state_ == runtime_world_state::starting || state_ == runtime_world_state::running ||
        state_ == runtime_world_state::stopping)
        return false;
    if (entities_ == &owned_entities_) return true;
    entities_ = &owned_entities_;
    ++epoch_;
    return true;
}
"""
text = replace_once(text, attach_end, detach, "runtime world detach implementation")
world_cpp.write_text(text)

host = Path("editor/native/src/arc_host_base.inc")
text = host.read_text()

# The authoring ECS must never remain borrowed by the runtime world. A play session
# explicitly attaches a cloned ECS world for its lifetime instead.
legacy_attach = """        simulation.pause();
        const auto worlds = simulation.worlds().ordered_worlds();
        if (!worlds.empty())
        {
            if (framework::runtime_world* world = simulation.worlds().find(worlds.front()))
                world->attach_entities(scene.scene);
        }
"""
text = replace_once(text, legacy_attach, "        simulation.pause();\n", "remove persistent authoring attachment")

play_helpers = r'''    framework::runtime_world* preview_runtime_world() noexcept
    {
        const auto worlds = simulation.worlds().ordered_worlds();
        return worlds.empty() ? nullptr : simulation.worlds().find(worlds.front());
    }

    bool begin_play_session()
    {
        if (play_world)
        {
            simulation.resume();
            preview_stopped = false;
            return true;
        }

        simulation.pause();
        play_world.emplace(scene.scene);
        play_source_scene_revision = scene_revision;

        std::optional<scene::camera_component> editor_camera;
        std::optional<scene::transform_component> editor_camera_transform;
        if (const auto* camera = std::as_const(scene.scene).try_get<scene::camera_component>(scene.camera_entity))
            editor_camera = *camera;
        if (const auto* transform =
                std::as_const(scene.scene).try_get<scene::transform_component>(scene.camera_entity))
            editor_camera_transform = *transform;

        // The viewport camera is editor-only. Never let game systems see or mutate it.
        if (play_world->alive(scene.camera_entity)) play_world->destroy(scene.camera_entity);
        for (const auto entity : play_world->entities())
        {
            if (auto* selection = play_world->try_get<scene::selection_component>(entity)) selection->selected = false;
        }

        // Prefer the first active authored game camera. If a scene has no game
        // camera yet, clone the current editor camera into a transient play-only camera.
        play_camera_entity = {};
        for (const auto entity : play_world->entities())
        {
            const auto* camera = std::as_const(*play_world).try_get<scene::camera_component>(entity);
            const auto* active = std::as_const(*play_world).try_get<scene::active_component>(entity);
            if (camera && camera->active && (!active || active->active))
            {
                play_camera_entity = entity;
                break;
            }
        }
        if (!play_camera_entity.valid() && editor_camera && editor_camera_transform)
        {
            play_camera_entity = play_world->create();
            play_world->emplace<scene::name_component>(play_camera_entity, "Play Preview Camera");
            play_world->emplace<scene::camera_component>(play_camera_entity, *editor_camera);
            play_world->emplace<scene::transform_component>(play_camera_entity, *editor_camera_transform);
            play_world->emplace<scene::active_component>(play_camera_entity, true);
        }
        scene::update_world_transforms(*play_world);

        auto* runtime_world = preview_runtime_world();
        if (!runtime_world || !runtime_world->attach_entities(*play_world))
        {
            play_world.reset();
            play_camera_entity = {};
            play_source_scene_revision = 0;
            return false;
        }

        pending_pick.reset();
        preview_stopped = false;
        simulation.resume();
        return true;
    }

    bool end_play_session() noexcept
    {
        if (!play_world)
        {
            preview_stopped = true;
            return true;
        }

        simulation.pause();
        auto* runtime_world = preview_runtime_world();
        if (!runtime_world || !runtime_world->detach_entities()) return false;

        play_world.reset();
        play_camera_entity = {};
        play_source_scene_revision = 0;
        pending_pick.reset();
        preview_stopped = true;
        return true;
    }

    ecs::world& viewport_scene() noexcept
    {
        return play_world ? *play_world : scene.scene;
    }

    ecs::entity viewport_camera_entity() const noexcept
    {
        return play_world ? play_camera_entity : scene.camera_entity;
    }

'''
text = replace_once(
    text,
    "    std::unique_ptr<render::renderer> renderer;\n",
    play_helpers + "    std::unique_ptr<render::renderer> renderer;\n",
    "play session helper insertion",
)
text = replace_once(
    text,
    "    bool preview_stopped{true};\n",
    "    bool preview_stopped{true};\n"
    "    std::optional<ecs::world> play_world;\n"
    "    ecs::entity play_camera_entity{};\n"
    "    std::uint64_t play_source_scene_revision{};\n",
    "play session state",
)
text = replace_once(
    text,
    "    if (state_)\n    {\n        state_->project_module.unload();\n",
    "    if (state_)\n    {\n        (void)state_->end_play_session();\n        state_->project_module.unload();\n",
    "host shutdown play cleanup",
)
text = replace_once(
    text,
    "{\n    state_->project_module.unload();\n    if (!command.read_only && !command.editor_module_path.empty())\n",
    "{\n    if (!state_->end_play_session())\n"
    "        return {.request_id = request_id, .succeeded = false, .error = \"Could not end the active play session\"};\n"
    "    state_->project_module.unload();\n    if (!command.read_only && !command.editor_module_path.empty())\n",
    "project open play cleanup",
)

execute_marker = """    const bool project_mutation = std::visit(
        [](const auto& payload) { return is_project_mutation<std::decay_t<decltype(payload)>>(); }, command.payload);
"""
execute_guard = execute_marker + """    const bool replaces_authoring_world = std::holds_alternative<host_open_scene_command>(command.payload) ||
                                           std::holds_alternative<host_new_scene_command>(command.payload) ||
                                           std::holds_alternative<host_close_project_command>(command.payload);
    if (!state_->preview_stopped && replaces_authoring_world && !state_->end_play_session())
        return {.request_id = command.request_id,
                .succeeded = false,
                .error = "Could not end the active play session",
                .scene_revision = state_->scene_revision,
                .world_epoch = state_->world_epoch,
                .frame_revision = state_->viewport_frame_index};
"""
text = replace_once(text, execute_marker, execute_guard, "authoring world replacement guard")

runtime_start = "            else if constexpr (std::is_same_v<command_type, host_runtime_resume_command>)\n"
runtime_end = "            else if constexpr (std::is_same_v<command_type, host_runtime_set_time_scale_command>)\n"
runtime_block = r'''            else if constexpr (std::is_same_v<command_type, host_runtime_resume_command>)
            {
                const bool starting = state_->preview_stopped;
                if (starting)
                {
                    if (!state_->begin_play_session()) return fail("Could not create the play session world");
                }
                else
                {
                    state_->simulation.resume();
                }
                ++state_->runtime_revision;
                const auto snapshot = runtime_snapshot();
                push_event(state_->events, state_->event_sequence, host_event_type::runtime_state_changed,
                           starting ? "Play session started" : "Play session resumed", {}, to_json(snapshot));
                return success(to_json(snapshot));
            }
            else if constexpr (std::is_same_v<command_type, host_runtime_pause_command>)
            {
                if (state_->preview_stopped) return fail("No play session is active");
                state_->simulation.pause();
                ++state_->runtime_revision;
                const auto snapshot = runtime_snapshot();
                push_event(state_->events, state_->event_sequence, host_event_type::runtime_state_changed,
                           "Play session paused", {}, to_json(snapshot));
                return success(to_json(snapshot));
            }
            else if constexpr (std::is_same_v<command_type, host_runtime_stop_command>)
            {
                if (!state_->end_play_session()) return fail("Could not stop the play session");
                ++state_->runtime_revision;
                const auto snapshot = runtime_snapshot();
                push_event(state_->events, state_->event_sequence, host_event_type::runtime_state_changed,
                           "Play session stopped", {}, to_json(snapshot));
                return success(to_json(snapshot));
            }
            else if constexpr (std::is_same_v<command_type, host_runtime_step_command>)
            {
                if (state_->preview_stopped) return fail("No play session is active");
                if (!state_->simulation.paused()) return fail("Pause the play session before stepping");
                if (!state_->simulation.step(payload.ticks)) return fail("Play session could not queue a fixed-step");
                const framework::frame_time stepped = state_->simulation.advance(0.0);
                ++state_->runtime_revision;
                const auto snapshot = runtime_snapshot();
                push_event(state_->events, state_->event_sequence, host_event_type::runtime_tick_completed,
                           "Play session stepped " + std::to_string(stepped.completed_ticks) + " tick(s)", {},
                           to_json(snapshot));
                return success(to_json(snapshot));
            }
'''
text = replace_between(text, runtime_start, runtime_end, runtime_block, "runtime play lifecycle")

capture_start = "            else if constexpr (std::is_same_v<command_type, host_runtime_capture_snapshot_command>)\n"
capture_end = "            else if constexpr (std::is_same_v<command_type, host_open_scene_command>)\n"
capture_block = r'''            else if constexpr (std::is_same_v<command_type, host_runtime_capture_snapshot_command>)
            {
                auto* runtime_world = state_->preview_runtime_world();
                if (!runtime_world) return fail("Preview runtime has no world to snapshot");
                const bool authoring_checkpoint = state_->preview_stopped;
                if (authoring_checkpoint && !runtime_world->attach_entities(state_->scene.scene))
                    return fail("Authoring world could not be attached for checkpoint capture");
                const framework::world_snapshot_result captured =
                    state_->simulation.capture_snapshot(runtime_world->id(), payload.label);
                if (authoring_checkpoint) (void)runtime_world->detach_entities();
                if (!captured.succeeded) return fail(captured.error);
                return success("{\"snapshotId\":" + std::to_string(captured.metadata.id.value) +
                               ",\"tickId\":" + std::to_string(captured.metadata.tick.value) + '}');
            }
            else if constexpr (std::is_same_v<command_type, host_runtime_restore_snapshot_command>)
            {
                auto* runtime_world = state_->preview_runtime_world();
                if (!runtime_world) return fail("Preview runtime has no world to restore");
                const bool authoring_checkpoint = state_->preview_stopped;
                if (authoring_checkpoint && !runtime_world->attach_entities(state_->scene.scene))
                    return fail("Authoring world could not be attached for checkpoint restore");
                const framework::world_snapshot_result restored =
                    state_->simulation.restore_snapshot({payload.snapshot_id});
                if (authoring_checkpoint) (void)runtime_world->detach_entities();
                if (!restored.succeeded) return fail(restored.error);
                if (authoring_checkpoint)
                {
                    if (state_->scene.selected_entity.valid() &&
                        !state_->scene.scene.alive(state_->scene.selected_entity))
                        clear_selection(state_->scene.scene, state_->scene.selected_entity);
                    synchronize_all_terrain_resources(state_->scene, *state_->renderer);
                    if (auto* camera_transform =
                            state_->scene.scene.try_get<scene::transform_component>(state_->scene.camera_entity))
                        state_->camera_controller.synchronize_from(*camera_transform);
                    push_event(state_->events, state_->event_sequence, host_event_type::scene_changed,
                               "Authoring checkpoint restored", state_->scene.selected_entity);
                }
                else if (state_->play_world)
                {
                    scene::update_world_transforms(*state_->play_world);
                }
                ++state_->runtime_revision;
                const auto snapshot = runtime_snapshot();
                push_event(state_->events, state_->event_sequence, host_event_type::runtime_state_changed,
                           authoring_checkpoint ? "Authoring checkpoint restored" : "Play checkpoint restored", {},
                           to_json(snapshot));
                return success(to_json(snapshot));
            }
'''
text = replace_between(text, capture_start, capture_end, capture_block, "runtime checkpoints")

# While Play owns the viewport, editor manipulation is inert. M3 will route these
# events into the game input system instead.
text = replace_once(
    text,
    """            else if constexpr (std::is_same_v<command_type, host_viewport_camera_input_command>)
            {
                if (payload.viewport_id != state_->active_viewport_id) return fail("Viewport is not attached");
""",
    """            else if constexpr (std::is_same_v<command_type, host_viewport_camera_input_command>)
            {
                if (payload.viewport_id != state_->active_viewport_id) return fail("Viewport is not attached");
                if (!state_->preview_stopped) return success("{\"playSession\":true}");
""",
    "play camera input isolation",
)
text = replace_once(
    text,
    """            else if constexpr (std::is_same_v<command_type, host_viewport_set_pose_command>)
            {
""",
    """            else if constexpr (std::is_same_v<command_type, host_viewport_set_pose_command>)
            {
                if (!state_->preview_stopped) return success("{\"playSession\":true}");
""",
    "play camera pose isolation",
)
text = replace_once(
    text,
    """            else if constexpr (std::is_same_v<command_type, host_viewport_pick_command>)
            {
                if (payload.viewport_id != state_->active_viewport_id) return fail("Viewport is not attached");
""",
    """            else if constexpr (std::is_same_v<command_type, host_viewport_pick_command>)
            {
                if (payload.viewport_id != state_->active_viewport_id) return fail("Viewport is not attached");
                if (!state_->preview_stopped)
                    return success("{\"pending\":false,\"playSession\":true}");
""",
    "play picking isolation",
)

text = replace_once(text, "    if (state_->pending_pick)\n", "    if (state_->preview_stopped && state_->pending_pick)\n", "pending pick gate")
text = replace_once(
    text,
    "    if (state_->scene.focus_imported_scene_requested)\n",
    "    if (state_->preview_stopped && state_->scene.focus_imported_scene_requested)\n",
    "focus gate",
)

overlay_start = "    auto debug_overlay = build_editor_gizmo_overlay(\n"
overlay_end = "    const auto append_skeleton = [&](ecs::entity entity, std::uint32_t selected_joint)\n"
overlay_block = r'''    auto debug_overlay = state_->preview_stopped
                             ? build_editor_gizmo_overlay(
                                   state_->scene.scene, state_->scene.selected_entity, state_->scene.camera_entity,
                                   editor_gizmo_context{
                                       .tool = to_editor_tool(state_->viewport_tool.tool),
                                       .coordinate_space =
                                           state_->viewport_tool.coordinate_space == host_coordinate_space::local
                                               ? gizmo_coordinate_space::local
                                               : gizmo_coordinate_space::world,
                                       .highlighted_axis = state_->gizmo_highlight,
                                       .viewport_width = request.width,
                                       .viewport_height = request.height})
                             : render::debug_overlay_stream{};
'''
text = replace_between(text, overlay_start, overlay_end, overlay_block, "play overlay isolation")
text = replace_once(
    text,
    "    if (state_->viewport_options.skeletons)\n",
    "    if (state_->preview_stopped && state_->viewport_options.skeletons)\n",
    "skeleton overlay gate",
)
text = replace_once(
    text,
    "    else if (find_imported_skeleton(state_->scene, state_->scene.selected_entity))\n",
    "    else if (state_->preview_stopped && find_imported_skeleton(state_->scene, state_->scene.selected_entity))\n",
    "selected skeleton overlay gate",
)
text = replace_once(
    text,
    "    if (state_->viewport_options.grid)\n",
    "    if (state_->preview_stopped && state_->viewport_options.grid)\n",
    "grid overlay gate",
)
text = replace_once(
    text,
    "    if (state_->viewport_tool.tool == host_viewport_tool::terrain && state_->terrain_brush_local_position &&\n",
    "    if (state_->preview_stopped && state_->viewport_tool.tool == host_viewport_tool::terrain &&\n        state_->terrain_brush_local_position &&\n",
    "terrain overlay gate",
)

render_start = "    state_->scene.last_render = scene::render_scene(\n"
render_end = "\n    auto view_config = state_->renderer->resolved_config();\n"
render_block = r'''    ecs::world& viewport_scene = state_->viewport_scene();
    const ecs::entity viewport_camera = state_->viewport_camera_entity();
    state_->scene.last_render = scene::render_scene(
        viewport_scene, *state_->renderer, request.width, request.height,
        to_render_mode(state_->viewport_options.render_mode), to_visualization(state_->viewport_options.visualization),
        to_overlay(state_->viewport_options.overlay), state_->viewport_options.shadows,
        to_scene_visibility(state_->viewport_options.environment), delta_seconds, std::move(debug_overlay),
        viewport_camera, &state_->scene.terrain_render_proxies);
'''
text = replace_between(text, render_start, render_end, render_block, "play viewport scene selection")

aa_old = """    if (state_->viewport_options.anti_aliasing == host_camera_anti_aliasing::inherit)
    {
        if (const auto* camera = state_->scene.scene.try_get<scene::camera_component>(state_->scene.camera_entity);
            camera && camera->anti_aliasing != render::camera_anti_aliasing_override::inherit)
"""
aa_new = """    if (state_->viewport_options.anti_aliasing == host_camera_anti_aliasing::inherit)
    {
        if (const auto* camera = viewport_scene.try_get<scene::camera_component>(viewport_camera);
            camera && camera->anti_aliasing != render::camera_anti_aliasing_override::inherit)
"""
text = replace_once(text, aa_old, aa_new, "play camera anti-aliasing")

host.write_text(text)

# Focused coverage for the borrowed-world lifetime contract.
framework_tests = Path("engine/framework/tests/framework_tests.cpp")
text = framework_tests.read_text()
text += r'''

TEST_CASE("runtime worlds can detach an externally attached entity world")
{
    recording_application app;
    arc::framework::runtime host(app);
    auto& world = host.worlds().create({.name = "Detached world", .install_placeholder_systems = false});
    arc::ecs::world external;
    const auto external_entity = external.create();
    external.emplace<counter_component>(external_entity, counter_component{42});

    REQUIRE(world.attach_entities(external));
    REQUIRE(world.entities().alive(external_entity));
    REQUIRE(world.detach_entities());
    REQUIRE_FALSE(world.entities().alive(external_entity));
}
'''
framework_tests.write_text(text)

# Host-level regression proving the editor renders the play copy, preserves the
# authoring camera/selection, and returns to authoring state on Stop.
editor_tests = Path("editor/native/tests/editor_tests.cpp")
text = editor_tests.read_text()
text += r'''

TEST_CASE("editor play session renders an isolated scene copy and restores the authoring viewport")
{
    auto renderer = std::make_unique<arc::render::renderer>();
    arc::editor::arc_host_manager manager;
    auto host = manager.acquire(std::move(renderer));
    REQUIRE(host->open_project({.name = "Play Session Isolation", .root = {}}, {}).succeeded);
    host->renderer_service().set_backend(std::make_unique<pick_test_backend>());
    host->poll_events();

    const auto selected_before = host->selected_entity_snapshot();
    REQUIRE(selected_before.entity.valid());
    const auto editor_camera = host->scene_state().camera_entity;
    const auto camera_before =
        std::as_const(host->scene_state().scene).get<arc::scene::transform_component>(editor_camera);

    auto frame =
        host->request_viewport({.viewport_id = "viewport-1", .frame_index = 1, .width = 640, .height = 360});
    REQUIRE(frame.submitted);
    const auto initial_renderables = host->scene_state().last_render.renderable_count;

    REQUIRE(host->execute(arc::editor::host_runtime_resume_command{}).succeeded);
    REQUIRE(host->runtime_snapshot().state == arc::editor::host_runtime_state::running);

    auto& authoring = host->scene_state();
    const auto authoring_only = authoring.scene.create();
    authoring.scene.emplace<arc::scene::name_component>(authoring_only, "Authoring Only During Play");
    authoring.scene.emplace<arc::scene::transform_component>(authoring_only);
    arc::scene::mesh_renderer_component mesh;
    mesh.mesh = authoring.default_mesh;
    mesh.material = authoring.default_material;
    authoring.scene.emplace<arc::scene::mesh_renderer_component>(authoring_only, mesh);
    arc::scene::update_world_transforms(authoring.scene);

    REQUIRE(host->execute(arc::editor::host_viewport_camera_input_command{.forward = 1.0f}).succeeded);
    const auto camera_during_play =
        std::as_const(host->scene_state().scene).get<arc::scene::transform_component>(editor_camera);
    CHECK(camera_during_play.position == camera_before.position);

    frame = host->request_viewport({.viewport_id = "viewport-1", .frame_index = 2, .width = 640, .height = 360});
    REQUIRE(frame.submitted);
    CHECK(host->scene_state().last_render.renderable_count == initial_renderables);

    REQUIRE(host->execute(arc::editor::host_runtime_pause_command{}).succeeded);
    const auto tick_before_step = host->runtime_snapshot().tick_id;
    REQUIRE(host->execute(arc::editor::host_runtime_step_command{.ticks = 1}).succeeded);
    CHECK(host->runtime_snapshot().tick_id == tick_before_step + 1);

    REQUIRE(host->execute(arc::editor::host_runtime_stop_command{}).succeeded);
    REQUIRE(host->runtime_snapshot().state == arc::editor::host_runtime_state::stopped);
    CHECK(host->selected_entity_snapshot().guid == selected_before.guid);

    frame = host->request_viewport({.viewport_id = "viewport-1", .frame_index = 3, .width = 640, .height = 360});
    REQUIRE(frame.submitted);
    CHECK(host->scene_state().last_render.renderable_count == initial_renderables + 1);
}
'''
editor_tests.write_text(text)
