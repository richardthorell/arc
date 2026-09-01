from pathlib import Path


def replace_once(text: str, old: str, new: str, label: str) -> str:
    if old not in text:
        raise SystemExit(f"missing anchor: {label}")
    return text.replace(old, new, 1)


p = Path("editor/native/src/arc_host_process_main.cpp")
s = p.read_text()

s = replace_once(
    s,
    '''        {
            std::lock_guard lock(host_mutex_);
            host_->request_viewport(arc::editor::host_viewport_request{.viewport_id = viewport_id_,
                                                                       .frame_index = frame_index_++,
                                                                       .width = value.width,
                                                                       .height = value.height});
        }
''',
    '''        const auto host_wait_start = std::chrono::steady_clock::now();
        std::unique_lock host_lock(host_mutex_);
        const auto host_lock_acquired = std::chrono::steady_clock::now();
        const auto request_start = host_lock_acquired;
        host_->request_viewport(arc::editor::host_viewport_request{.viewport_id = viewport_id_,
                                                                   .frame_index = frame_index_++,
                                                                   .width = value.width,
                                                                   .height = value.height});
        const auto request_end = std::chrono::steady_clock::now();
        host_lock.unlock();
        const auto host_wait_ms = std::chrono::duration<double, std::milli>(host_lock_acquired - host_wait_start).count();
        const auto request_ms = std::chrono::duration<double, std::milli>(request_end - request_start).count();
        if (host_wait_ms >= 5.0 || request_ms >= 5.0)
            std::cerr << "[perf][host.viewport] " << viewport_id_ << " mutex-wait=" << std::fixed
                      << std::setprecision(1) << host_wait_ms << "ms request_viewport=" << request_ms << "ms\\n";
''',
    "native render_once viewport timing",
)

s = replace_once(
    s,
    '''        {
            std::lock_guard lock(host_mutex_);
            host_->request_viewport(arc::editor::host_viewport_request{.viewport_id = target.viewport_id,
                                                                       .frame_index = target.frame_index,
                                                                       .width = target.width,
                                                                       .height = target.height});
        }
''',
    '''        const auto host_wait_start = std::chrono::steady_clock::now();
        std::unique_lock host_lock(host_mutex_);
        const auto host_lock_acquired = std::chrono::steady_clock::now();
        const auto request_start = host_lock_acquired;
        host_->request_viewport(arc::editor::host_viewport_request{.viewport_id = target.viewport_id,
                                                                   .frame_index = target.frame_index,
                                                                   .width = target.width,
                                                                   .height = target.height});
        const auto request_end = std::chrono::steady_clock::now();
        host_lock.unlock();
        const auto host_wait_ms = std::chrono::duration<double, std::milli>(host_lock_acquired - host_wait_start).count();
        const auto request_ms = std::chrono::duration<double, std::milli>(request_end - request_start).count();
        if (host_wait_ms >= 5.0 || request_ms >= 5.0)
            std::cerr << "[perf][host.viewport] " << target.viewport_id << " mutex-wait=" << std::fixed
                      << std::setprecision(1) << host_wait_ms << "ms request_viewport=" << request_ms << "ms\\n";
''',
    "shared render viewport timing",
)

s = replace_once(
    s,
    '''            arc::editor::host_response response;
            {
                std::lock_guard lock(host_mutex);
                response = host->query(query);
            }
            write_response(response);
''',
    '''            arc::editor::host_response response;
            const auto lock_wait_start = std::chrono::steady_clock::now();
            std::unique_lock lock(host_mutex);
            const auto lock_acquired = std::chrono::steady_clock::now();
            const auto execute_start = lock_acquired;
            response = host->query(query);
            const auto execute_end = std::chrono::steady_clock::now();
            lock.unlock();
            const auto lock_wait_ms = std::chrono::duration<double, std::milli>(lock_acquired - lock_wait_start).count();
            const auto execute_ms = std::chrono::duration<double, std::milli>(execute_end - execute_start).count();
            if (lock_wait_ms >= 5.0 || execute_ms >= 5.0)
                std::cerr << "[perf][host.request] query " << query.query_type << " id=" << query.request_id
                          << " mutex-wait=" << std::fixed << std::setprecision(1) << lock_wait_ms
                          << "ms execute=" << execute_ms << "ms\\n";
            write_response(response);
''',
    "query mutex timing",
)

s = replace_once(
    s,
    '''                {
                    std::lock_guard lock(host_mutex);
                    response = host->execute(command);
                }
                if (response.succeeded)
''',
    '''                const auto lock_wait_start = std::chrono::steady_clock::now();
                std::unique_lock lock(host_mutex);
                const auto lock_acquired = std::chrono::steady_clock::now();
                const auto execute_start = lock_acquired;
                response = host->execute(command);
                const auto execute_end = std::chrono::steady_clock::now();
                lock.unlock();
                const auto lock_wait_ms = std::chrono::duration<double, std::milli>(lock_acquired - lock_wait_start).count();
                const auto execute_ms = std::chrono::duration<double, std::milli>(execute_end - execute_start).count();
                if (lock_wait_ms >= 5.0 || execute_ms >= 5.0)
                    std::cerr << "[perf][host.request] command " << command.command_type << " id=" << command.request_id
                              << " mutex-wait=" << std::fixed << std::setprecision(1) << lock_wait_ms
                              << "ms execute=" << execute_ms << "ms\\n";
                if (response.succeeded)
''',
    "shared viewport create command timing",
)

s = replace_once(
    s,
    '''            else
            {
                std::lock_guard lock(host_mutex);
                response = host->execute(command);
            }
            write_response(response);
''',
    '''            else
            {
                const auto lock_wait_start = std::chrono::steady_clock::now();
                std::unique_lock lock(host_mutex);
                const auto lock_acquired = std::chrono::steady_clock::now();
                const auto execute_start = lock_acquired;
                response = host->execute(command);
                const auto execute_end = std::chrono::steady_clock::now();
                lock.unlock();
                const auto lock_wait_ms = std::chrono::duration<double, std::milli>(lock_acquired - lock_wait_start).count();
                const auto execute_ms = std::chrono::duration<double, std::milli>(execute_end - execute_start).count();
                if (lock_wait_ms >= 5.0 || execute_ms >= 5.0)
                    std::cerr << "[perf][host.request] command " << command.command_type << " id=" << command.request_id
                              << " mutex-wait=" << std::fixed << std::setprecision(1) << lock_wait_ms
                              << "ms execute=" << execute_ms << "ms\\n";
            }
            write_response(response);
''',
    "command mutex timing",
)

p.write_text(s)

p = Path("editor/native/src/arc_host_base.inc")
s = p.read_text()

s = replace_once(
    s,
    '''    if (state_->asset_registry)
    {
        state_->asset_registry->poll();
''',
    '''    const auto asset_poll_start = std::chrono::steady_clock::now();
    if (state_->asset_registry)
    {
        state_->asset_registry->poll();
''',
    "viewport asset poll start",
)

s = replace_once(
    s,
    '''    const auto frame_start = std::chrono::steady_clock::now();
    float delta_seconds = 0.0f;
''',
    '''    const auto asset_poll_end = std::chrono::steady_clock::now();
    const auto asset_poll_ms =
        std::chrono::duration<double, std::milli>(asset_poll_end - asset_poll_start).count();
    if (asset_poll_ms >= 5.0)
        arc::diagnostics::info("editor.performance", "[perf][viewport.stage] asset-poll=" +
                                                        std::to_string(asset_poll_ms) + "ms");

    const auto frame_start = std::chrono::steady_clock::now();
    float delta_seconds = 0.0f;
''',
    "viewport asset poll end",
)

s = replace_once(
    s,
    '''    const framework::frame_time runtime_frame = state_->simulation.advance(delta_seconds);
''',
    '''    const auto simulation_start = std::chrono::steady_clock::now();
    const framework::frame_time runtime_frame = state_->simulation.advance(delta_seconds);
    const auto simulation_ms = std::chrono::duration<double, std::milli>(
                                   std::chrono::steady_clock::now() - simulation_start)
                                   .count();
    if (simulation_ms >= 5.0)
        arc::diagnostics::info("editor.performance", "[perf][viewport.stage] simulation=" +
                                                        std::to_string(simulation_ms) + "ms");
''',
    "viewport simulation timing",
)

s = replace_once(
    s,
    '''    const auto submit_result = state_->renderer->render_frame(
        request.frame_index,
        render::make_scene_draw_graph("viewport", view_config, true, state_->scene.last_render.environment));
    const auto frame_end = std::chrono::steady_clock::now();
''',
    '''    const auto graph_start = std::chrono::steady_clock::now();
    auto draw_graph = render::make_scene_draw_graph("viewport", view_config, true, state_->scene.last_render.environment);
    const auto graph_end = std::chrono::steady_clock::now();
    const auto render_start = graph_end;
    const auto submit_result = state_->renderer->render_frame(request.frame_index, std::move(draw_graph));
    const auto frame_end = std::chrono::steady_clock::now();
    const auto graph_ms = std::chrono::duration<double, std::milli>(graph_end - graph_start).count();
    const auto render_ms = std::chrono::duration<double, std::milli>(frame_end - render_start).count();
    if (graph_ms >= 5.0 || render_ms >= 5.0)
        arc::diagnostics::info("editor.performance", "[perf][viewport.stage] draw-graph=" +
                                                        std::to_string(graph_ms) + "ms render-frame=" +
                                                        std::to_string(render_ms) + "ms");
''',
    "viewport graph/render timing",
)

p.write_text(s)
