from pathlib import Path

p = Path('editor/native/src/arc_host_process_main.cpp')
s = p.read_text()
replacements = [
('''        if (renderer_running) return true;
        {
            std::lock_guard lock(setup_mutex_);
            setup_complete_ = false;
            setup_error_.clear();
        }
        if (!running_.exchange(true))
            render_task_ = jobs_->submit({.name = "editor.shared_viewport",
                                          .priority = arc::jobs::job_priority::critical,
                                          .affinity = arc::jobs::job_affinity::render_thread},
                                         [this] { render_loop(); });
        std::unique_lock lock(setup_mutex_);
        if (!setup_cv_.wait_for(lock, std::chrono::seconds(10), [this] { return setup_complete_; }))
            setup_error_ = "Timed out creating shared viewport renderer";
        error = setup_error_;
        lock.unlock();
        if (!error.empty() && render_task_.valid()) (void)render_task_.wait_result();
        return error.empty();
''','''        if (renderer_running) return true;
        {
            std::lock_guard lock(setup_mutex_);
            setup_complete_ = false;
            setup_error_.clear();
        }
        if (!running_.exchange(true))
            render_task_ = jobs_->submit({.name = "editor.shared_viewport",
                                          .priority = arc::jobs::job_priority::critical,
                                          .affinity = arc::jobs::job_affinity::render_thread},
                                         [this] { render_loop(); });
        // Do not block the stdin command loop while Vulkan starts on the render thread.
        error.clear();
        return true;
'''),
('''        {
            std::lock_guard lock(host_mutex_);
            host_->request_viewport(arc::editor::host_viewport_request{.viewport_id = viewport_id_,
                                                                       .frame_index = frame_index_++,
                                                                       .width = value.width,
                                                                       .height = value.height});
            auto present = backend_->present_surface_frame(value.width, value.height);
            rendered = present.has_value();
            if (!rendered) message = std::move(present.error().message);
        }
''','''        {
            std::lock_guard lock(host_mutex_);
            host_->request_viewport(arc::editor::host_viewport_request{.viewport_id = viewport_id_,
                                                                       .frame_index = frame_index_++,
                                                                       .width = value.width,
                                                                       .height = value.height});
        }
        auto present = backend_->present_surface_frame(value.width, value.height);
        rendered = present.has_value();
        if (!rendered) message = std::move(present.error().message);
'''),
('''        {
            std::lock_guard lock(host_mutex_);
            host_->request_viewport(arc::editor::host_viewport_request{.viewport_id = target.viewport_id,
                                                                       .frame_index = target.frame_index,
                                                                       .width = target.width,
                                                                       .height = target.height});
            auto present = backend_->present_viewport_output(target.viewport_id);
            rendered = present.has_value();
            if (!rendered) message = std::move(present.error().message);
        }
''','''        {
            std::lock_guard lock(host_mutex_);
            host_->request_viewport(arc::editor::host_viewport_request{.viewport_id = target.viewport_id,
                                                                       .frame_index = target.frame_index,
                                                                       .width = target.width,
                                                                       .height = target.height});
        }
        auto present = backend_->present_viewport_output(target.viewport_id);
        rendered = present.has_value();
        if (!rendered) message = std::move(present.error().message);
'''),
('''        auto result = arc::render::vulkan::create_vulkan_backend(config);
        if (!result)
''','''        const auto backend_start = std::chrono::steady_clock::now();
        auto result = arc::render::vulkan::create_vulkan_backend(config);
        const auto backend_ms = std::chrono::duration<double, std::milli>(
                                    std::chrono::steady_clock::now() - backend_start)
                                    .count();
        std::cerr << "[perf][render.vulkan] backend creation " << std::fixed << std::setprecision(1) << backend_ms
                  << "ms\\n";
        if (!result)
''')]
for old,new in replacements:
    if old not in s: raise SystemExit('native replacement anchor not found')
    s=s.replace(old,new,1)
p.write_text(s)

p = Path('editor/src/main/main.ts')
s = p.read_text()
old='''  private pendingRuntimeTick: HostEvent | null = null;
  private runtimeTickScheduled = false;
  private readonly eventListeners = new Set<(event: HostEvent) => void>();
'''
new='''  private pendingRuntimeTick: HostEvent | null = null;
  private runtimeTickScheduled = false;
  private readonly coalescedQueries = new Map<string, Promise<HostResponse>>();
  private pointerMoveInFlight = new Map<string, Promise<HostResponse>>();
  private pendingPointerMove = new Map<
    string,
    { payload: Record<string, unknown>; waiters: Array<{ resolve: (value: HostResponse) => void; reject: (error: unknown) => void }> }
  >();
  private thumbnailActive = 0;
  private readonly thumbnailQueue: Array<() => void> = [];
  private static readonly maxConcurrentThumbnails = 2;
  private readonly eventListeners = new Set<(event: HostEvent) => void>();
'''
if old not in s: raise SystemExit('main fields anchor missing')
s=s.replace(old,new,1)
old='''  command(
    type: string,
    payload: Record<string, unknown> = {},
    edit?: Record<string, unknown>,
    expectedSceneRevision?: number,
  ): Promise<HostResponse> {
    return this.send({ kind: 'command', type, payload, edit, expectedSceneRevision });
  }

  query(type: string, payload: Record<string, unknown> = {}): Promise<HostResponse> {
    return this.send({ kind: 'query', type, payload });
  }
'''
new='''  command(
    type: string,
    payload: Record<string, unknown> = {},
    edit?: Record<string, unknown>,
    expectedSceneRevision?: number,
  ): Promise<HostResponse> {
    if (type === 'viewport.pointer' && payload.phase === 'move') return this.sendPointerMove(payload);
    return this.send({ kind: 'command', type, payload, edit, expectedSceneRevision });
  }

  query(type: string, payload: Record<string, unknown> = {}): Promise<HostResponse> {
    if (type === 'viewport.state') {
      const key = `${type}:${JSON.stringify(payload)}`;
      const current = this.coalescedQueries.get(key);
      if (current) return current;
      const request = this.send({ kind: 'query', type, payload });
      this.coalescedQueries.set(key, request);
      void request.finally(() => {
        if (this.coalescedQueries.get(key) === request) this.coalescedQueries.delete(key);
      });
      return request;
    }
    if (type === 'asset.thumbnail') return this.sendThumbnailQuery(type, payload);
    return this.send({ kind: 'query', type, payload });
  }

  private sendPointerMove(payload: Record<string, unknown>): Promise<HostResponse> {
    const viewportId = typeof payload.viewportId === 'string' ? payload.viewportId : 'viewport-1';
    if (!this.pointerMoveInFlight.has(viewportId)) {
      const request = this.send({ kind: 'command', type: 'viewport.pointer', payload });
      this.pointerMoveInFlight.set(viewportId, request);
      void request.finally(() => this.flushPointerMove(viewportId));
      return request;
    }
    return new Promise((resolve, reject) => {
      const pending = this.pendingPointerMove.get(viewportId);
      if (pending) {
        pending.payload = payload;
        pending.waiters.push({ resolve, reject });
      } else {
        this.pendingPointerMove.set(viewportId, { payload, waiters: [{ resolve, reject }] });
      }
    });
  }

  private flushPointerMove(viewportId: string): void {
    this.pointerMoveInFlight.delete(viewportId);
    const pending = this.pendingPointerMove.get(viewportId);
    if (!pending) return;
    this.pendingPointerMove.delete(viewportId);
    const request = this.send({ kind: 'command', type: 'viewport.pointer', payload: pending.payload });
    this.pointerMoveInFlight.set(viewportId, request);
    void request
      .then(
        (response) => pending.waiters.forEach((waiter) => waiter.resolve(response)),
        (error) => pending.waiters.forEach((waiter) => waiter.reject(error)),
      )
      .finally(() => this.flushPointerMove(viewportId));
  }

  private sendThumbnailQuery(type: string, payload: Record<string, unknown>): Promise<HostResponse> {
    return new Promise((resolve, reject) => {
      const run = () => {
        this.thumbnailActive += 1;
        void this.send({ kind: 'query', type, payload })
          .then(resolve, reject)
          .finally(() => {
            this.thumbnailActive -= 1;
            this.thumbnailQueue.shift()?.();
          });
      };
      if (this.thumbnailActive < ArcHostClient.maxConcurrentThumbnails) run();
      else this.thumbnailQueue.push(run);
    });
  }
'''
if old not in s: raise SystemExit('main command/query anchor missing')
s=s.replace(old,new,1)
p.write_text(s)
