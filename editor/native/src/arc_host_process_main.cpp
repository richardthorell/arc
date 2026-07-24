#include <arc/editor/arc_host.h>
#include <arc/editor/editor_defaults.h>
#include <arc/editor/editor_gizmo.h>
#include <arc/editor/editor_state.h>
#include <arc/jobs/jobs.h>
#include <arc/memory/memory.h>
#include <arc/render/render.h>

#if defined(_WIN32) && defined(ARC_EDITOR_HOST_ENABLE_VULKAN_RENDER)
#define VK_USE_PLATFORM_WIN32_KHR
#include <arc/render/vulkan/vulkan_backend.h>

#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <windowsx.h>
#include <volk.h>
#endif

#include <atomic>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <variant>
#include <vector>

namespace
{

const char* job_priority_name(arc::job_priority value) noexcept
{
    switch (value)
    {
    case arc::job_priority::critical: return "critical";
    case arc::job_priority::high: return "high";
    case arc::job_priority::normal: return "normal";
    case arc::job_priority::low: return "low";
    case arc::job_priority::background: return "background";
    case arc::job_priority::count: break;
    }
    return "unknown";
}

const char* job_affinity_name(arc::job_affinity value) noexcept
{
    switch (value)
    {
    case arc::job_affinity::any_worker: return "worker";
    case arc::job_affinity::main_thread: return "main";
    case arc::job_affinity::render_thread: return "render";
    case arc::job_affinity::io_thread: return "io";
    }
    return "unknown";
}

const char* job_status_name(arc::job_status value) noexcept
{
    switch (value)
    {
    case arc::job_status::invalid: return "invalid";
    case arc::job_status::waiting_dependencies: return "dependencies";
    case arc::job_status::queued: return "queued";
    case arc::job_status::running: return "running";
    case arc::job_status::waiting_children: return "children";
    case arc::job_status::succeeded: return "succeeded";
    case arc::job_status::failed: return "failed";
    case arc::job_status::cancelled: return "cancelled";
    }
    return "unknown";
}

arc::editor::host_profiler_snapshot make_profiler_snapshot(
    const arc::job_system_snapshot& jobs,
    const arc::memory_snapshot& memory)
{
    arc::editor::host_profiler_snapshot result;
    result.timestamp_nanoseconds = static_cast<std::uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now().time_since_epoch()).count());
    result.memory_bytes = memory.global_bytes_outstanding;
    result.memory_soft_limit = memory.global_budget.soft_limit;
    result.memory_hard_limit = memory.global_budget.hard_limit;
    result.memory_pressure_events = memory.pressure_event_count;
    result.jobs_submitted = jobs.submitted;
    result.jobs_completed = jobs.completed;
    result.jobs_stolen = jobs.stolen;
    result.jobs_cancelled = jobs.cancelled;
    result.jobs_failed = jobs.failed;
    result.jobs_queued = jobs.queued_general + jobs.queued_main + jobs.queued_render + jobs.queued_io;
    result.dropped_profile_events = jobs.dropped_profile_events;
    result.memory_domains.reserve(memory.domains.size());
    for (const auto& domain : memory.domains)
    {
        result.memory_domains.push_back({
            .domain = std::string(arc::to_string(domain.domain)),
            .bytes_outstanding = domain.stats.bytes_outstanding,
            .peak_bytes = domain.stats.peak_bytes_outstanding,
            .soft_limit = domain.budget.soft_limit,
            .hard_limit = domain.budget.hard_limit,
            .pressure = domain.soft_limit_exceeded
        });
    }
    result.allocation_groups.reserve(memory.allocation_groups.size());
    for (const auto& group : memory.allocation_groups)
    {
        result.allocation_groups.push_back({
            .domain = std::string(arc::to_string(group.domain)),
            .tag = std::string(group.tag.name),
            .world_id = group.world_id,
            .thread_id = group.thread_id,
            .stack_id = group.stack_id,
            .allocation_count = group.allocation_count,
            .bytes_outstanding = group.bytes_outstanding
        });
    }
    result.jobs.reserve(jobs.recent_events.size());
    for (const auto& job : jobs.recent_events)
    {
        result.jobs.push_back({
            .sequence = job.sequence,
            .name = job.name,
            .priority = job_priority_name(job.priority),
            .affinity = job_affinity_name(job.affinity),
            .status = job_status_name(job.status),
            .thread_id = job.thread_id,
            .queued_nanoseconds = job.queued_nanoseconds,
            .started_nanoseconds = job.started_nanoseconds,
            .completed_nanoseconds = job.completed_nanoseconds
        });
    }
    return result;
}

#if defined(_WIN32) && defined(ARC_EDITOR_HOST_ENABLE_VULKAN_RENDER)

class native_viewport_controller;
LRESULT CALLBACK native_viewport_wnd_proc(HWND window, UINT message, WPARAM wparam, LPARAM lparam);

bool create_win32_surface(VkInstance instance, VkSurfaceKHR* surface, void* user_data)
{
    PFN_vkCreateWin32SurfaceKHR create_surface = reinterpret_cast<PFN_vkCreateWin32SurfaceKHR>(
        vkGetInstanceProcAddr(instance, "vkCreateWin32SurfaceKHR"));
    if (!create_surface)
        return false;

    VkWin32SurfaceCreateInfoKHR info{};
    info.sType = VK_STRUCTURE_TYPE_WIN32_SURFACE_CREATE_INFO_KHR;
    info.hinstance = GetModuleHandleW(nullptr);
    info.hwnd = static_cast<HWND>(user_data);
    return create_surface(instance, &info, nullptr, surface) == VK_SUCCESS;
}

class native_viewport_controller
{
public:
    native_viewport_controller(
        std::shared_ptr<arc::editor::arc_host> host,
        std::mutex& host_mutex,
        arc::job_system& jobs)
        : host_(std::move(host))
        , host_mutex_(host_mutex)
        , jobs_(&jobs)
    {
    }

    ~native_viewport_controller()
    {
        stop();
    }

    void attach(std::uint64_t native_handle, std::int32_t x, std::int32_t y, std::uint32_t width, std::uint32_t height)
    {
        {
            std::lock_guard lock(bounds_mutex_);
            parent_ = reinterpret_cast<HWND>(static_cast<std::uintptr_t>(native_handle));
            x_ = x;
            y_ = y;
            width_ = std::max(static_cast<std::uint32_t>(arc::editor::defaults::native_viewport_min_dimension), width);
            height_ = std::max(static_cast<std::uint32_t>(arc::editor::defaults::native_viewport_min_dimension), height);
            bounds_dirty_ = true;
        }

        if (!running_.exchange(true))
            render_task_ = jobs_->submit({
                .name = "editor.native_viewport",
                .priority = arc::job_priority::critical,
                .affinity = arc::job_affinity::render_thread
            }, [this] { render_loop(); });
    }

    void resize(std::int32_t x, std::int32_t y, std::uint32_t width, std::uint32_t height)
    {
        std::lock_guard lock(bounds_mutex_);
        x_ = x;
        y_ = y;
        width_ = std::max(static_cast<std::uint32_t>(arc::editor::defaults::native_viewport_min_dimension), width);
        height_ = std::max(static_cast<std::uint32_t>(arc::editor::defaults::native_viewport_min_dimension), height);
        bounds_dirty_ = true;
        terrain_hover_dirty_ = true;
    }

    void stop()
    {
        if (!running_.exchange(false))
            return;
        if (window_)
            PostMessageW(window_, WM_CLOSE, 0, 0);
        if (render_task_.valid())
            (void)render_task_.wait_result();
    }

    LRESULT handle_message(HWND window, UINT message, WPARAM wparam, LPARAM lparam)
    {
        switch (message)
        {
        case WM_ERASEBKGND:
            return 1;
        case WM_LBUTTONDOWN:
        case WM_RBUTTONDOWN:
        case WM_MBUTTONDOWN:
            begin_drag(window, message, GET_X_LPARAM(lparam), GET_Y_LPARAM(lparam));
            return 0;
        case WM_LBUTTONUP:
        case WM_RBUTTONUP:
        case WM_MBUTTONUP:
            end_drag(window, GET_X_LPARAM(lparam), GET_Y_LPARAM(lparam));
            return 0;
        case WM_MOUSEMOVE:
            if (!mouse_tracking_)
            {
                TRACKMOUSEEVENT tracking{ sizeof(TRACKMOUSEEVENT), TME_LEAVE, window, 0 };
                TrackMouseEvent(&tracking);
                mouse_tracking_ = true;
            }
            update_drag(GET_X_LPARAM(lparam), GET_Y_LPARAM(lparam));
            return 0;
        case WM_MOUSELEAVE:
            mouse_tracking_ = false;
            pointer_inside_ = false;
            clear_terrain_hover();
            return 0;
        case WM_MOUSEWHEEL:
            send_camera_input(arc::editor::host_viewport_camera_input_command{
                .zoom = static_cast<float>(GET_WHEEL_DELTA_WPARAM(wparam)) / static_cast<float>(WHEEL_DELTA)
            });
            return 0;
        case WM_CAPTURECHANGED:
            if (manipulating_)
                cancel_manipulation();
            if (terrain_stroking_)
                finish_terrain_stroke(false, drag_x_, drag_y_);
            dragging_ = false;
            drag_button_ = drag_button::none;
            selection_candidate_ = false;
            camera_drag_started_ = false;
            return 0;
        case WM_KEYDOWN:
            if (wparam == VK_ESCAPE && manipulating_)
            {
                cancel_manipulation();
                return 0;
            }
            if (wparam == VK_ESCAPE && terrain_stroking_)
            {
                finish_terrain_stroke(false, drag_x_, drag_y_);
                return 0;
            }
            handle_key(wparam);
            return 0;
        case WM_CLOSE:
            running_ = false;
            return 0;
        default:
            return DefWindowProcW(window, message, wparam, lparam);
        }
    }

private:
    enum class drag_button
    {
        none,
        left,
        right,
        middle
    };

    struct bounds
    {
        HWND parent{};
        std::int32_t x{};
        std::int32_t y{};
        std::uint32_t width{ 1 };
        std::uint32_t height{ 1 };
    };

    bounds current_bounds()
    {
        std::lock_guard lock(bounds_mutex_);
        bounds_dirty_ = false;
        return {
            .parent = parent_,
            .x = x_,
            .y = y_,
            .width = width_,
            .height = height_
        };
    }

    bool register_window_class()
    {
        WNDCLASSEXW window_class{};
        window_class.cbSize = sizeof(window_class);
        window_class.lpfnWndProc = native_viewport_wnd_proc;
        window_class.hInstance = GetModuleHandleW(nullptr);
        window_class.lpszClassName = L"ArcEditor2NativeViewport";
        return RegisterClassExW(&window_class) != 0 || GetLastError() == ERROR_CLASS_ALREADY_EXISTS;
    }

    bool create_window(const bounds& value)
    {
        if (!register_window_class() || value.parent == nullptr)
            return false;

        window_ = CreateWindowExW(
            WS_EX_NOACTIVATE,
            L"ArcEditor2NativeViewport",
            L"ARC Viewport",
            WS_CHILD | WS_VISIBLE | WS_CLIPSIBLINGS | WS_CLIPCHILDREN,
            value.x,
            value.y,
            static_cast<int>(value.width),
            static_cast<int>(value.height),
            value.parent,
            nullptr,
            GetModuleHandleW(nullptr),
            this);
        return window_ != nullptr;
    }

    bool create_backend()
    {
        arc::render::vulkan::vulkan_backend_config config{};
        config.instance_extensions = {
            VK_KHR_SURFACE_EXTENSION_NAME,
            VK_KHR_WIN32_SURFACE_EXTENSION_NAME
        };
        config.create_surface = create_win32_surface;
        config.surface_user_data = window_;

        auto result = arc::render::vulkan::create_vulkan_backend(config);
        if (!result.succeeded())
        {
            std::cerr << "arc_host_process Vulkan backend error: " << result.message << '\n';
            return false;
        }

        std::lock_guard lock(host_mutex_);
        host_->renderer_service().set_backend(std::move(result.backend));
        backend_ = arc::render::vulkan::as_vulkan_backend(host_->renderer_service().backend());
        return backend_ != nullptr;
    }

    void apply_bounds(const bounds& value)
    {
        if (!window_)
            return;
        SetWindowPos(
            window_,
            HWND_TOP,
            value.x,
            value.y,
            static_cast<int>(value.width),
            static_cast<int>(value.height),
            SWP_SHOWWINDOW | SWP_NOACTIVATE);
    }

    void render_once(const bounds& value)
    {
        if (!backend_)
            return;

        std::string message;
        bool rendered{};
        update_terrain_hover();
        {
            std::lock_guard lock(host_mutex_);
            host_->request_viewport(arc::editor::host_viewport_request{
                .frame_index = frame_index_++,
                .width = value.width,
                .height = value.height
            });
            rendered = backend_->render_native_viewport_frame(value.width, value.height, message);
        }
        if (rendered)
        {
            last_render_error_.clear();
            return;
        }
        if (message.empty())
            return;

        const auto now = std::chrono::steady_clock::now();
        if (message != last_render_error_ ||
            now - last_render_error_time_ >= std::chrono::seconds{ 5 })
        {
            std::cerr << "arc_host_process native viewport render error: " << message << '\n';
            last_render_error_ = message;
            last_render_error_time_ = now;
        }

        const bool recreate_backend =
            message.find("backend recreation required") != std::string::npos;
        if (!recreate_backend ||
            now - last_backend_recovery_attempt_ < std::chrono::seconds{ 2 })
            return;

        last_backend_recovery_attempt_ = now;
        {
            std::lock_guard lock(host_mutex_);
            backend_ = nullptr;
            host_->renderer_service().set_backend(nullptr);
        }
        if (!create_backend())
        {
            // Stop retrying every render tick. A bounds/parent update or host
            // restart can establish a new native surface.
            running_ = false;
        }
    }

    void begin_drag(HWND window, UINT message, int x, int y)
    {
        dragging_ = true;
        drag_start_x_ = x;
        drag_start_y_ = y;
        drag_x_ = x;
        drag_y_ = y;
        drag_distance_ = 0;
        camera_drag_started_ = false;
        drag_button_ = message == WM_MBUTTONDOWN
            ? drag_button::middle
            : message == WM_RBUTTONDOWN ? drag_button::right : drag_button::left;
        const bool alt = (GetKeyState(VK_MENU) & arc::editor::defaults::viewport_modifier_key_down_mask) != 0;
        selection_candidate_ = drag_button_ == drag_button::left && !alt;
        if (drag_button_ == drag_button::left && !alt)
        {
            if (!begin_terrain_stroke(x, y))
                begin_manipulation(x, y);
            else
                selection_candidate_ = false;
        }
        SetCapture(window);
        SetFocus(window);
    }

    void update_drag(int x, int y)
    {
        pointer_x_ = x;
        pointer_y_ = y;
        pointer_inside_ = true;
        terrain_hover_dirty_ = true;
        if (!dragging_)
        {
            bool terrain_mode{};
            {
                std::lock_guard lock(host_mutex_);
                terrain_mode = host_->viewport_tool_state().tool == arc::editor::host_viewport_tool::terrain;
            }
            if (!terrain_mode)
                update_gizmo_hover(x, y);
            return;
        }

        const int delta_x = x - drag_x_;
        const int delta_y = y - drag_y_;
        drag_x_ = x;
        drag_y_ = y;
        if (delta_x == 0 && delta_y == 0)
            return;
        drag_distance_ += std::abs(delta_x) + std::abs(delta_y);
        if (drag_distance_ > arc::editor::defaults::viewport_click_movement_threshold)
            selection_candidate_ = false;

        if (manipulating_)
        {
            update_manipulation(x, y);
            return;
        }
        if (terrain_stroking_)
        {
            update_terrain_stroke(x, y);
            return;
        }

        const bool shift = (GetKeyState(VK_SHIFT) & arc::editor::defaults::viewport_modifier_key_down_mask) != 0;
        const bool alt = (GetKeyState(VK_MENU) & arc::editor::defaults::viewport_modifier_key_down_mask) != 0;

        arc::editor::host_viewport_camera_input_command input;
        if (alt && drag_button_ == drag_button::left)
        {
            input.orbit_x = static_cast<float>(delta_x);
            input.orbit_y = static_cast<float>(delta_y);
        }
        else if (shift || drag_button_ == drag_button::middle)
        {
            input.pan_x = static_cast<float>(delta_x);
            input.pan_y = static_cast<float>(delta_y);
        }
        else if (drag_button_ == drag_button::right)
        {
            input.orbit_x = static_cast<float>(delta_x);
            input.orbit_y = static_cast<float>(delta_y);
        }
        else if (drag_button_ == drag_button::left &&
            drag_distance_ > arc::editor::defaults::viewport_click_movement_threshold)
        {
            // An unmodified left press remains a selection candidate until it
            // crosses the click threshold. At that point navigation owns the
            // gesture and receives the complete movement since mouse-down.
            input.orbit_x = static_cast<float>(camera_drag_started_ ? delta_x : x - drag_start_x_);
            input.orbit_y = static_cast<float>(camera_drag_started_ ? delta_y : y - drag_start_y_);
            camera_drag_started_ = true;
        }
        if (input.orbit_x != 0.0f || input.orbit_y != 0.0f || input.pan_x != 0.0f || input.pan_y != 0.0f)
        {
            send_camera_input(input);
            terrain_hover_dirty_ = true;
        }
    }

    void end_drag(HWND window, int x, int y)
    {
        const auto completed_button = drag_button_;
        dragging_ = false;
        drag_button_ = drag_button::none;
        if (manipulating_)
        {
            finish_manipulation(drag_distance_ > arc::editor::defaults::viewport_click_movement_threshold);
            if (GetCapture() == window) ReleaseCapture();
            selection_candidate_ = false;
            camera_drag_started_ = false;
            return;
        }
        if (terrain_stroking_)
        {
            finish_terrain_stroke(true, x, y);
            if (GetCapture() == window) ReleaseCapture();
            selection_candidate_ = false;
            camera_drag_started_ = false;
            return;
        }
        if (GetCapture() == window)
            ReleaseCapture();
        if (completed_button == drag_button::left && selection_candidate_)
            send_pick(std::max(0, x), std::max(0, y));
        selection_candidate_ = false;
        camera_drag_started_ = false;
    }

    void send_pick(int x, int y)
    {
        std::lock_guard lock(host_mutex_);
        host_->execute(arc::editor::host_viewport_pick_command{
            .x = static_cast<std::uint32_t>(x), .y = static_cast<std::uint32_t>(y) });
    }

    static arc::editor::editor_tool editor_tool_for(arc::editor::host_viewport_tool tool) noexcept
    {
        switch (tool)
        {
        case arc::editor::host_viewport_tool::translate: return arc::editor::editor_tool::translate;
        case arc::editor::host_viewport_tool::rotate: return arc::editor::editor_tool::rotate;
        case arc::editor::host_viewport_tool::scale: return arc::editor::editor_tool::scale;
        case arc::editor::host_viewport_tool::select: return arc::editor::editor_tool::select;
        case arc::editor::host_viewport_tool::terrain: return arc::editor::editor_tool::select;
        }
        return arc::editor::editor_tool::select;
    }

    bool begin_terrain_stroke(int x, int y)
    {
        std::lock_guard lock(host_mutex_);
        if (host_->viewport_tool_state().tool != arc::editor::host_viewport_tool::terrain)
            return false;
        const auto snapshot = host_->selected_entity_snapshot();
        if (!snapshot.entity.valid() || !snapshot.terrain || !host_->terrain_tool_snapshot().hover_visible)
            return false;
        const auto transaction = ++next_manipulation_transaction_;
        const bool invert = (GetKeyState(VK_SHIFT) & arc::editor::defaults::viewport_modifier_key_down_mask) != 0;
        const auto response = host_->execute(arc::editor::host_command_envelope{
            .command_type = "terrain.stroke",
            .payload = arc::editor::host_terrain_stroke_command{
                snapshot.entity, static_cast<std::uint32_t>(std::max(0, x)), static_cast<std::uint32_t>(std::max(0, y)),
                arc::editor::host_edit_phase::begin, invert },
            .edit = arc::editor::host_edit_transaction{
                transaction, arc::editor::host_edit_phase::begin, "Terrain Stroke" }
        });
        if (!response.succeeded || response.payload_json.find("\"hit\":true") == std::string::npos)
        {
            host_->execute(arc::editor::host_command_envelope{
                .command_type = "terrain.stroke",
                .payload = arc::editor::host_terrain_stroke_command{
                    snapshot.entity, static_cast<std::uint32_t>(std::max(0, x)), static_cast<std::uint32_t>(std::max(0, y)),
                    arc::editor::host_edit_phase::cancel, false },
                .edit = arc::editor::host_edit_transaction{
                    transaction, arc::editor::host_edit_phase::cancel, "Terrain Stroke" }
            });
            return false;
        }
        terrain_stroking_ = true;
        terrain_entity_ = snapshot.entity;
        terrain_transaction_ = transaction;
        terrain_last_preview_frame_ = frame_index_;
        return true;
    }

    void update_terrain_hover()
    {
        if (!terrain_hover_dirty_ || dragging_ || !pointer_inside_)
            return;
        terrain_hover_dirty_ = false;
        std::lock_guard lock(host_mutex_);
        if (host_->viewport_tool_state().tool != arc::editor::host_viewport_tool::terrain)
            return;
        const auto snapshot = host_->selected_entity_snapshot();
        if (!snapshot.entity.valid() || !snapshot.terrain)
            return;
        host_->execute(arc::editor::host_terrain_hover_command{
            .entity = snapshot.entity,
            .x = static_cast<std::uint32_t>(std::max(0, pointer_x_)),
            .y = static_cast<std::uint32_t>(std::max(0, pointer_y_))
        });
    }

    void clear_terrain_hover()
    {
        std::lock_guard lock(host_mutex_);
        const auto snapshot = host_->selected_entity_snapshot();
        if (!snapshot.entity.valid() || !snapshot.terrain)
            return;
        host_->execute(arc::editor::host_terrain_hover_command{
            .entity = snapshot.entity,
            .clear = true
        });
    }

    void update_terrain_stroke(int x, int y)
    {
        if (terrain_last_preview_frame_ == frame_index_)
            return;
        terrain_last_preview_frame_ = frame_index_;
        std::lock_guard lock(host_mutex_);
        const bool invert = (GetKeyState(VK_SHIFT) & arc::editor::defaults::viewport_modifier_key_down_mask) != 0;
        host_->execute(arc::editor::host_command_envelope{
            .command_type = "terrain.stroke",
            .payload = arc::editor::host_terrain_stroke_command{
                terrain_entity_, static_cast<std::uint32_t>(std::max(0, x)), static_cast<std::uint32_t>(std::max(0, y)),
                arc::editor::host_edit_phase::update, invert },
            .edit = arc::editor::host_edit_transaction{
                terrain_transaction_, arc::editor::host_edit_phase::update, "Terrain Stroke" }
        });
    }

    void finish_terrain_stroke(bool commit, int x, int y)
    {
        const auto phase = commit ? arc::editor::host_edit_phase::commit : arc::editor::host_edit_phase::cancel;
        std::lock_guard lock(host_mutex_);
        host_->execute(arc::editor::host_command_envelope{
            .command_type = "terrain.stroke",
            .payload = arc::editor::host_terrain_stroke_command{
                terrain_entity_, static_cast<std::uint32_t>(std::max(0, x)), static_cast<std::uint32_t>(std::max(0, y)), phase, false },
            .edit = arc::editor::host_edit_transaction{ terrain_transaction_, phase, "Terrain Stroke" }
        });
        terrain_stroking_ = false;
        terrain_entity_ = {};
    }

    arc::editor::editor_gizmo_context gizmo_context() const
    {
        const auto& tool = host_->viewport_tool_state();
        std::lock_guard bounds_lock(bounds_mutex_);
        return {
            .tool = editor_tool_for(tool.tool),
            .coordinate_space = tool.coordinate_space == arc::editor::host_coordinate_space::local
                ? arc::editor::gizmo_coordinate_space::local : arc::editor::gizmo_coordinate_space::world,
            .highlighted_axis = active_axis_,
            .viewport_width = width_,
            .viewport_height = height_
        };
    }

    void update_gizmo_hover(int x, int y)
    {
        std::lock_guard lock(host_mutex_);
        const auto& state = host_->scene_state();
        const auto axis = arc::editor::hit_test_editor_gizmo(state.scene, state.selected_entity, state.camera_entity,
            gizmo_context(), static_cast<float>(x), static_cast<float>(y));
        active_axis_ = axis;
        host_->set_viewport_gizmo_highlight(axis);
    }

    void begin_manipulation(int x, int y)
    {
        std::lock_guard lock(host_mutex_);
        const auto& state = host_->scene_state();
        const auto axis = arc::editor::hit_test_editor_gizmo(state.scene, state.selected_entity, state.camera_entity,
            gizmo_context(), static_cast<float>(x), static_cast<float>(y));
        if (axis == arc::editor::gizmo_axis::none) return;
        const auto snapshot = host_->selected_entity_snapshot();
        if (!snapshot.entity.valid() || !snapshot.transform) return;
        manipulating_ = true;
        active_axis_ = axis;
        manipulation_entity_ = snapshot.entity;
        manipulation_original_ = *snapshot.transform;
        manipulation_current_ = *snapshot.transform;
        manipulation_start_x_ = x;
        manipulation_start_y_ = y;
        manipulation_transaction_ = ++next_manipulation_transaction_;
        manipulation_local_axis_ = {};
        manipulation_local_axis_[static_cast<std::size_t>(axis) - 1u] = 1.0f;
        manipulation_rotation_axis_ = manipulation_local_axis_;
        manipulation_world_units_per_pixel_ = 0.02f;
        const auto selected_entity = arc::scene::entity{ snapshot.entity.index, snapshot.entity.generation };
        const auto* selected_transform = state.scene.try_get<arc::scene::transform_component>(selected_entity);
        const auto* camera = state.scene.try_get<arc::scene::camera_component>(state.camera_entity);
        const auto* camera_transform = state.scene.try_get<arc::scene::transform_component>(state.camera_entity);
        if (selected_transform && camera && camera_transform)
        {
            const std::size_t axis_index = static_cast<std::size_t>(axis) - 1u;
            arc::math::vector3f world_axis{};
            world_axis[axis_index] = 1.0f;
            if (gizmo_context().coordinate_space == arc::editor::gizmo_coordinate_space::local)
                world_axis = arc::math::normalize(arc::math::vector3f{
                    selected_transform->world(0, axis_index), selected_transform->world(1, axis_index), selected_transform->world(2, axis_index) });
            manipulation_local_axis_ = world_axis;
            manipulation_rotation_axis_ = world_axis;
            auto parent = state.scene.try_get<arc::scene::hierarchy_component>(selected_entity)
                ? state.scene.get<arc::scene::hierarchy_component>(selected_entity).parent : arc::scene::entity{};
            while (state.scene.alive(parent) && !state.scene.has<arc::scene::transform_component>(parent))
            {
                const auto* hierarchy = state.scene.try_get<arc::scene::hierarchy_component>(parent);
                parent = hierarchy ? hierarchy->parent : arc::scene::entity{};
            }
            if (const auto* parent_transform = state.scene.try_get<arc::scene::transform_component>(parent))
            {
                arc::math::matrix4f inverse_parent;
                if (arc::scene::inverse_affine(parent_transform->world, inverse_parent))
                {
                    manipulation_local_axis_ = arc::math::transform_vector(inverse_parent, world_axis);
                    manipulation_rotation_axis_ = arc::math::normalize(manipulation_local_axis_);
                }
            }
            manipulation_world_units_per_pixel_ = arc::editor::editor_gizmo_world_scale(*camera, *camera_transform,
                arc::scene::world_position(*selected_transform), height_) / arc::editor::editor_gizmo_pixel_length;
            manipulation_rotation_is_local_ = gizmo_context().coordinate_space == arc::editor::gizmo_coordinate_space::local;
        }
        host_->set_viewport_gizmo_highlight(axis);
        host_->execute(arc::editor::host_command_envelope{
            .command_type = "entity.setTransform",
            .payload = arc::editor::host_set_transform_command{ manipulation_entity_, manipulation_original_ },
            .edit = arc::editor::host_edit_transaction{ manipulation_transaction_, arc::editor::host_edit_phase::begin, "Gizmo Transform" }
        });
    }

    static float snapped(float value, float interval) noexcept
    {
        return interval > 0.0f ? std::round(value / interval) * interval : value;
    }

    static float axis_value(const arc::editor::host_vec3& value, std::size_t axis) noexcept
    {
        return axis == 0 ? value.x : axis == 1 ? value.y : value.z;
    }

    static void set_axis_value(arc::editor::host_vec3& value, std::size_t axis, float next) noexcept
    {
        if (axis == 0) value.x = next;
        else if (axis == 1) value.y = next;
        else value.z = next;
    }

    static arc::math::quatf multiply_rotation(const arc::math::quatf& lhs, const arc::math::quatf& rhs) noexcept
    {
        return arc::math::normalize(arc::math::quatf{
            lhs[3] * rhs[0] + lhs[0] * rhs[3] + lhs[1] * rhs[2] - lhs[2] * rhs[1],
            lhs[3] * rhs[1] - lhs[0] * rhs[2] + lhs[1] * rhs[3] + lhs[2] * rhs[0],
            lhs[3] * rhs[2] + lhs[0] * rhs[1] - lhs[1] * rhs[0] + lhs[2] * rhs[3],
            lhs[3] * rhs[3] - lhs[0] * rhs[0] - lhs[1] * rhs[1] - lhs[2] * rhs[2] });
    }

    void update_manipulation(int x, int y)
    {
        std::lock_guard lock(host_mutex_);
        const auto& tool = host_->viewport_tool_state();
        const float pixel_delta = active_axis_ == arc::editor::gizmo_axis::y
            ? static_cast<float>(manipulation_start_y_ - y)
            : active_axis_ == arc::editor::gizmo_axis::z
                ? static_cast<float>((x - manipulation_start_x_) + (manipulation_start_y_ - y)) * 0.5f
                : static_cast<float>(x - manipulation_start_x_);
        const std::size_t axis = static_cast<std::size_t>(active_axis_) - 1u;
        manipulation_current_ = manipulation_original_;
        if (tool.tool == arc::editor::host_viewport_tool::translate)
        {
            float delta = pixel_delta * manipulation_world_units_per_pixel_;
            if (tool.snapping) delta = snapped(delta, tool.translation_snap);
            manipulation_current_.position.x = manipulation_original_.position.x + manipulation_local_axis_[0] * delta;
            manipulation_current_.position.y = manipulation_original_.position.y + manipulation_local_axis_[1] * delta;
            manipulation_current_.position.z = manipulation_original_.position.z + manipulation_local_axis_[2] * delta;
        }
        else if (tool.tool == arc::editor::host_viewport_tool::scale)
        {
            float value = std::max(0.001f, axis_value(manipulation_original_.scale, axis) * std::exp(pixel_delta * 0.01f));
            if (tool.snapping) value = std::max(0.001f, snapped(value, tool.scale_snap));
            set_axis_value(manipulation_current_.scale, axis, value);
        }
        else if (tool.tool == arc::editor::host_viewport_tool::rotate)
        {
            float degrees = pixel_delta * 0.35f;
            if (tool.snapping) degrees = snapped(degrees, tool.rotation_snap_degrees);
            arc::math::vector3f local_axis{};
            local_axis[axis] = 1.0f;
            const auto delta = arc::math::from_axis_angle(
                manipulation_rotation_is_local_ ? local_axis : manipulation_rotation_axis_, arc::math::to_radians(degrees));
            const arc::math::quatf original{ manipulation_original_.rotation.x, manipulation_original_.rotation.y,
                manipulation_original_.rotation.z, manipulation_original_.rotation.w };
            const auto result = manipulation_rotation_is_local_
                ? multiply_rotation(original, delta) : multiply_rotation(delta, original);
            manipulation_current_.rotation = { result[0], result[1], result[2], result[3] };
        }
        host_->execute(arc::editor::host_command_envelope{
            .command_type = "entity.setTransform",
            .payload = arc::editor::host_set_transform_command{ manipulation_entity_, manipulation_current_ },
            .edit = arc::editor::host_edit_transaction{ manipulation_transaction_, arc::editor::host_edit_phase::update, "Gizmo Transform" }
        });
    }

    void finish_manipulation(bool commit)
    {
        const auto phase = commit ? arc::editor::host_edit_phase::commit : arc::editor::host_edit_phase::cancel;
        {
            std::lock_guard lock(host_mutex_);
            host_->execute(arc::editor::host_command_envelope{
                .command_type = "entity.setTransform",
                .payload = arc::editor::host_set_transform_command{ manipulation_entity_, manipulation_current_ },
                .edit = arc::editor::host_edit_transaction{ manipulation_transaction_, phase, "Gizmo Transform" }
            });
            host_->set_viewport_gizmo_highlight(arc::editor::gizmo_axis::none);
        }
        manipulating_ = false;
        active_axis_ = arc::editor::gizmo_axis::none;
    }

    void cancel_manipulation()
    {
        finish_manipulation(false);
    }

    void handle_key(WPARAM key)
    {
        if (key == 'F')
            return send_camera_input(arc::editor::host_viewport_camera_input_command{ .focus_selected = true });
        if (key == VK_DELETE || ((GetKeyState(VK_CONTROL) & 0x8000) && key == 'D'))
        {
            std::lock_guard lock(host_mutex_);
            const auto selected = host_->selected_entity_snapshot().entity;
            if (!selected.valid()) return;
            if (key == VK_DELETE) host_->execute(arc::editor::host_delete_entity_command{ selected });
            else host_->execute(arc::editor::host_duplicate_entity_command{ selected });
            return;
        }
        if (key == VK_OEM_4 || key == VK_OEM_6)
        {
            std::lock_guard lock(host_mutex_);
            const auto snapshot = host_->selected_entity_snapshot();
            if (!snapshot.entity.valid() || !snapshot.terrain) return;
            const float multiplier = key == VK_OEM_4 ? 0.8f : 1.25f;
            const auto& terrain = *snapshot.terrain;
            host_->execute(arc::editor::host_set_terrain_brush_command{
                snapshot.entity,
                terrain.brush_tool,
                std::clamp(terrain.brush_radius * multiplier, 0.25f, 128.0f),
                terrain.brush_strength,
                terrain.brush_falloff,
                terrain.active_layer });
            return;
        }
        arc::editor::host_viewport_tool tool;
        if (key == 'Q') tool = arc::editor::host_viewport_tool::select;
        else if (key == 'W') tool = arc::editor::host_viewport_tool::translate;
        else if (key == 'E') tool = arc::editor::host_viewport_tool::rotate;
        else if (key == 'R') tool = arc::editor::host_viewport_tool::scale;
        else return;
        std::lock_guard lock(host_mutex_);
        auto command = host_->viewport_tool_state();
        command.tool = tool;
        host_->execute(command);
    }

    void send_camera_input(const arc::editor::host_viewport_camera_input_command& input)
    {
        std::lock_guard lock(host_mutex_);
        host_->execute(arc::editor::host_command_envelope{
            .command_type = "viewport.cameraInput",
            .payload = input
        });
    }

    void render_loop()
    {
        auto value = current_bounds();
        if (!create_window(value))
        {
            std::cerr << "arc_host_process failed to create native viewport window\n";
            running_ = false;
            return;
        }
        apply_bounds(value);
        if (!create_backend())
        {
            running_ = false;
            return;
        }

        while (running_)
        {
            jobs_->pump_render_thread(32);
            MSG message{};
            while (PeekMessageW(&message, nullptr, 0, 0, PM_REMOVE))
            {
                if (message.message == WM_QUIT)
                    running_ = false;
                TranslateMessage(&message);
                DispatchMessageW(&message);
            }

            {
                std::lock_guard lock(bounds_mutex_);
                if (bounds_dirty_)
                {
                    value = {
                        .parent = parent_,
                        .x = x_,
                        .y = y_,
                        .width = width_,
                        .height = height_
                    };
                    bounds_dirty_ = false;
                    apply_bounds(value);
                }
            }

            render_once(value);
            std::this_thread::sleep_for(arc::editor::defaults::native_viewport_frame_interval);
        }

        {
            std::lock_guard lock(host_mutex_);
            host_->renderer_service().set_backend(nullptr);
        }
        if (window_)
        {
            DestroyWindow(window_);
            window_ = nullptr;
        }
    }

    std::shared_ptr<arc::editor::arc_host> host_;
    std::mutex& host_mutex_;
    arc::job_system* jobs_{};
    arc::job_handle render_task_;
    std::atomic<bool> running_{};
    mutable std::mutex bounds_mutex_;
    HWND parent_{};
    HWND window_{};
    arc::render::vulkan::vulkan_backend* backend_{};
    std::int32_t x_{};
    std::int32_t y_{};
    std::uint32_t width_{ 1 };
    std::uint32_t height_{ 1 };
    bool bounds_dirty_{};
    std::uint64_t frame_index_{};
    std::string last_render_error_;
    std::chrono::steady_clock::time_point last_render_error_time_{};
    std::chrono::steady_clock::time_point last_backend_recovery_attempt_{};
    bool dragging_{};
    drag_button drag_button_{ drag_button::none };
    int drag_x_{};
    int drag_y_{};
    int drag_start_x_{};
    int drag_start_y_{};
    int drag_distance_{};
    bool selection_candidate_{};
    bool camera_drag_started_{};
    bool mouse_tracking_{};
    bool pointer_inside_{};
    bool terrain_hover_dirty_{};
    int pointer_x_{};
    int pointer_y_{};
    bool manipulating_{};
    bool terrain_stroking_{};
    arc::editor::host_entity_id terrain_entity_{};
    std::uint64_t terrain_transaction_{};
    std::uint64_t terrain_last_preview_frame_{};
    arc::editor::gizmo_axis active_axis_{ arc::editor::gizmo_axis::none };
    arc::editor::host_entity_id manipulation_entity_{};
    arc::editor::host_transform manipulation_original_{};
    arc::editor::host_transform manipulation_current_{};
    int manipulation_start_x_{};
    int manipulation_start_y_{};
    std::uint64_t manipulation_transaction_{};
    std::uint64_t next_manipulation_transaction_{};
    arc::math::vector3f manipulation_local_axis_{ 1.0f, 0.0f, 0.0f };
    arc::math::vector3f manipulation_rotation_axis_{ 1.0f, 0.0f, 0.0f };
    float manipulation_world_units_per_pixel_{ 0.02f };
    bool manipulation_rotation_is_local_{};
};

LRESULT CALLBACK native_viewport_wnd_proc(HWND window, UINT message, WPARAM wparam, LPARAM lparam)
{
    if (message == WM_NCCREATE)
    {
        const auto* create = reinterpret_cast<CREATESTRUCTW*>(lparam);
        SetWindowLongPtrW(window, GWLP_USERDATA, reinterpret_cast<LONG_PTR>(create->lpCreateParams));
    }

    if (auto* controller = reinterpret_cast<native_viewport_controller*>(GetWindowLongPtrW(window, GWLP_USERDATA)))
        return controller->handle_message(window, message, wparam, lparam);

    return DefWindowProcW(window, message, wparam, lparam);
}

#else

class native_viewport_controller
{
public:
    native_viewport_controller(std::shared_ptr<arc::editor::arc_host>, std::mutex&, arc::job_system&)
    {
    }

    void attach(std::uint64_t, std::int32_t, std::int32_t, std::uint32_t, std::uint32_t)
    {
        std::cerr << "arc_host_process native viewport rendering is not available in this build\n";
    }

    void resize(std::int32_t, std::int32_t, std::uint32_t, std::uint32_t)
    {
    }

    void stop()
    {
    }
};

#endif

} // namespace

int main()
{
    auto& memory = arc::default_memory_system();
    arc::job_system jobs({ .memory = &memory });
    jobs.register_main_thread();
    auto host = std::make_shared<arc::editor::arc_host>(std::make_unique<arc::render::renderer>());
    std::mutex host_mutex;
    std::mutex output_mutex;
    native_viewport_controller native_viewport(host, host_mutex, jobs);
    const auto write_response = [&](const arc::editor::host_response& response) {
        std::lock_guard output_lock(output_mutex);
        std::cout << arc::editor::to_json(response) << '\n';
        std::cout.flush();
    };
    std::jthread event_pump([&](std::stop_token stop) {
        auto next_profiler_sample = std::chrono::steady_clock::now();
        std::uint64_t profiler_sequence = std::uint64_t{ 1 } << 63u;
        while (!stop.stop_requested())
        {
            std::vector<arc::editor::host_event> events;
            {
                std::lock_guard host_lock(host_mutex);
                events = host->poll_events();
            }
            if (!events.empty())
            {
                std::lock_guard output_lock(output_mutex);
                for (const auto& event : events)
                    std::cout << arc::editor::to_json(event) << '\n';
                std::cout.flush();
            }
            const auto now = std::chrono::steady_clock::now();
            if (now >= next_profiler_sample)
            {
                const auto snapshot = make_profiler_snapshot(jobs.snapshot(true), memory.snapshot());
                const arc::editor::host_event event{
                    .sequence = profiler_sequence++,
                    .event_type = arc::editor::host_event_type::profiler_snapshot,
                    .message = "Profiler snapshot",
                    .payload_json = arc::editor::to_json(snapshot)
                };
                std::lock_guard output_lock(output_mutex);
                std::cout << arc::editor::to_json(event) << '\n';
                std::cout.flush();
                next_profiler_sample = now + std::chrono::milliseconds(100);
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(16));
        }
    });

    std::string line;
    while (std::getline(std::cin, line))
    {
        jobs.pump_main_thread();
        if (line.empty())
            continue;

        std::string error;
        if (line.find("\"kind\":\"query\"") != std::string::npos || line.find("\"kind\": \"query\"") != std::string::npos)
        {
            arc::editor::host_query_envelope query;
            if (!arc::editor::from_json(line, query, error))
            {
                std::cerr << "arc_host_process query parse error: " << error << '\n';
                write_response(arc::editor::host_response{
                    .request_id = query.request_id,
                    .succeeded = false,
                    .error = error });
                continue;
            }
            arc::editor::host_response response;
            {
                std::lock_guard lock(host_mutex);
                response = host->query(query);
            }
            write_response(response);
        }
        else
        {
            arc::editor::host_command_envelope command;
            if (!arc::editor::from_json(line, command, error))
            {
                std::cerr << "arc_host_process command parse error: " << error << '\n';
                write_response(arc::editor::host_response{
                    .request_id = command.request_id,
                    .succeeded = false,
                    .error = error });
                continue;
            }
            arc::editor::host_response response;
            {
                std::lock_guard lock(host_mutex);
                response = host->execute(command);
            }
            write_response(response);

            if (response.succeeded)
            {
                if (const auto* attach = std::get_if<arc::editor::host_viewport_attach_command>(&command.payload))
                    native_viewport.attach(attach->native_handle, attach->x, attach->y, attach->width, attach->height);
                else if (const auto* resize = std::get_if<arc::editor::host_viewport_resize_command>(&command.payload))
                    native_viewport.resize(resize->x, resize->y, resize->width, resize->height);
            }
        }

    }

    event_pump.request_stop();
    native_viewport.stop();
    return 0;
}
