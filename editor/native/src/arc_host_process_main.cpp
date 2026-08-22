#include <arc/editor/arc_host.h>
#include <arc/editor/editor_defaults.h>
#include <arc/editor/editor_gizmo.h>
#include <arc/editor/editor_state.h>
#include <arc/editor/world_environment_host.h>
#include <arc/jobs/jobs.h>
#include <arc/memory/memory.h>
#include <arc/render/render.h>

#if defined(_WIN32) && defined(ARC_EDITOR_HOST_ENABLE_VULKAN_RENDER)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#define VK_USE_PLATFORM_WIN32_KHR
#include <arc/render/vulkan/vulkan_backend.h>

#include <windows.h>
#include <windowsx.h>
#include <volk.h>
#endif

#include <atomic>
#include <algorithm>
#include <chrono>
#include <charconv>
#include <cmath>
#include <condition_variable>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <mutex>
#include <string>
#include <sstream>
#include <thread>
#include <unordered_map>
#include <variant>
#include <vector>

namespace
{

const char* job_priority_name(arc::jobs::job_priority value) noexcept
{
    switch (value)
    {
        case arc::jobs::job_priority::critical:
            return "critical";
        case arc::jobs::job_priority::high:
            return "high";
        case arc::jobs::job_priority::normal:
            return "normal";
        case arc::jobs::job_priority::low:
            return "low";
        case arc::jobs::job_priority::background:
            return "background";
        case arc::jobs::job_priority::count:
            break;
    }
    return "unknown";
}

const char* job_affinity_name(arc::jobs::job_affinity value) noexcept
{
    switch (value)
    {
        case arc::jobs::job_affinity::any_worker:
            return "worker";
        case arc::jobs::job_affinity::main_thread:
            return "main";
        case arc::jobs::job_affinity::render_thread:
            return "render";
        case arc::jobs::job_affinity::io_thread:
            return "io";
    }
    return "unknown";
}

const char* job_status_name(arc::jobs::job_status value) noexcept
{
    switch (value)
    {
        case arc::jobs::job_status::invalid:
            return "invalid";
        case arc::jobs::job_status::waiting_dependencies:
            return "dependencies";
        case arc::jobs::job_status::queued:
            return "queued";
        case arc::jobs::job_status::running:
            return "running";
        case arc::jobs::job_status::waiting_children:
            return "children";
        case arc::jobs::job_status::succeeded:
            return "succeeded";
        case arc::jobs::job_status::failed:
            return "failed";
        case arc::jobs::job_status::cancelled:
            return "cancelled";
    }
    return "unknown";
}

arc::editor::host_profiler_snapshot make_profiler_snapshot(const arc::jobs::job_system_snapshot& jobs,
                                                           const arc::memory::memory_snapshot& memory)
{
    arc::editor::host_profiler_snapshot result;
    result.timestamp_nanoseconds = static_cast<std::uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch())
            .count());
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
        result.memory_domains.push_back({.domain = std::string(arc::memory::to_string(domain.domain)),
                                         .bytes_outstanding = domain.stats.bytes_outstanding,
                                         .peak_bytes = domain.stats.peak_bytes_outstanding,
                                         .soft_limit = domain.budget.soft_limit,
                                         .hard_limit = domain.budget.hard_limit,
                                         .pressure = domain.soft_limit_exceeded});
    }
    result.allocation_groups.reserve(memory.allocation_groups.size());
    for (const auto& group : memory.allocation_groups)
    {
        result.allocation_groups.push_back({.domain = std::string(arc::memory::to_string(group.domain)),
                                            .tag = std::string(group.tag.name),
                                            .world_id = group.world_id,
                                            .thread_id = group.thread_id,
                                            .stack_id = group.stack_id,
                                            .allocation_count = group.allocation_count,
                                            .bytes_outstanding = group.bytes_outstanding});
    }
    result.jobs.reserve(jobs.recent_events.size());
    for (const auto& job : jobs.recent_events)
    {
        result.jobs.push_back({.sequence = job.sequence,
                               .name = job.name,
                               .priority = job_priority_name(job.priority),
                               .affinity = job_affinity_name(job.affinity),
                               .status = job_status_name(job.status),
                               .thread_id = job.thread_id,
                               .queued_nanoseconds = job.queued_nanoseconds,
                               .started_nanoseconds = job.started_nanoseconds,
                               .completed_nanoseconds = job.completed_nanoseconds});
    }
    return result;
}

#if defined(_WIN32) && defined(ARC_EDITOR_HOST_ENABLE_VULKAN_RENDER)

class native_viewport_controller;
LRESULT CALLBACK native_viewport_wnd_proc(HWND window, UINT message, WPARAM wparam, LPARAM lparam);

bool create_win32_surface(VkInstance instance, PFN_vkGetInstanceProcAddr get_instance_proc_address,
                          VkSurfaceKHR* surface, void* user_data)
{
    if (!get_instance_proc_address) return false;
    PFN_vkCreateWin32SurfaceKHR create_surface =
        reinterpret_cast<PFN_vkCreateWin32SurfaceKHR>(get_instance_proc_address(instance, "vkCreateWin32SurfaceKHR"));
    if (!create_surface) return false;

    VkWin32SurfaceCreateInfoKHR info{};
    info.sType = VK_STRUCTURE_TYPE_WIN32_SURFACE_CREATE_INFO_KHR;
    info.hinstance = GetModuleHandleW(nullptr);
    info.hwnd = static_cast<HWND>(user_data);
    return create_surface(instance, &info, nullptr, surface) == VK_SUCCESS;
}

class native_viewport_controller
{
public:
    native_viewport_controller(std::shared_ptr<arc::editor::arc_host> host, std::mutex& host_mutex,
                               std::mutex& output_mutex, arc::jobs::job_system& jobs)
        : host_(std::move(host)), host_mutex_(host_mutex), output_mutex_(output_mutex), jobs_(&jobs)
    {
    }

    ~native_viewport_controller()
    {
        stop();
    }

    void attach(std::string viewport_id, std::uint64_t native_handle, std::int32_t x, std::int32_t y,
                std::uint32_t width, std::uint32_t height)
    {
        // Native-window presentation and shared-texture presentation require
        // different Vulkan backend configurations. Preserve the native fallback
        // as a single surface and restart cleanly if it replaces shared outputs.
        if (running_.load() && shared_texture_) stop();
        {
            std::lock_guard lock(bounds_mutex_);
            parent_ = reinterpret_cast<HWND>(static_cast<std::uintptr_t>(native_handle));
            viewport_id_ = viewport_id.empty() ? "viewport-1" : std::move(viewport_id);
            x_ = x;
            y_ = y;
            width_ = std::max(static_cast<std::uint32_t>(arc::editor::defaults::native_viewport_min_dimension), width);
            height_ =
                std::max(static_cast<std::uint32_t>(arc::editor::defaults::native_viewport_min_dimension), height);
            bounds_dirty_ = true;
            attached_ = true;
            shared_texture_ = false;
        }

        if (!running_.exchange(true))
            render_task_ = jobs_->submit({.name = "editor.native_viewport",
                                          .priority = arc::jobs::job_priority::critical,
                                          .affinity = arc::jobs::job_affinity::render_thread,
                                          .dependencies = {},
                                          .dependency_view = {},
                                          .parent = {},
                                          .cancellation = {},
                                          .dependency_policy = arc::jobs::job_dependency_policy::cancel_on_failure},
                                         [this] { render_loop(); });
    }

    bool create_shared(std::string viewport_id, std::uint64_t consumer_process_id, std::uint32_t width,
                       std::uint32_t height, std::string& error)
    {
        if (consumer_process_id == 0 || consumer_process_id > std::numeric_limits<DWORD>::max())
        {
            error = "Shared viewport consumer process is invalid";
            return false;
        }
        const auto id = viewport_id.empty() ? std::string{"viewport-1"} : std::move(viewport_id);
        const bool renderer_running = running_.load();
        {
            std::lock_guard lock(bounds_mutex_);
            if (renderer_running && !shared_texture_)
            {
                error = "A native viewport renderer is already active";
                return false;
            }
            auto [entry, inserted] = shared_surfaces_.try_emplace(id);
            auto& surface = entry->second;
            surface.viewport_id = id;
            surface.consumer_process_id = static_cast<DWORD>(consumer_process_id);
            surface.width = std::max(1u, width);
            surface.height = std::max(1u, height);
            surface.attached = true;
            surface.visible = true;
            surface.destroy_dirty = false;
            if (inserted || !surface.output_created)
                surface.create_dirty = true;
            else
            {
                surface.resize_dirty = true;
                surface.visibility_dirty = true;
            }
            parent_ = nullptr;
            shared_texture_ = true;
            attached_ = true;
            if (shared_surfaces_.size() == 1u || id == "viewport-1") activate_shared_surface_locked(surface);
        }
        if (renderer_running) return true;
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
    }

    void release_frame(std::string viewport_id, std::uint64_t generation, std::uint64_t frame_id,
                       std::string consumer_handle)
    {
        DWORD consumer_process_id{};
        {
            std::lock_guard lock(bounds_mutex_);
            if (const auto found = shared_surfaces_.find(viewport_id); found != shared_surfaces_.end())
                consumer_process_id = found->second.consumer_process_id;
        }
        if (!consumer_handle.empty() && consumer_process_id != 0)
        {
            std::uint64_t value{};
            const auto first = consumer_handle.starts_with("0x") ? consumer_handle.data() + 2 : consumer_handle.data();
            const auto last = consumer_handle.data() + consumer_handle.size();
            if (std::from_chars(first, last, value, 16).ec == std::errc{})
            {
                HANDLE consumer = OpenProcess(PROCESS_DUP_HANDLE, FALSE, consumer_process_id);
                if (consumer != nullptr)
                {
                    HANDLE local{};
                    if (DuplicateHandle(consumer, reinterpret_cast<HANDLE>(static_cast<std::uintptr_t>(value)),
                                        GetCurrentProcess(), &local, 0, FALSE,
                                        DUPLICATE_SAME_ACCESS | DUPLICATE_CLOSE_SOURCE))
                        CloseHandle(local);
                    CloseHandle(consumer);
                }
            }
        }
        std::lock_guard lock(bounds_mutex_);
        pending_releases_.push_back({std::move(viewport_id), generation, frame_id});
    }

    void set_visible(std::string_view viewport_id, bool visible)
    {
        std::lock_guard lock(bounds_mutex_);
        if (shared_texture_)
        {
            const auto found = shared_surfaces_.find(std::string{viewport_id});
            if (found == shared_surfaces_.end()) return;
            found->second.visible = visible;
            found->second.visibility_dirty = true;
            return;
        }
        if (viewport_id != viewport_id_) return;
        visible_ = visible;
        visibility_dirty_ = true;
    }

    void pointer(const arc::editor::host_viewport_pointer_command& pointer)
    {
        std::lock_guard lock(bounds_mutex_);
        if (shared_texture_)
        {
            const auto found = shared_surfaces_.find(pointer.viewport_id);
            if (found == shared_surfaces_.end() || !found->second.attached) return;
        }
        else if (pointer.viewport_id != viewport_id_)
            return;
        pending_pointer_inputs_.push_back(pointer);
    }

    void key(const arc::editor::host_viewport_key_command& key)
    {
        std::lock_guard lock(bounds_mutex_);
        if (shared_texture_)
        {
            const auto found = shared_surfaces_.find(key.viewport_id);
            if (found == shared_surfaces_.end() || !found->second.attached) return;
        }
        else if (key.viewport_id != viewport_id_)
            return;
        pending_key_inputs_.push_back(key);
    }

private:
    void process_pointer(const arc::editor::host_viewport_pointer_command& pointer)
    {
        activate_interaction_surface(pointer.viewport_id);
        input_alt_ = pointer.alt;
        input_shift_ = pointer.shift;
        input_control_ = pointer.control;
        if (pointer.phase == arc::editor::host_viewport_pointer_phase::down)
        {
            const UINT message = pointer.button == 1   ? WM_MBUTTONDOWN
                                 : pointer.button == 2 ? WM_RBUTTONDOWN
                                                       : WM_LBUTTONDOWN;
            begin_drag(nullptr, message, pointer.x, pointer.y);
        }
        else if (pointer.phase == arc::editor::host_viewport_pointer_phase::move)
            update_drag(pointer.x, pointer.y);
        else if (pointer.phase == arc::editor::host_viewport_pointer_phase::up)
            end_drag(nullptr, pointer.x, pointer.y);
        else if (pointer.phase == arc::editor::host_viewport_pointer_phase::wheel)
            send_camera_input(arc::editor::host_viewport_camera_input_command{.zoom = pointer.wheel});
        else
            cancel_pointer_interaction();
    }

    void process_key(const arc::editor::host_viewport_key_command& key)
    {
        activate_interaction_surface(key.viewport_id);
        input_alt_ = key.alt;
        input_shift_ = key.shift;
        input_control_ = key.control;
        if (!key.down)
        {
            if (sun_rotating_ && (key.key == "l" || key.key == "L" || key.key == "Control")) finish_sun_rotation(true);
            return;
        }
        WPARAM native_key = key.key == "Escape"   ? VK_ESCAPE
                            : key.key == "Delete" ? VK_DELETE
                            : key.key == "["      ? VK_OEM_4
                            : key.key == "]"      ? VK_OEM_6
                            : key.key.empty()     ? 0
                                                  : static_cast<WPARAM>(std::toupper(key.key.front()));
        if (native_key == VK_ESCAPE)
        {
            cancel_pointer_interaction();
            return;
        }
        if (!sun_rotating_ && input_control_ && native_key == 'L') begin_sun_rotation();
        if (!key.repeat) handle_key(native_key);
    }

public:
    void detach(std::string_view viewport_id)
    {
        std::lock_guard lock(bounds_mutex_);
        if (shared_texture_)
        {
            const auto found = shared_surfaces_.find(std::string{viewport_id});
            if (found == shared_surfaces_.end()) return;
            found->second.attached = false;
            found->second.visible = false;
            found->second.destroy_dirty = found->second.output_created;
            found->second.create_dirty = false;
            attached_ = std::any_of(shared_surfaces_.begin(), shared_surfaces_.end(),
                                    [](const auto& entry) { return entry.second.attached; });
            if (viewport_id_ == viewport_id)
                for (auto& [id, surface] : shared_surfaces_)
                    if (surface.attached)
                    {
                        activate_shared_surface_locked(surface);
                        break;
                    }
            return;
        }
        if (viewport_id != viewport_id_) return;
        attached_ = false;
        parent_ = nullptr;
        bounds_dirty_ = true;
        if (window_) ShowWindow(window_, SW_HIDE);
    }

    void resize(std::string_view viewport_id, std::int32_t x, std::int32_t y, std::uint32_t width, std::uint32_t height)
    {
        std::lock_guard lock(bounds_mutex_);
        if (shared_texture_)
        {
            const auto found = shared_surfaces_.find(std::string{viewport_id});
            if (found == shared_surfaces_.end()) return;
            auto& surface = found->second;
            surface.width = std::max(1u, width);
            surface.height = std::max(1u, height);
            surface.resize_dirty = surface.output_created;
            if (viewport_id_ == viewport_id) activate_shared_surface_locked(surface);
            terrain_hover_dirty_ = true;
            return;
        }
        if (viewport_id != viewport_id_) return;
        x_ = x;
        y_ = y;
        width_ = std::max(static_cast<std::uint32_t>(arc::editor::defaults::native_viewport_min_dimension), width);
        height_ = std::max(static_cast<std::uint32_t>(arc::editor::defaults::native_viewport_min_dimension), height);
        bounds_dirty_ = true;
        terrain_hover_dirty_ = true;
    }

    void stop()
    {
        if (!running_.exchange(false)) return;
        if (window_) PostMessageW(window_, WM_CLOSE, 0, 0);
        if (render_task_.valid()) (void)render_task_.wait_result();
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
                    TRACKMOUSEEVENT tracking{sizeof(TRACKMOUSEEVENT), TME_LEAVE, window, 0};
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
                    .zoom = static_cast<float>(GET_WHEEL_DELTA_WPARAM(wparam)) / static_cast<float>(WHEEL_DELTA)});
                return 0;
            case WM_CAPTURECHANGED:
                if (manipulating_) cancel_manipulation();
                if (terrain_stroking_) finish_terrain_stroke(false, drag_x_, drag_y_);
                dragging_ = false;
                drag_button_ = drag_button::none;
                camera_drag_mode_ = camera_drag_mode::none;
                selection_candidate_ = false;
                camera_drag_started_ = false;
                return 0;
            case WM_KILLFOCUS:
                if (sun_rotating_) finish_sun_rotation(false);
                return 0;
            case WM_KEYDOWN:
                if (wparam == VK_ESCAPE && sun_rotating_)
                {
                    finish_sun_rotation(false);
                    return 0;
                }
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
                if (!sun_rotating_ && sun_shortcut_down()) begin_sun_rotation();
                if (!sun_rotating_) handle_key(wparam);
                return 0;
            case WM_KEYUP:
                if (sun_rotating_ &&
                    (wparam == 'L' || wparam == VK_CONTROL || wparam == VK_LCONTROL || wparam == VK_RCONTROL))
                    finish_sun_rotation(true);
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

    enum class camera_drag_mode
    {
        none,
        orbit,
        pan,
        look,
        pending_left_look
    };

    struct bounds
    {
        HWND parent{};
        std::int32_t x{};
        std::int32_t y{};
        std::uint32_t width{1};
        std::uint32_t height{1};
    };

    struct pending_release
    {
        std::string viewport_id;
        std::uint64_t generation{};
        std::uint64_t frame_id{};
    };

    struct shared_surface_state
    {
        std::string viewport_id;
        DWORD consumer_process_id{};
        std::uint32_t width{1};
        std::uint32_t height{1};
        bool attached{};
        bool visible{true};
        bool output_created{};
        bool create_dirty{};
        bool resize_dirty{};
        bool visibility_dirty{};
        bool destroy_dirty{};
        std::uint64_t frame_index{};
    };

    struct shared_render_target
    {
        std::string viewport_id;
        DWORD consumer_process_id{};
        std::uint32_t width{1};
        std::uint32_t height{1};
        std::uint64_t frame_index{};
    };

    void activate_shared_surface_locked(const shared_surface_state& surface)
    {
        viewport_id_ = surface.viewport_id;
        consumer_process_id_ = surface.consumer_process_id;
        width_ = surface.width;
        height_ = surface.height;
    }

    void activate_interaction_surface(std::string_view viewport_id)
    {
        if (!shared_texture_) return;
        std::lock_guard lock(bounds_mutex_);
        const auto found = shared_surfaces_.find(std::string{viewport_id});
        if (found != shared_surfaces_.end() && found->second.attached) activate_shared_surface_locked(found->second);
    }

    bounds current_bounds()
    {
        std::lock_guard lock(bounds_mutex_);
        bounds_dirty_ = false;
        return {.parent = parent_, .x = x_, .y = y_, .width = width_, .height = height_};
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
        if (!register_window_class() || value.parent == nullptr) return false;

        window_ = CreateWindowExW(WS_EX_NOACTIVATE, L"ArcEditor2NativeViewport", L"ARC Viewport",
                                  WS_CHILD | WS_VISIBLE | WS_CLIPSIBLINGS | WS_CLIPCHILDREN, value.x, value.y,
                                  static_cast<int>(value.width), static_cast<int>(value.height), value.parent, nullptr,
                                  GetModuleHandleW(nullptr), this);
        return window_ != nullptr;
    }

    bool create_backend()
    {
        arc::render::vulkan::vulkan_backend_config config{};
        if (shared_texture_)
        {
            config.viewport_output = arc::render::viewport_output_type::shared_texture;
        }
        else
        {
            config.instance_extensions = {VK_KHR_SURFACE_EXTENSION_NAME, VK_KHR_WIN32_SURFACE_EXTENSION_NAME};
            config.create_surface = create_win32_surface;
            config.surface_user_data = window_;
        }

        auto result = arc::render::vulkan::create_vulkan_backend(config);
        if (!result)
        {
            std::cerr << "arc_host_process Vulkan backend error: " << result.error().message << '\n';
            return false;
        }

        std::lock_guard lock(host_mutex_);
        host_->renderer_service().set_backend(std::move(result).value());
        backend_ = host_->renderer_service().backend();
        return backend_ != nullptr;
    }

    void apply_bounds(const bounds& value)
    {
        if (!window_ || value.parent == nullptr) return;
        if (GetParent(window_) != value.parent) SetParent(window_, value.parent);
        SetWindowPos(window_, HWND_TOP, value.x, value.y, static_cast<int>(value.width), static_cast<int>(value.height),
                     SWP_SHOWWINDOW | SWP_NOACTIVATE);
    }

    bool recover_backend_if_needed(const std::string& message)
    {
        const auto now = std::chrono::steady_clock::now();
        if (message != last_render_error_ || now - last_render_error_time_ >= std::chrono::seconds{5})
        {
            std::cerr << "arc_host_process native viewport render error: " << message << '\n';
            last_render_error_ = message;
            last_render_error_time_ = now;
        }
        const bool recreate_backend = message.find("backend recreation required") != std::string::npos;
        if (!recreate_backend || now - last_backend_recovery_attempt_ < std::chrono::seconds{2}) return false;

        last_backend_recovery_attempt_ = now;
        {
            std::lock_guard lock(host_mutex_);
            backend_ = nullptr;
            host_->renderer_service().set_backend(nullptr);
        }
        if (!create_backend())
        {
            running_ = false;
            return true;
        }
        if (shared_texture_)
        {
            std::lock_guard lock(bounds_mutex_);
            for (auto& [id, surface] : shared_surfaces_)
            {
                surface.output_created = false;
                surface.create_dirty = surface.attached;
                surface.resize_dirty = false;
                surface.visibility_dirty = false;
            }
        }
        return true;
    }

    void render_once(const bounds& value)
    {
        if (!backend_) return;

        bool rendered{};
        std::string message;
        update_terrain_hover();
        {
            std::lock_guard lock(host_mutex_);
            host_->request_viewport(arc::editor::host_viewport_request{.viewport_id = viewport_id_,
                                                                       .frame_index = frame_index_++,
                                                                       .width = value.width,
                                                                       .height = value.height});
            auto present = backend_->present_surface_frame(value.width, value.height);
            rendered = present.has_value();
            if (!rendered) message = std::move(present.error().message);
        }
        if (rendered)
        {
            last_render_error_.clear();
            return;
        }
        if (message.empty()) return;
        (void)recover_backend_if_needed(message);
    }

    void render_shared_once(const shared_render_target& target)
    {
        if (!backend_) return;
        bool rendered{};
        std::string message;
        {
            std::lock_guard lock(host_mutex_);
            host_->request_viewport(arc::editor::host_viewport_request{.viewport_id = target.viewport_id,
                                                                       .frame_index = target.frame_index,
                                                                       .width = target.width,
                                                                       .height = target.height});
            auto present = backend_->present_viewport_output(target.viewport_id);
            rendered = present.has_value();
            if (!rendered) message = std::move(present.error().message);
        }
        if (rendered)
        {
            last_render_error_.clear();
            publish_ready_frame(target.viewport_id, target.consumer_process_id);
            return;
        }
        if (message.empty()) return;
        (void)recover_backend_if_needed(message);
    }

    void publish_ready_frame(std::string_view viewport_id, DWORD consumer_process_id)
    {
        const auto polled = backend_->poll_viewport_output(viewport_id);
        if (!polled || !polled.value()) return;
        const auto& frame = *polled.value();
        HANDLE consumer = OpenProcess(PROCESS_DUP_HANDLE, FALSE, consumer_process_id);
        HANDLE duplicate{};
        const auto source = reinterpret_cast<HANDLE>(static_cast<std::uintptr_t>(frame.texture.payload));
        if (consumer == nullptr ||
            !DuplicateHandle(GetCurrentProcess(), source, consumer, &duplicate, 0, FALSE, DUPLICATE_SAME_ACCESS))
        {
            if (consumer != nullptr) CloseHandle(consumer);
            backend_->release_viewport_frame(frame.viewport_id, frame.generation, frame.frame_id);
            std::cerr << "arc_host_process shared viewport handle duplication failed: " << GetLastError() << '\n';
            return;
        }
        CloseHandle(consumer);
        std::ostringstream handle;
        handle << "0x" << std::hex << std::setw(sizeof(std::uintptr_t) * 2) << std::setfill('0')
               << reinterpret_cast<std::uintptr_t>(duplicate);
        const auto payload =
            "{\"viewportId\":\"" + frame.viewport_id + "\",\"frameId\":" + std::to_string(frame.frame_id) +
            ",\"generation\":" + std::to_string(frame.generation) + ",\"width\":" + std::to_string(frame.width) +
            ",\"height\":" + std::to_string(frame.height) +
            ",\"format\":\"bgra\",\"handleType\":\"win32NtHandle\",\"handle\":\"" + handle.str() +
            "\",\"producerComplete\":true}";
        const arc::editor::host_event event{.sequence = shared_event_sequence_++,
                                            .event_type = arc::editor::host_event_type::viewport_frame_ready,
                                            .message = "Shared viewport frame ready",
                                            .payload_json = payload};
        std::lock_guard output_lock(output_mutex_);
        std::cout << arc::editor::to_json(event) << '\n';
        std::cout.flush();
    }

    void begin_drag(HWND window, UINT message, int x, int y)
    {
        if (sun_rotating_) return;
        dragging_ = true;
        drag_start_x_ = x;
        drag_start_y_ = y;
        drag_x_ = x;
        drag_y_ = y;
        drag_distance_ = 0;
        camera_drag_started_ = false;
        drag_button_ = message == WM_MBUTTONDOWN   ? drag_button::middle
                       : message == WM_RBUTTONDOWN ? drag_button::right
                                                   : drag_button::left;
        const bool alt = shared_texture_
                             ? input_alt_
                             : (GetKeyState(VK_MENU) & arc::editor::defaults::viewport_modifier_key_down_mask) != 0;
        const bool shift = shared_texture_
                               ? input_shift_
                               : (GetKeyState(VK_SHIFT) & arc::editor::defaults::viewport_modifier_key_down_mask) != 0;
        camera_drag_mode_ = camera_drag_mode::none;
        if (alt && drag_button_ == drag_button::left)
            camera_drag_mode_ = camera_drag_mode::orbit;
        else if (shift || drag_button_ == drag_button::middle)
            camera_drag_mode_ = camera_drag_mode::pan;
        else if (drag_button_ == drag_button::right)
            camera_drag_mode_ = camera_drag_mode::look;
        else if (drag_button_ == drag_button::left)
            camera_drag_mode_ = camera_drag_mode::pending_left_look;
        selection_candidate_ = drag_button_ == drag_button::left && !alt && !shift;
        if (drag_button_ == drag_button::left && !alt)
        {
            if (!begin_terrain_stroke(x, y))
            {
                if (!shift) begin_manipulation(x, y);
            }
            else
                selection_candidate_ = false;
        }
        if (window)
        {
            SetCapture(window);
            SetFocus(window);
        }
    }

    void update_drag(int x, int y)
    {
        const int pointer_delta_x = x - pointer_x_;
        const int pointer_delta_y = y - pointer_y_;
        pointer_x_ = x;
        pointer_y_ = y;
        pointer_inside_ = true;
        terrain_hover_dirty_ = true;
        if (sun_rotating_)
        {
            if (pointer_delta_x != 0 || pointer_delta_y != 0) update_sun_rotation(pointer_delta_x, pointer_delta_y);
            return;
        }
        if (!dragging_)
        {
            bool terrain_mode{};
            {
                std::lock_guard lock(host_mutex_);
                terrain_mode = host_->viewport_tool_state().tool == arc::editor::host_viewport_tool::terrain;
            }
            if (!terrain_mode) update_gizmo_hover(x, y);
            return;
        }

        const int delta_x = x - drag_x_;
        const int delta_y = y - drag_y_;
        drag_x_ = x;
        drag_y_ = y;
        if (delta_x == 0 && delta_y == 0) return;
        drag_distance_ += std::abs(delta_x) + std::abs(delta_y);
        if (drag_distance_ > arc::editor::defaults::viewport_click_movement_threshold) selection_candidate_ = false;

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

        arc::editor::host_viewport_camera_input_command input;
        if (camera_drag_mode_ == camera_drag_mode::orbit)
        {
            input.orbit_x = static_cast<float>(delta_x);
            input.orbit_y = static_cast<float>(delta_y);
        }
        else if (camera_drag_mode_ == camera_drag_mode::pan)
        {
            input.pan_x = static_cast<float>(delta_x);
            input.pan_y = static_cast<float>(delta_y);
        }
        else if (camera_drag_mode_ == camera_drag_mode::look)
        {
            input.look_x = static_cast<float>(delta_x);
            input.look_y = static_cast<float>(delta_y);
        }
        else if (camera_drag_mode_ == camera_drag_mode::pending_left_look &&
                 drag_distance_ > arc::editor::defaults::viewport_click_movement_threshold)
        {
            // An unmodified left press remains a selection candidate until it
            // crosses the click threshold. At that point navigation owns the
            // gesture and receives the complete movement since mouse-down.
            input.look_x = static_cast<float>(camera_drag_started_ ? delta_x : x - drag_start_x_);
            input.look_y = static_cast<float>(camera_drag_started_ ? delta_y : y - drag_start_y_);
            camera_drag_started_ = true;
        }
        if (input.orbit_x != 0.0f || input.orbit_y != 0.0f || input.look_x != 0.0f || input.look_y != 0.0f ||
            input.pan_x != 0.0f || input.pan_y != 0.0f)
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
        camera_drag_mode_ = camera_drag_mode::none;
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
        if (GetCapture() == window) ReleaseCapture();
        if (completed_button == drag_button::left && selection_candidate_) send_pick(std::max(0, x), std::max(0, y));
        selection_candidate_ = false;
        camera_drag_started_ = false;
    }

    void cancel_pointer_interaction()
    {
        if (sun_rotating_) finish_sun_rotation(false);
        if (manipulating_) cancel_manipulation();
        if (terrain_stroking_) finish_terrain_stroke(false, drag_x_, drag_y_);
        dragging_ = false;
        drag_button_ = drag_button::none;
        camera_drag_mode_ = camera_drag_mode::none;
        selection_candidate_ = false;
        camera_drag_started_ = false;
        pointer_inside_ = false;
        clear_terrain_hover();
    }

    void send_pick(int x, int y)
    {
        std::lock_guard lock(host_mutex_);
        host_->execute(arc::editor::host_viewport_pick_command{
            .viewport_id = viewport_id_, .x = static_cast<std::uint32_t>(x), .y = static_cast<std::uint32_t>(y)});
    }

    static arc::editor::editor_tool editor_tool_for(arc::editor::host_viewport_tool tool) noexcept
    {
        switch (tool)
        {
            case arc::editor::host_viewport_tool::translate:
                return arc::editor::editor_tool::translate;
            case arc::editor::host_viewport_tool::rotate:
                return arc::editor::editor_tool::rotate;
            case arc::editor::host_viewport_tool::scale:
                return arc::editor::editor_tool::scale;
            case arc::editor::host_viewport_tool::select:
                return arc::editor::editor_tool::select;
            case arc::editor::host_viewport_tool::terrain:
                return arc::editor::editor_tool::select;
        }
        return arc::editor::editor_tool::select;
    }

    bool begin_terrain_stroke(int x, int y)
    {
        std::lock_guard lock(host_mutex_);
        if (host_->viewport_tool_state().tool != arc::editor::host_viewport_tool::terrain) return false;
        const auto snapshot = host_->selected_entity_snapshot();
        if (!snapshot.entity.valid() || !snapshot.terrain || !host_->terrain_tool_snapshot().hover_visible)
            return false;
        const auto transaction = ++next_manipulation_transaction_;
        const bool invert = shared_texture_
                                ? input_shift_
                                : (GetKeyState(VK_SHIFT) & arc::editor::defaults::viewport_modifier_key_down_mask) != 0;
        const auto response = host_->execute(arc::editor::host_command_envelope{
            .command_type = "terrain.stroke",
            .payload =
                arc::editor::host_terrain_stroke_command{snapshot.entity, static_cast<std::uint32_t>(std::max(0, x)),
                                                         static_cast<std::uint32_t>(std::max(0, y)),
                                                         arc::editor::host_edit_phase::begin, invert},
            .edit = arc::editor::host_edit_transaction{transaction, arc::editor::host_edit_phase::begin,
                                                       "Terrain Stroke"}});
        if (!response.succeeded || response.payload_json.find("\"hit\":true") == std::string::npos)
        {
            host_->execute(arc::editor::host_command_envelope{
                .command_type = "terrain.stroke",
                .payload = arc::editor::host_terrain_stroke_command{snapshot.entity,
                                                                    static_cast<std::uint32_t>(std::max(0, x)),
                                                                    static_cast<std::uint32_t>(std::max(0, y)),
                                                                    arc::editor::host_edit_phase::cancel, false},
                .edit = arc::editor::host_edit_transaction{transaction, arc::editor::host_edit_phase::cancel,
                                                           "Terrain Stroke"}});
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
        if (!terrain_hover_dirty_ || dragging_ || !pointer_inside_) return;
        terrain_hover_dirty_ = false;
        std::lock_guard lock(host_mutex_);
        if (host_->viewport_tool_state().tool != arc::editor::host_viewport_tool::terrain) return;
        const auto snapshot = host_->selected_entity_snapshot();
        if (!snapshot.entity.valid() || !snapshot.terrain) return;
        host_->execute(
            arc::editor::host_terrain_hover_command{.entity = snapshot.entity,
                                                    .x = static_cast<std::uint32_t>(std::max(0, pointer_x_)),
                                                    .y = static_cast<std::uint32_t>(std::max(0, pointer_y_))});
    }

    void clear_terrain_hover()
    {
        std::lock_guard lock(host_mutex_);
        const auto snapshot = host_->selected_entity_snapshot();
        if (!snapshot.entity.valid() || !snapshot.terrain) return;
        host_->execute(arc::editor::host_terrain_hover_command{.entity = snapshot.entity, .clear = true});
    }

    void update_terrain_stroke(int x, int y)
    {
        if (terrain_last_preview_frame_ == frame_index_) return;
        terrain_last_preview_frame_ = frame_index_;
        std::lock_guard lock(host_mutex_);
        const bool invert = (GetKeyState(VK_SHIFT) & arc::editor::defaults::viewport_modifier_key_down_mask) != 0;
        host_->execute(arc::editor::host_command_envelope{
            .command_type = "terrain.stroke",
            .payload =
                arc::editor::host_terrain_stroke_command{terrain_entity_, static_cast<std::uint32_t>(std::max(0, x)),
                                                         static_cast<std::uint32_t>(std::max(0, y)),
                                                         arc::editor::host_edit_phase::update, invert},
            .edit = arc::editor::host_edit_transaction{terrain_transaction_, arc::editor::host_edit_phase::update,
                                                       "Terrain Stroke"}});
    }

    void finish_terrain_stroke(bool commit, int x, int y)
    {
        const auto phase = commit ? arc::editor::host_edit_phase::commit : arc::editor::host_edit_phase::cancel;
        std::lock_guard lock(host_mutex_);
        host_->execute(arc::editor::host_command_envelope{
            .command_type = "terrain.stroke",
            .payload =
                arc::editor::host_terrain_stroke_command{terrain_entity_, static_cast<std::uint32_t>(std::max(0, x)),
                                                         static_cast<std::uint32_t>(std::max(0, y)), phase, false},
            .edit = arc::editor::host_edit_transaction{terrain_transaction_, phase, "Terrain Stroke"}});
        terrain_stroking_ = false;
        terrain_entity_ = {};
    }

    arc::editor::editor_gizmo_context gizmo_context() const
    {
        const auto& tool = host_->viewport_tool_state();
        std::lock_guard bounds_lock(bounds_mutex_);
        return {.tool = editor_tool_for(tool.tool),
                .coordinate_space = tool.coordinate_space == arc::editor::host_coordinate_space::local
                                        ? arc::editor::gizmo_coordinate_space::local
                                        : arc::editor::gizmo_coordinate_space::world,
                .highlighted_axis = active_axis_,
                .viewport_width = width_,
                .viewport_height = height_};
    }

    void update_gizmo_hover(int x, int y)
    {
        std::lock_guard lock(host_mutex_);
        const auto& state = host_->scene_state();
        const auto context = gizmo_context();
        const auto axis = arc::editor::hit_test_editor_gizmo(state.scene, state.selected_entity, state.camera_entity,
                                                             context, static_cast<float>(x), static_cast<float>(y));
        active_axis_ = axis;
        host_->set_viewport_gizmo_highlight(axis);
    }

    void begin_manipulation(int x, int y)
    {
        std::lock_guard lock(host_mutex_);
        const auto& state = host_->scene_state();
        const auto context = gizmo_context();
        const auto axis = arc::editor::hit_test_editor_gizmo(state.scene, state.selected_entity, state.camera_entity,
                                                             context, static_cast<float>(x), static_cast<float>(y));
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
        const bool uniform_scale = axis == arc::editor::gizmo_axis::all;
        if (!uniform_scale) manipulation_local_axis_[static_cast<std::size_t>(axis) - 1u] = 1.0f;
        manipulation_rotation_axis_ = manipulation_local_axis_;
        manipulation_screen_direction_ = uniform_scale ? arc::math::normalize(arc::math::vector2f{1.0f, -1.0f})
                                         : axis == arc::editor::gizmo_axis::x ? arc::math::vector2f{1.0f, 0.0f}
                                                                              : arc::math::vector2f{0.0f, -1.0f};
        manipulation_world_units_per_pixel_ = 0.02f;
        const auto selected_entity = arc::ecs::entity{snapshot.entity.index, snapshot.entity.generation};
        const auto* selected_transform = state.scene.try_get<arc::scene::transform_component>(selected_entity);
        const auto* camera = state.scene.try_get<arc::scene::camera_component>(state.camera_entity);
        const auto* camera_transform = state.scene.try_get<arc::scene::transform_component>(state.camera_entity);
        if (selected_transform && camera && camera_transform && !uniform_scale)
        {
            const std::size_t axis_index = static_cast<std::size_t>(axis) - 1u;
            arc::math::vector3f world_axis{};
            world_axis[axis_index] = 1.0f;
            if (context.coordinate_space == arc::editor::gizmo_coordinate_space::local)
                world_axis = arc::math::normalize(arc::math::vector3f{selected_transform->world(0, axis_index),
                                                                      selected_transform->world(1, axis_index),
                                                                      selected_transform->world(2, axis_index)});
            manipulation_local_axis_ = world_axis;
            manipulation_rotation_axis_ = world_axis;
            auto parent = state.scene.try_get<arc::scene::hierarchy_component>(selected_entity)
                              ? state.scene.get<arc::scene::hierarchy_component>(selected_entity).parent
                              : arc::ecs::entity{};
            while (state.scene.alive(parent) && !state.scene.has<arc::scene::transform_component>(parent))
            {
                const auto* hierarchy = state.scene.try_get<arc::scene::hierarchy_component>(parent);
                parent = hierarchy ? hierarchy->parent : arc::ecs::entity{};
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
            manipulation_world_units_per_pixel_ =
                arc::editor::editor_gizmo_world_scale(*camera, *camera_transform,
                                                      arc::scene::world_position(*selected_transform), height_) /
                arc::editor::editor_gizmo_pixel_length;
            manipulation_rotation_is_local_ = context.coordinate_space == arc::editor::gizmo_coordinate_space::local;
            arc::editor::editor_gizmo_drag_direction(state.scene, state.selected_entity, state.camera_entity, context,
                                                     axis, static_cast<float>(x), static_cast<float>(y),
                                                     manipulation_screen_direction_);
        }
        host_->set_viewport_gizmo_highlight(axis);
        host_->execute(arc::editor::host_command_envelope{
            .command_type = "entity.setTransform",
            .payload = arc::editor::host_set_transform_command{manipulation_entity_, manipulation_original_},
            .edit = arc::editor::host_edit_transaction{manipulation_transaction_, arc::editor::host_edit_phase::begin,
                                                       "Gizmo Transform"}});
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
        if (axis == 0)
            value.x = next;
        else if (axis == 1)
            value.y = next;
        else
            value.z = next;
    }

    static arc::math::quatf multiply_rotation(const arc::math::quatf& lhs, const arc::math::quatf& rhs) noexcept
    {
        return arc::math::normalize(
            arc::math::quatf{lhs[3] * rhs[0] + lhs[0] * rhs[3] + lhs[1] * rhs[2] - lhs[2] * rhs[1],
                             lhs[3] * rhs[1] - lhs[0] * rhs[2] + lhs[1] * rhs[3] + lhs[2] * rhs[0],
                             lhs[3] * rhs[2] + lhs[0] * rhs[1] - lhs[1] * rhs[0] + lhs[2] * rhs[3],
                             lhs[3] * rhs[3] - lhs[0] * rhs[0] - lhs[1] * rhs[1] - lhs[2] * rhs[2]});
    }

    void update_manipulation(int x, int y)
    {
        std::lock_guard lock(host_mutex_);
        const auto& tool = host_->viewport_tool_state();
        const arc::math::vector2f pointer_delta{static_cast<float>(x - manipulation_start_x_),
                                                static_cast<float>(y - manipulation_start_y_)};
        const float pixel_delta = arc::math::dot(pointer_delta, manipulation_screen_direction_);
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
            const float factor = std::exp(pixel_delta * 0.01f);
            if (active_axis_ == arc::editor::gizmo_axis::all)
            {
                for (std::size_t component = 0; component < 3u; ++component)
                {
                    float value = std::max(0.001f, axis_value(manipulation_original_.scale, component) * factor);
                    if (tool.snapping) value = std::max(0.001f, snapped(value, tool.scale_snap));
                    set_axis_value(manipulation_current_.scale, component, value);
                }
            }
            else
            {
                float value = std::max(0.001f, axis_value(manipulation_original_.scale, axis) * factor);
                if (tool.snapping) value = std::max(0.001f, snapped(value, tool.scale_snap));
                set_axis_value(manipulation_current_.scale, axis, value);
            }
        }
        else if (tool.tool == arc::editor::host_viewport_tool::rotate)
        {
            float degrees = pixel_delta * 0.35f;
            if (tool.snapping) degrees = snapped(degrees, tool.rotation_snap_degrees);
            arc::math::vector3f local_axis{};
            local_axis[axis] = 1.0f;
            const auto delta =
                arc::math::from_axis_angle(manipulation_rotation_is_local_ ? local_axis : manipulation_rotation_axis_,
                                           arc::math::to_radians(degrees));
            const arc::math::quatf original{manipulation_original_.rotation.x, manipulation_original_.rotation.y,
                                            manipulation_original_.rotation.z, manipulation_original_.rotation.w};
            const auto result = manipulation_rotation_is_local_ ? multiply_rotation(original, delta)
                                                                : multiply_rotation(delta, original);
            manipulation_current_.rotation = {result[0], result[1], result[2], result[3]};
        }
        host_->execute(arc::editor::host_command_envelope{
            .command_type = "entity.setTransform",
            .payload = arc::editor::host_set_transform_command{manipulation_entity_, manipulation_current_},
            .edit = arc::editor::host_edit_transaction{manipulation_transaction_, arc::editor::host_edit_phase::update,
                                                       "Gizmo Transform"}});
    }

    void finish_manipulation(bool commit)
    {
        const auto phase = commit ? arc::editor::host_edit_phase::commit : arc::editor::host_edit_phase::cancel;
        {
            std::lock_guard lock(host_mutex_);
            host_->execute(arc::editor::host_command_envelope{
                .command_type = "entity.setTransform",
                .payload = arc::editor::host_set_transform_command{manipulation_entity_, manipulation_current_},
                .edit = arc::editor::host_edit_transaction{manipulation_transaction_, phase, "Gizmo Transform"}});
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
            return send_camera_input(arc::editor::host_viewport_camera_input_command{.focus_selected = true});
        if (key == VK_DELETE ||
            ((shared_texture_ ? input_control_ : (GetKeyState(VK_CONTROL) & 0x8000) != 0) && key == 'D'))
        {
            std::lock_guard lock(host_mutex_);
            const auto selected = host_->selected_entity_snapshot().entity;
            if (!selected.valid()) return;
            if (key == VK_DELETE)
                host_->execute(arc::editor::host_delete_entity_command{selected});
            else
                host_->execute(arc::editor::host_duplicate_entity_command{selected});
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
                snapshot.entity, terrain.brush_tool, std::clamp(terrain.brush_radius * multiplier, 0.25f, 128.0f),
                terrain.brush_strength, terrain.brush_falloff, terrain.active_layer});
            return;
        }
        if ((shared_texture_
                 ? input_control_
                 : (GetKeyState(VK_CONTROL) & arc::editor::defaults::viewport_modifier_key_down_mask) != 0) ||
            (shared_texture_ ? input_alt_
                             : (GetKeyState(VK_MENU) & arc::editor::defaults::viewport_modifier_key_down_mask) != 0))
            return;
        const auto editor_tool = arc::editor::editor_tool_from_shortcut(static_cast<std::uint32_t>(key));
        if (!editor_tool) return;
        const auto tool = *editor_tool == arc::editor::editor_tool::select ? arc::editor::host_viewport_tool::select
                          : *editor_tool == arc::editor::editor_tool::translate
                              ? arc::editor::host_viewport_tool::translate
                          : *editor_tool == arc::editor::editor_tool::rotate ? arc::editor::host_viewport_tool::rotate
                                                                             : arc::editor::host_viewport_tool::scale;
        std::lock_guard lock(host_mutex_);
        auto command = host_->viewport_tool_state();
        command.tool = tool;
        host_->execute(command);
    }

    bool sun_shortcut_down() const noexcept
    {
        if (shared_texture_) return input_control_;
        return (GetKeyState(VK_CONTROL) & arc::editor::defaults::viewport_modifier_key_down_mask) != 0 &&
               (GetKeyState('L') & arc::editor::defaults::viewport_modifier_key_down_mask) != 0;
    }

    void begin_sun_rotation()
    {
        if (dragging_ || manipulating_ || terrain_stroking_) return;
        POINT pointer{};
        if (window_ && GetCursorPos(&pointer) && ScreenToClient(window_, &pointer))
        {
            pointer_x_ = pointer.x;
            pointer_y_ = pointer.y;
            pointer_inside_ = true;
        }
        std::lock_guard lock(host_mutex_);
        auto& state = host_->scene_state();
        if (!state.scene.alive(state.sun_entity)) return;
        auto* transform = state.scene.try_get<arc::scene::transform_component>(state.sun_entity);
        if (!transform) return;

        const auto transaction = ++next_manipulation_transaction_;
        const auto begin = host_->execute(arc::editor::host_history_begin_transaction_command{
            .id = transaction, .label = "Rotate Directional Light"});
        if (!begin.succeeded) return;

        sun_rotating_ = true;
        sun_entity_ = {state.sun_entity.index, state.sun_entity.generation};
        sun_transaction_ = transaction;
        sun_controller_.synchronize_from(*transform);
        if (state.scene.alive(state.world_environment_entity))
        {
            if (auto settings =
                    arc::scene::read_world_environment_settings(state.scene, state.world_environment_entity))
            {
                settings->celestial.sun_mode = arc::scene::sun_position_mode::manual_light;
                settings->celestial.playing = false;
                settings->celestial.automatic_sun_light = false;
                arc::scene::set_world_environment_settings(state.scene, state.world_environment_entity, *settings);
            }
        }
    }

    void update_sun_rotation(int delta_x, int delta_y)
    {
        std::lock_guard lock(host_mutex_);
        auto& state = host_->scene_state();
        const auto entity = arc::ecs::entity{sun_entity_.index, sun_entity_.generation};
        auto* transform = state.scene.try_get<arc::scene::transform_component>(entity);
        if (!transform)
        {
            finish_sun_rotation_locked(false);
            return;
        }
        sun_controller_.rotate(static_cast<float>(delta_x), static_cast<float>(delta_y));
        sun_controller_.apply_to(*transform);
        arc::scene::mark_transform_subtree_dirty(state.scene, entity);
        arc::scene::update_world_transforms(state.scene);
    }

    void finish_sun_rotation(bool commit)
    {
        std::lock_guard lock(host_mutex_);
        finish_sun_rotation_locked(commit);
    }

    void finish_sun_rotation_locked(bool commit)
    {
        if (!sun_rotating_) return;
        auto& state = host_->scene_state();
        if (!commit)
        {
            host_->execute(arc::editor::host_history_cancel_transaction_command{.id = sun_transaction_});
        }
        else if (state.scene.alive(state.world_environment_entity))
        {
            if (const auto settings =
                    arc::scene::read_world_environment_settings(state.scene, state.world_environment_entity))
            {
                host_->execute(arc::editor::host_command_envelope{
                    .command_type = "environment.update",
                    .payload =
                        arc::editor::host_set_world_environment_command{
                            .environment = arc::editor::to_host_world_environment_snapshot(
                                {state.world_environment_entity.index, state.world_environment_entity.generation},
                                *settings, state.world_environment_hdri_path)},
                    .edit = arc::editor::host_edit_transaction{sun_transaction_, arc::editor::host_edit_phase::commit,
                                                               "Rotate Directional Light"}});
            }
            else
            {
                host_->execute(arc::editor::host_history_cancel_transaction_command{.id = sun_transaction_});
            }
        }
        else
        {
            const auto entity = arc::ecs::entity{sun_entity_.index, sun_entity_.generation};
            if (const auto* transform = state.scene.try_get<arc::scene::transform_component>(entity))
            {
                host_->execute(arc::editor::host_command_envelope{
                    .command_type = "entity.setTransform",
                    .payload =
                        arc::editor::host_set_transform_command{
                            .entity = sun_entity_,
                            .transform = {.position = {transform->position[0], transform->position[1],
                                                       transform->position[2]},
                                          .rotation = {transform->rotation[0], transform->rotation[1],
                                                       transform->rotation[2], transform->rotation[3]},
                                          .scale = {transform->scale[0], transform->scale[1], transform->scale[2]}}},
                    .edit = arc::editor::host_edit_transaction{sun_transaction_, arc::editor::host_edit_phase::commit,
                                                               "Rotate Directional Light"}});
            }
            else
            {
                host_->execute(arc::editor::host_history_cancel_transaction_command{.id = sun_transaction_});
            }
        }
        sun_rotating_ = false;
        sun_entity_ = {};
        sun_transaction_ = 0;
    }

    void send_camera_input(const arc::editor::host_viewport_camera_input_command& input)
    {
        auto routed_input = input;
        routed_input.viewport_id = viewport_id_;
        std::lock_guard lock(host_mutex_);
        host_->execute(
            arc::editor::host_command_envelope{.command_type = "viewport.cameraInput", .payload = routed_input});
    }

    std::string update_shared_outputs(std::vector<shared_render_target>& render_targets)
    {
        std::lock_guard lock(bounds_mutex_);
        std::string first_error;
        for (const auto& release : pending_releases_)
            if (backend_) backend_->release_viewport_frame(release.viewport_id, release.generation, release.frame_id);
        pending_releases_.clear();

        for (auto& [id, surface] : shared_surfaces_)
        {
            if (surface.destroy_dirty && backend_)
            {
                backend_->destroy_viewport_output(id);
                surface.destroy_dirty = false;
                surface.output_created = false;
            }
            if (surface.create_dirty && surface.attached && backend_)
            {
                const auto created =
                    backend_->create_viewport_output({.id = id,
                                                      .type = arc::render::viewport_output_type::shared_texture,
                                                      .width = surface.width,
                                                      .height = surface.height,
                                                      .visible = surface.visible});
                if (!created)
                {
                    if (first_error.empty()) first_error = created.error().message;
                    surface.output_created = false;
                }
                else
                    surface.output_created = true;
                surface.create_dirty = false;
            }
            if (surface.resize_dirty && surface.output_created && backend_)
            {
                const auto resized = backend_->resize_viewport_output(id, surface.width, surface.height);
                if (!resized && first_error.empty()) first_error = resized.error().message;
                surface.resize_dirty = false;
            }
            if (surface.visibility_dirty && surface.output_created && backend_)
            {
                backend_->set_viewport_output_visible(id, surface.visible);
                surface.visibility_dirty = false;
            }
            if (surface.attached && surface.visible && surface.output_created)
                render_targets.push_back({.viewport_id = id,
                                          .consumer_process_id = surface.consumer_process_id,
                                          .width = surface.width,
                                          .height = surface.height,
                                          .frame_index = surface.frame_index++});
        }
        attached_ = std::any_of(shared_surfaces_.begin(), shared_surfaces_.end(),
                                [](const auto& entry) { return entry.second.attached; });
        return first_error;
    }

    void render_loop()
    {
        auto value = current_bounds();
        if (!shared_texture_ && !create_window(value))
        {
            std::cerr << "arc_host_process failed to create native viewport window\n";
            signal_setup("Failed to create native viewport window");
            running_ = false;
            return;
        }
        if (!shared_texture_) apply_bounds(value);
        if (!create_backend())
        {
            signal_setup("Failed to create Vulkan viewport backend");
            running_ = false;
            return;
        }
        if (shared_texture_)
        {
            std::vector<shared_render_target> initial_targets;
            const auto setup_error = update_shared_outputs(initial_targets);
            if (!setup_error.empty())
            {
                signal_setup(setup_error);
                running_ = false;
            }
            if (!running_)
            {
                std::lock_guard lock(host_mutex_);
                backend_ = nullptr;
                host_->renderer_service().set_backend(nullptr);
                return;
            }
        }
        signal_setup({});

        while (running_)
        {
            jobs_->pump_render_thread(32);
            std::vector<arc::editor::host_viewport_pointer_command> pointer_inputs;
            std::vector<arc::editor::host_viewport_key_command> key_inputs;
            std::vector<shared_render_target> shared_targets;
            MSG message{};
            while (PeekMessageW(&message, nullptr, 0, 0, PM_REMOVE))
            {
                if (message.message == WM_QUIT) running_ = false;
                TranslateMessage(&message);
                DispatchMessageW(&message);
            }

            if (shared_texture_)
            {
                {
                    std::lock_guard lock(bounds_mutex_);
                    pointer_inputs.swap(pending_pointer_inputs_);
                    key_inputs.swap(pending_key_inputs_);
                }
                const auto update_error = update_shared_outputs(shared_targets);
                if (!update_error.empty())
                    std::cerr << "arc_host_process shared viewport output error: " << update_error << '\n';
            }
            else
            {
                std::lock_guard lock(bounds_mutex_);
                pointer_inputs.swap(pending_pointer_inputs_);
                key_inputs.swap(pending_key_inputs_);
                for (const auto& release : pending_releases_)
                    if (backend_)
                        backend_->release_viewport_frame(release.viewport_id, release.generation, release.frame_id);
                pending_releases_.clear();
                if (visibility_dirty_ && backend_)
                {
                    visibility_dirty_ = false;
                }
                if (bounds_dirty_)
                {
                    value = {.parent = parent_, .x = x_, .y = y_, .width = width_, .height = height_};
                    bounds_dirty_ = false;
                    apply_bounds(value);
                }
            }

            for (const auto& pointer_input : pointer_inputs)
                process_pointer(pointer_input);
            for (const auto& key_input : key_inputs)
                process_key(key_input);

            if (!attached_)
            {
                std::this_thread::sleep_for(arc::editor::defaults::native_viewport_frame_interval);
                continue;
            }

            if (shared_texture_)
            {
                update_terrain_hover();
                for (const auto& target : shared_targets)
                {
                    if (!running_) break;
                    render_shared_once(target);
                }
            }
            else
                render_once(value);
            std::this_thread::sleep_for(arc::editor::defaults::native_viewport_frame_interval);
        }

        {
            std::lock_guard lock(host_mutex_);
            if (backend_ && shared_texture_)
                for (const auto& [id, surface] : shared_surfaces_)
                    if (surface.output_created) backend_->destroy_viewport_output(id);
            host_->renderer_service().set_backend(nullptr);
            backend_ = nullptr;
        }
        if (window_)
        {
            DestroyWindow(window_);
            window_ = nullptr;
        }
    }

    void signal_setup(std::string error)
    {
        {
            std::lock_guard lock(setup_mutex_);
            setup_error_ = std::move(error);
            setup_complete_ = true;
        }
        setup_cv_.notify_all();
    }

    std::shared_ptr<arc::editor::arc_host> host_;
    std::mutex& host_mutex_;
    std::mutex& output_mutex_;
    arc::jobs::job_system* jobs_{};
    arc::jobs::job_handle render_task_;
    std::atomic<bool> running_{};
    std::atomic<bool> attached_{};
    mutable std::mutex bounds_mutex_;
    HWND parent_{};
    HWND window_{};
    arc::render::render_backend* backend_{};
    std::int32_t x_{};
    std::int32_t y_{};
    std::uint32_t width_{1};
    std::uint32_t height_{1};
    std::string viewport_id_{"viewport-1"};
    bool bounds_dirty_{};
    bool shared_texture_{};
    bool visible_{true};
    bool visibility_dirty_{};
    DWORD consumer_process_id_{};
    std::unordered_map<std::string, shared_surface_state> shared_surfaces_;
    std::vector<pending_release> pending_releases_;
    std::vector<arc::editor::host_viewport_pointer_command> pending_pointer_inputs_;
    std::vector<arc::editor::host_viewport_key_command> pending_key_inputs_;
    std::mutex setup_mutex_;
    std::condition_variable setup_cv_;
    bool setup_complete_{};
    std::string setup_error_;
    std::uint64_t shared_event_sequence_{std::uint64_t{1} << 62u};
    bool input_alt_{};
    bool input_shift_{};
    bool input_control_{};
    std::uint64_t frame_index_{};
    std::string last_render_error_;
    std::chrono::steady_clock::time_point last_render_error_time_{};
    std::chrono::steady_clock::time_point last_backend_recovery_attempt_{};
    bool dragging_{};
    drag_button drag_button_{drag_button::none};
    camera_drag_mode camera_drag_mode_{camera_drag_mode::none};
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
    arc::editor::gizmo_axis active_axis_{arc::editor::gizmo_axis::none};
    arc::editor::host_entity_id manipulation_entity_{};
    arc::editor::host_transform manipulation_original_{};
    arc::editor::host_transform manipulation_current_{};
    int manipulation_start_x_{};
    int manipulation_start_y_{};
    std::uint64_t manipulation_transaction_{};
    std::uint64_t next_manipulation_transaction_{};
    arc::math::vector3f manipulation_local_axis_{1.0f, 0.0f, 0.0f};
    arc::math::vector3f manipulation_rotation_axis_{1.0f, 0.0f, 0.0f};
    arc::math::vector2f manipulation_screen_direction_{1.0f, 0.0f};
    float manipulation_world_units_per_pixel_{0.02f};
    bool manipulation_rotation_is_local_{};
    bool sun_rotating_{};
    arc::editor::host_entity_id sun_entity_{};
    std::uint64_t sun_transaction_{};
    arc::editor::editor_sun_controller sun_controller_{};
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
    native_viewport_controller(std::shared_ptr<arc::editor::arc_host>, std::mutex&, std::mutex&, arc::jobs::job_system&)
    {
    }

    void attach(std::string, std::uint64_t, std::int32_t, std::int32_t, std::uint32_t, std::uint32_t)
    {
        std::cerr << "arc_host_process native viewport rendering is not available in this build\n";
    }

    bool create_shared(std::string, std::uint64_t, std::uint32_t, std::uint32_t, std::string& error)
    {
        error = "Shared viewport rendering is not available in this build";
        return false;
    }

    void release_frame(std::string, std::uint64_t, std::uint64_t, std::string) {}

    void set_visible(std::string_view, bool) {}

    void pointer(const arc::editor::host_viewport_pointer_command&) {}

    void key(const arc::editor::host_viewport_key_command&) {}

    void resize(std::string_view, std::int32_t, std::int32_t, std::uint32_t, std::uint32_t) {}

    void detach(std::string_view) {}

    void stop() {}
};

#endif

} // namespace

int main()
{
    auto& memory = arc::memory::default_memory_system();
    arc::jobs::job_system jobs({.worker_count = 0,
                                .run_inline = false,
                                .io_worker_count = 2,
                                .enable_render_thread = true,
                                .profile_event_capacity = 8192,
                                .memory = &memory});
    jobs.register_main_thread();
    auto host = std::make_shared<arc::editor::arc_host>(std::make_unique<arc::render::renderer>());
    std::mutex host_mutex;
    std::mutex output_mutex;
    native_viewport_controller native_viewport(host, host_mutex, output_mutex, jobs);
    const auto write_response = [&](const arc::editor::host_response& response)
    {
        std::lock_guard output_lock(output_mutex);
        std::cout << arc::editor::to_json(response) << '\n';
        std::cout.flush();
    };
    std::jthread event_pump(
        [&](std::stop_token stop)
        {
            auto next_profiler_sample = std::chrono::steady_clock::now();
            std::uint64_t profiler_sequence = std::uint64_t{1} << 63u;
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
                    const arc::editor::host_event event{.sequence = profiler_sequence++,
                                                        .event_type = arc::editor::host_event_type::profiler_snapshot,
                                                        .message = "Profiler snapshot",
                                                        .payload_json = arc::editor::to_json(snapshot)};
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
        if (line.empty()) continue;

        std::string error;
        if (line.find("\"kind\":\"query\"") != std::string::npos ||
            line.find("\"kind\": \"query\"") != std::string::npos)
        {
            arc::editor::host_query_envelope query;
            if (!arc::editor::from_json(line, query, error))
            {
                std::cerr << "arc_host_process query parse error: " << error << '\n';
                write_response(
                    arc::editor::host_response{.request_id = query.request_id, .succeeded = false, .error = error});
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
                write_response(
                    arc::editor::host_response{.request_id = command.request_id, .succeeded = false, .error = error});
                continue;
            }
            arc::editor::host_response response;
            if (const auto* create = std::get_if<arc::editor::host_viewport_create_command>(&command.payload);
                create && create->output == arc::editor::host_viewport_output_type::shared_texture)
            {
                {
                    std::lock_guard lock(host_mutex);
                    response = host->execute(command);
                }
                if (response.succeeded)
                {
                    std::string setup_error;
                    if (!native_viewport.create_shared(create->viewport_id, create->consumer_process_id, create->width,
                                                       create->height, setup_error))
                    {
                        response.succeeded = false;
                        response.error = std::move(setup_error);
                    }
                }
            }
            else
            {
                std::lock_guard lock(host_mutex);
                response = host->execute(command);
            }
            write_response(response);

            if (response.succeeded)
            {
                if (const auto* attach = std::get_if<arc::editor::host_viewport_attach_command>(&command.payload))
                    native_viewport.attach(attach->viewport_id, attach->native_handle, attach->x, attach->y,
                                           attach->width, attach->height);
                else if (const auto* resize = std::get_if<arc::editor::host_viewport_resize_command>(&command.payload))
                    native_viewport.resize(resize->viewport_id, resize->x, resize->y, resize->width, resize->height);
                else if (const auto* detach = std::get_if<arc::editor::host_viewport_detach_command>(&command.payload))
                    native_viewport.detach(detach->viewport_id);
                else if (const auto* release =
                             std::get_if<arc::editor::host_viewport_frame_released_command>(&command.payload))
                    native_viewport.release_frame(release->viewport_id, release->generation, release->frame_id,
                                                  release->consumer_handle);
                else if (const auto* visibility =
                             std::get_if<arc::editor::host_viewport_set_visibility_command>(&command.payload))
                    native_viewport.set_visible(visibility->viewport_id, visibility->visible);
                else if (const auto* pointer =
                             std::get_if<arc::editor::host_viewport_pointer_command>(&command.payload))
                    native_viewport.pointer(*pointer);
                else if (const auto* key = std::get_if<arc::editor::host_viewport_key_command>(&command.payload))
                    native_viewport.key(*key);
            }
        }
    }

    event_pump.request_stop();
    native_viewport.stop();
    return 0;
}
