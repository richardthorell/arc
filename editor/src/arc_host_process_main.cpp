#include <arc/editor/arc_host.h>
#include <arc/editor/editor_defaults.h>
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
#include <iostream>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <variant>
#include <vector>

namespace
{

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
    native_viewport_controller(std::shared_ptr<arc::editor::arc_host> host, std::mutex& host_mutex)
        : host_(std::move(host))
        , host_mutex_(host_mutex)
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
            render_thread_ = std::thread([this] { render_loop(); });
    }

    void resize(std::int32_t x, std::int32_t y, std::uint32_t width, std::uint32_t height)
    {
        std::lock_guard lock(bounds_mutex_);
        x_ = x;
        y_ = y;
        width_ = std::max(static_cast<std::uint32_t>(arc::editor::defaults::native_viewport_min_dimension), width);
        height_ = std::max(static_cast<std::uint32_t>(arc::editor::defaults::native_viewport_min_dimension), height);
        bounds_dirty_ = true;
    }

    void stop()
    {
        if (!running_.exchange(false))
            return;
        if (window_)
            PostMessageW(window_, WM_CLOSE, 0, 0);
        if (render_thread_.joinable())
            render_thread_.join();
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
            end_drag(window);
            return 0;
        case WM_MOUSEMOVE:
            update_drag(GET_X_LPARAM(lparam), GET_Y_LPARAM(lparam));
            return 0;
        case WM_MOUSEWHEEL:
            send_camera_input(arc::editor::host_viewport_camera_input_command{
                .zoom = static_cast<float>(GET_WHEEL_DELTA_WPARAM(wparam)) / static_cast<float>(WHEEL_DELTA)
            });
            return 0;
        case WM_CAPTURECHANGED:
            dragging_ = false;
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
        {
            std::lock_guard lock(host_mutex_);
            host_->request_viewport(arc::editor::host_viewport_request{
                .frame_index = frame_index_++,
                .width = value.width,
                .height = value.height
            });
            if (!backend_->render_native_viewport_frame(value.width, value.height, message) && !message.empty())
                std::cerr << "arc_host_process native viewport render error: " << message << '\n';
        }
    }

    void begin_drag(HWND window, UINT message, int x, int y)
    {
        dragging_ = true;
        drag_x_ = x;
        drag_y_ = y;
        drag_button_ = message == WM_MBUTTONDOWN
            ? drag_button::middle
            : message == WM_RBUTTONDOWN ? drag_button::right : drag_button::left;
        SetCapture(window);
    }

    void update_drag(int x, int y)
    {
        if (!dragging_)
            return;

        const int delta_x = x - drag_x_;
        const int delta_y = y - drag_y_;
        drag_x_ = x;
        drag_y_ = y;
        if (delta_x == 0 && delta_y == 0)
            return;

        const bool shift = (GetKeyState(VK_SHIFT) & arc::editor::defaults::viewport_modifier_key_down_mask) != 0;
        const bool alt = (GetKeyState(VK_MENU) & arc::editor::defaults::viewport_modifier_key_down_mask) != 0;

        arc::editor::host_viewport_camera_input_command input;
        if (alt)
        {
            input.forward = static_cast<float>(delta_y);
        }
        else if (shift || drag_button_ == drag_button::middle)
        {
            input.pan_x = static_cast<float>(delta_x);
            input.pan_y = static_cast<float>(delta_y);
        }
        else
        {
            input.orbit_x = static_cast<float>(delta_x);
            input.orbit_y = static_cast<float>(delta_y);
        }
        send_camera_input(input);
    }

    void end_drag(HWND window)
    {
        dragging_ = false;
        drag_button_ = drag_button::none;
        if (GetCapture() == window)
            ReleaseCapture();
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
    std::thread render_thread_;
    std::atomic<bool> running_{};
    std::mutex bounds_mutex_;
    HWND parent_{};
    HWND window_{};
    arc::render::vulkan::vulkan_backend* backend_{};
    std::int32_t x_{};
    std::int32_t y_{};
    std::uint32_t width_{ 1 };
    std::uint32_t height_{ 1 };
    bool bounds_dirty_{};
    std::uint64_t frame_index_{};
    bool dragging_{};
    drag_button drag_button_{ drag_button::none };
    int drag_x_{};
    int drag_y_{};
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
    native_viewport_controller(std::shared_ptr<arc::editor::arc_host>, std::mutex&)
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
    auto host = std::make_shared<arc::editor::arc_host>(std::make_unique<arc::render::renderer>());
    std::mutex host_mutex;
    native_viewport_controller native_viewport(host, host_mutex);

    std::string line;
    while (std::getline(std::cin, line))
    {
        if (line.empty())
            continue;

        std::string error;
        if (line.find("\"kind\":\"query\"") != std::string::npos || line.find("\"kind\": \"query\"") != std::string::npos)
        {
            arc::editor::host_query_envelope query;
            if (!arc::editor::from_json(line, query, error))
            {
                std::cerr << "arc_host_process query parse error: " << error << '\n';
                std::cout << arc::editor::to_json(arc::editor::host_response{
                    .request_id = query.request_id,
                    .succeeded = false,
                    .error = error }) << '\n';
                std::cout.flush();
                continue;
            }
            {
                std::lock_guard lock(host_mutex);
                std::cout << arc::editor::to_json(host->query(query)) << '\n';
            }
        }
        else
        {
            arc::editor::host_command_envelope command;
            if (!arc::editor::from_json(line, command, error))
            {
                std::cerr << "arc_host_process command parse error: " << error << '\n';
                std::cout << arc::editor::to_json(arc::editor::host_response{
                    .request_id = command.request_id,
                    .succeeded = false,
                    .error = error }) << '\n';
                std::cout.flush();
                continue;
            }
            arc::editor::host_response response;
            {
                std::lock_guard lock(host_mutex);
                response = host->execute(command);
            }
            std::cout << arc::editor::to_json(response) << '\n';

            if (response.succeeded)
            {
                if (const auto* attach = std::get_if<arc::editor::host_viewport_attach_command>(&command.payload))
                    native_viewport.attach(attach->native_handle, attach->x, attach->y, attach->width, attach->height);
                else if (const auto* resize = std::get_if<arc::editor::host_viewport_resize_command>(&command.payload))
                    native_viewport.resize(resize->x, resize->y, resize->width, resize->height);
            }
        }

        std::vector<arc::editor::host_event> events;
        {
            std::lock_guard lock(host_mutex);
            events = host->poll_events();
        }
        for (const auto& event : events)
            std::cout << arc::editor::to_json(event) << '\n';
        std::cout.flush();
    }

    native_viewport.stop();
    return 0;
}
