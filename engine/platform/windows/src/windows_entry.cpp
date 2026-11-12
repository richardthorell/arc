#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif

#include "windows_controller_manager.h"
#include "windows_input_backend.h"

#include <arc/framework/framework.h>
#if defined(ARC_WINDOWS_PACKAGED_BOOTSTRAP)
#include <arc/platform/windows/packaged_client.h>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <system_error>
#include <utility>
#endif

#include <windows.h>
#include <windowsx.h>

#include <cstdint>
#include <iterator>
#include <memory>
#include <string>
#include <string_view>

namespace
{

struct window_state
{
    arc::framework::runtime* runtime{};
    arc::platform::windows::windows_input_backend* input{};
    arc::platform::windows::windows_controller_manager* controllers{};
#if defined(ARC_WINDOWS_PACKAGED_BOOTSTRAP)
    arc::project::packaged_runtime_session* session{};
    std::string failure;
#endif
};

std::wstring widen(const std::string& value)
{
    if (value.empty()) return L"ARC Application";

    const int required =
        MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS, value.data(), static_cast<int>(value.size()), nullptr, 0);

    if (required <= 0) return std::wstring(value.begin(), value.end());

    std::wstring result(static_cast<std::size_t>(required), L'\0');
    MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS, value.data(), static_cast<int>(value.size()), result.data(),
                        required);
    return result;
}

arc::framework::mouse_button translate_mouse_button(UINT message, WPARAM wparam)
{
    switch (message)
    {
        case WM_LBUTTONDOWN:
        case WM_LBUTTONUP:
            return arc::framework::mouse_button::left;
        case WM_RBUTTONDOWN:
        case WM_RBUTTONUP:
            return arc::framework::mouse_button::right;
        case WM_MBUTTONDOWN:
        case WM_MBUTTONUP:
            return arc::framework::mouse_button::middle;
        case WM_XBUTTONDOWN:
        case WM_XBUTTONUP:
            return HIWORD(wparam) == XBUTTON1 ? arc::framework::mouse_button::x1 : arc::framework::mouse_button::x2;
        default:
            return arc::framework::mouse_button::unknown;
    }
}

window_state* state_from_window(HWND window)
{
    return reinterpret_cast<window_state*>(GetWindowLongPtrW(window, GWLP_USERDATA));
}

#if defined(ARC_WINDOWS_PACKAGED_BOOTSTRAP)
void record_status(window_state* state, arc::project::project_status status)
{
    if (state && !status)
    {
        state->failure = status.error().field + ": " + status.error().message;
        state->session->request_stop();
    }
}
#endif

void forward_event(window_state* state, const arc::framework::event& event)
{
#if defined(ARC_WINDOWS_PACKAGED_BOOTSTRAP)
    if (state && state->session)
    {
        record_status(state, state->session->dispatch(event));
        return;
    }
#endif
    if (state && state->runtime) state->runtime->dispatch(event);
}

LRESULT CALLBACK window_proc(HWND window, UINT message, WPARAM wparam, LPARAM lparam)
{
    if (message == WM_NCCREATE)
    {
        const auto* create = reinterpret_cast<const CREATESTRUCTW*>(lparam);
        SetWindowLongPtrW(window, GWLP_USERDATA, reinterpret_cast<LONG_PTR>(create->lpCreateParams));
    }

    window_state* state = state_from_window(window);
    if (state && state->input) state->input->process_message(message, wparam, lparam);
    if (state && state->controllers) state->controllers->process_message(message, wparam, lparam);
    arc::framework::runtime* runtime = state ? state->runtime : nullptr;

    switch (message)
    {
        case WM_CLOSE:
            if (runtime)
            {
                arc::framework::event event{};
                event.type = arc::framework::event_type::close_requested;
                forward_event(state, event);
            }
#if defined(ARC_WINDOWS_PACKAGED_BOOTSTRAP)
            if (state && state->session) state->session->request_stop();
#endif
            DestroyWindow(window);
            return 0;

        case WM_DESTROY:
            PostQuitMessage(0);
            return 0;

        case WM_SIZE:
            if (runtime)
            {
                arc::framework::event event{};
                event.type = arc::framework::event_type::resized;
                event.width = static_cast<std::uint32_t>(LOWORD(lparam));
                event.height = static_cast<std::uint32_t>(HIWORD(lparam));
#if defined(ARC_WINDOWS_PACKAGED_BOOTSTRAP)
                if (state && state->session)
                    record_status(state, event.width && event.height
                                             ? state->session->surface_ready(event.width, event.height)
                                             : state->session->surface_lost());
                else
#endif
                    forward_event(state, event);
            }
            return 0;

        case WM_DPICHANGED:
        {
            const auto* suggested = reinterpret_cast<const RECT*>(lparam);
            if (suggested)
            {
                SetWindowPos(window, nullptr, suggested->left, suggested->top, suggested->right - suggested->left,
                             suggested->bottom - suggested->top, SWP_NOACTIVATE | SWP_NOZORDER);
            }
            return 0;
        }

        case WM_SETFOCUS:
        case WM_KILLFOCUS:
            if (runtime)
            {
                arc::framework::event event{};
                event.type = message == WM_SETFOCUS ? arc::framework::event_type::focus_gained
                                                    : arc::framework::event_type::focus_lost;
                forward_event(state, event);
#if defined(ARC_WINDOWS_PACKAGED_BOOTSTRAP)
                if (state && state->session &&
                    (state->session->state() == arc::project::packaged_runtime_state::running ||
                     state->session->state() == arc::project::packaged_runtime_state::suspended))
                    record_status(state, message == WM_SETFOCUS ? state->session->resume() : state->session->pause());
#endif
            }
            return 0;

        case WM_KEYDOWN:
        case WM_SYSKEYDOWN:
        case WM_KEYUP:
        case WM_SYSKEYUP:
            if (runtime)
            {
                arc::framework::event event{};
                event.type = (message == WM_KEYDOWN || message == WM_SYSKEYDOWN) ? arc::framework::event_type::key_down
                                                                                 : arc::framework::event_type::key_up;
                event.key_code = static_cast<int>(wparam);
                event.repeat = (lparam & (1 << 30)) != 0;
                forward_event(state, event);
            }
            return 0;

        case WM_MOUSEMOVE:
            if (runtime)
            {
                arc::framework::event event{};
                event.type = arc::framework::event_type::mouse_moved;
                event.x = GET_X_LPARAM(lparam);
                event.y = GET_Y_LPARAM(lparam);
                forward_event(state, event);
            }
            return 0;

        case WM_LBUTTONDOWN:
        case WM_RBUTTONDOWN:
        case WM_MBUTTONDOWN:
        case WM_XBUTTONDOWN:
        case WM_LBUTTONUP:
        case WM_RBUTTONUP:
        case WM_MBUTTONUP:
        case WM_XBUTTONUP:
            if (runtime)
            {
                arc::framework::event event{};
                event.type = (message == WM_LBUTTONDOWN || message == WM_RBUTTONDOWN || message == WM_MBUTTONDOWN ||
                              message == WM_XBUTTONDOWN)
                                 ? arc::framework::event_type::mouse_button_down
                                 : arc::framework::event_type::mouse_button_up;
                event.button = translate_mouse_button(message, wparam);
                event.x = GET_X_LPARAM(lparam);
                event.y = GET_Y_LPARAM(lparam);
                forward_event(state, event);
            }
            return 0;

        case WM_MOUSEWHEEL:
            if (runtime)
            {
                arc::framework::event event{};
                event.type = arc::framework::event_type::mouse_wheel;
                event.wheel_delta =
                    static_cast<float>(GET_WHEEL_DELTA_WPARAM(wparam)) / static_cast<float>(WHEEL_DELTA);
                event.x = GET_X_LPARAM(lparam);
                event.y = GET_Y_LPARAM(lparam);
                forward_event(state, event);
            }
            return 0;

        default:
            return DefWindowProcW(window, message, wparam, lparam);
    }
}

DWORD window_style(const arc::framework::application_config& config)
{
    if (config.resizable) return WS_OVERLAPPEDWINDOW;

    return WS_OVERLAPPED | WS_CAPTION | WS_SYSMENU | WS_MINIMIZEBOX;
}

arc::framework::platform_capabilities query_platform_capabilities() noexcept
{
    SYSTEM_INFO system{};
    GetNativeSystemInfo(&system);
    MEMORYSTATUSEX memory{};
    memory.dwLength = sizeof(memory);
    const bool memory_available = GlobalMemoryStatusEx(&memory) == TRUE;
    return {.family = arc::framework::platform_family::windows,
            .form_factor = arc::framework::device_form_factor::desktop,
            .logical_processor_count = system.dwNumberOfProcessors,
            .system_memory_bytes = memory_available ? memory.ullTotalPhys : 0u,
            .window_system = true,
            .high_dpi = true,
            .multiple_windows = true,
            .dynamic_libraries = true,
            .persistent_local_storage = true,
            .native_package_assets = false};
}

#if defined(ARC_WINDOWS_PACKAGED_BOOTSTRAP)
std::filesystem::path executable_path()
{
    std::wstring buffer(512, L'\0');
    for (;;)
    {
        const DWORD size = GetModuleFileNameW(nullptr, buffer.data(), static_cast<DWORD>(buffer.size()));
        if (!size || size >= 32768) return {};
        if (size < buffer.size())
        {
            buffer.resize(size);
            return std::filesystem::path(buffer);
        }
        buffer.resize(buffer.size() * 2);
    }
}

std::filesystem::path writable_data_root(const std::string& project_guid)
{
    const DWORD required = GetEnvironmentVariableW(L"LOCALAPPDATA", nullptr, 0);
    if (required <= 1) return {};
    std::wstring value(required, L'\0');
    const DWORD size = GetEnvironmentVariableW(L"LOCALAPPDATA", value.data(), required);
    if (!size || size >= required) return {};
    value.resize(size);
    return std::filesystem::path(value) / L"ARC" / widen(project_guid);
}

// GUI-subsystem applications do not necessarily have stdout/stderr. Write a
// deterministic CI-only diagnostic file rather than displaying modal UI.
bool write_ci_smoke_diagnostic(std::string_view text)
{
    const DWORD length = GetEnvironmentVariableW(L"ARC_CI_SMOKE_LOG", nullptr, 0);
    if (length <= 1 || length > 32768) return false;
    std::wstring path(length, L'\0');
    const DWORD actual = GetEnvironmentVariableW(L"ARC_CI_SMOKE_LOG", path.data(), length);
    if (actual == 0 || actual >= length) return false;
    path.resize(actual);
    std::ofstream out(std::filesystem::path(path), std::ios::binary | std::ios::trunc);
    if (!out) return false;
    out << text << '\n';
    out.flush();
    return out.good();
}

int boot_failure(const std::string& message)
{
    const auto wide = widen(message);
    OutputDebugStringW(wide.c_str());
    if (!write_ci_smoke_diagnostic("failure: " + message))
        MessageBoxW(nullptr, wide.c_str(), L"ARC packaged client failure", MB_OK | MB_ICONERROR);
    return -5;
}
#endif

} // namespace

int WINAPI WinMain(HINSTANCE instance, HINSTANCE, LPSTR command_line, int show_command)
{
    SetProcessDpiAwarenessContext(DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2);

#if !defined(ARC_WINDOWS_PACKAGED_BOOTSTRAP)
    (void)command_line;
#endif

#if defined(ARC_WINDOWS_PACKAGED_BOOTSTRAP)
    namespace fs = std::filesystem;
    namespace project = arc::project;
    auto definition = arc::platform::windows::configure_packaged_client();
    const fs::path executable = executable_path();
    const fs::path storage = writable_data_root(definition.expected.project_guid);
    if (!executable.is_absolute() || !storage.is_absolute())
        return boot_failure("Cannot resolve the native executable or per-user application storage");

    std::error_code storage_error;
    fs::create_directories(storage, storage_error);
    if (storage_error) return boot_failure("Cannot create writable application storage: " + storage_error.message());

    arc::jobs::job_system jobs{
        {.worker_count = 1, .run_inline = false, .io_worker_count = 2, .enable_render_thread = false}};
    arc::io::async_file_service native_files{jobs};
    arc::io::virtual_file_system files{jobs};
    project::runtime_boot_request boot;
    boot.role = project::runtime_launch_role::client;
    boot.host.distribution.kind = project::runtime_distribution_location_kind::executable;
    boot.host.distribution.native_path = executable;
    boot.host.writable_storage_root = storage;
    boot.host.capabilities = query_platform_capabilities();
    boot.expected = std::move(definition.expected);
    project::packaged_runtime_request request{.boot = std::move(boot),
                                              .files = files,
                                              .native_files = &native_files,
                                              .registrations = std::move(definition.registrations),
                                              .config = std::move(definition.config)};
    auto started = project::start_packaged_runtime(std::move(request));
    if (!started) return boot_failure(started.error().field + ": " + started.error().message);
    auto session = std::move(started).value();
    // This exercises the real Shipping bootstrap and startup scene on CI
    // without requiring a GPU or opening a window. Normal launches are unchanged.
    if (command_line && std::string_view(command_line) == "--ci-smoke")
    {
        auto stopped = session->shutdown();
        if (!stopped) return boot_failure(stopped.error().field + ": " + stopped.error().message);
        if (!write_ci_smoke_diagnostic("ARC packaged client bootstrap smoke passed"))
            return boot_failure("Cannot publish CI smoke success diagnostic");
        return 0;
    }
    arc::framework::runtime& runtime = *session->runtime();
#else
    std::unique_ptr<arc::framework::application> app = arc::framework::create_application();
    if (!app) return -1;

    arc::framework::runtime runtime(*app);
    runtime.set_platform_capabilities(query_platform_capabilities());
#endif
    arc::platform::windows::windows_input_backend input_backend(runtime.input());
    arc::platform::windows::windows_controller_manager controller_manager(runtime.input());
    window_state state{.runtime = &runtime,
                       .input = &input_backend,
                       .controllers = &controller_manager
#if defined(ARC_WINDOWS_PACKAGED_BOOTSTRAP)
                       ,
                       .session = session.get()
#endif
    };
    const arc::framework::application_config& config = runtime.config();
    const std::wstring class_name = L"ArcWindowsHost";
    const std::wstring title = widen(config.title);

    WNDCLASSEXW window_class{};
    window_class.cbSize = sizeof(window_class);
    window_class.style = CS_HREDRAW | CS_VREDRAW;
    window_class.lpfnWndProc = window_proc;
    window_class.hInstance = instance;
    window_class.hCursor = LoadCursorW(nullptr, MAKEINTRESOURCEW(32512));
    window_class.lpszClassName = class_name.c_str();

    if (!RegisterClassExW(&window_class)) return -2;

    RECT rect{};
    rect.right = static_cast<LONG>(config.initial_width);
    rect.bottom = static_cast<LONG>(config.initial_height);
    const DWORD style = window_style(config);
    AdjustWindowRectExForDpi(&rect, style, FALSE, 0, GetDpiForSystem());

    HWND window = CreateWindowExW(0, class_name.c_str(), title.c_str(), style, CW_USEDEFAULT, CW_USEDEFAULT,
                                  rect.right - rect.left, rect.bottom - rect.top, nullptr, nullptr, instance, &state);

    if (!window) return -3;
    if (!input_backend.attach(window))
    {
        DestroyWindow(window);
        return -4;
    }
    static_cast<void>(controller_manager.attach(window));

#if !defined(ARC_WINDOWS_PACKAGED_BOOTSTRAP)
    runtime.start();
#else
    record_status(&state, session->surface_ready(config.initial_width, config.initial_height));
#endif

    if (config.visible)
    {
        ShowWindow(window, config.maximized ? SW_SHOWMAXIMIZED : show_command);
        UpdateWindow(window);
        if (config.start_focused) SetForegroundWindow(window);
    }

    MSG message{};
#if defined(ARC_WINDOWS_PACKAGED_BOOTSTRAP)
    using clock = std::chrono::steady_clock;
    auto last_frame = clock::now();
    while (session->state() == project::packaged_runtime_state::running ||
           session->state() == project::packaged_runtime_state::suspended)
#else
    while (runtime.running())
#endif
    {
        runtime.input().begin_frame();
        controller_manager.poll();
        while (PeekMessageW(&message, nullptr, 0, 0, PM_REMOVE))
        {
            if (message.message == WM_QUIT)
            {
#if defined(ARC_WINDOWS_PACKAGED_BOOTSTRAP)
                session->request_stop();
#else
                runtime.request_stop();
#endif
                break;
            }

            TranslateMessage(&message);
            DispatchMessageW(&message);
        }

#if defined(ARC_WINDOWS_PACKAGED_BOOTSTRAP)
        const auto now = clock::now();
        const double delta = std::chrono::duration<double>(now - last_frame).count();
        last_frame = now;
        if (session->state() == project::packaged_runtime_state::running)
        {
            auto frame = session->advance(delta);
            if (!frame)
            {
                state.failure = frame.error().field + ": " + frame.error().message;
                session->request_stop();
            }
        }
        else if (session->state() == project::packaged_runtime_state::suspended)
            Sleep(8);
#else
        if (runtime.running()) runtime.tick();
#endif
    }

#if defined(ARC_WINDOWS_PACKAGED_BOOTSTRAP)
    state.runtime = nullptr;
    state.session = nullptr;
    auto stopped = session->shutdown();
    if (!stopped && state.failure.empty()) state.failure = stopped.error().field + ": " + stopped.error().message;
    if (!state.failure.empty()) return boot_failure(state.failure);
#else
    runtime.shutdown();
#endif
    return 0;
}
