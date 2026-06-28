#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif

#include <arc/framework.h>

#include <windows.h>
#include <windowsx.h>

#include <cstdint>
#include <memory>
#include <string>

namespace
{

std::wstring widen(const std::string& value)
{
    if (value.empty())
        return L"ARC Application";

    const int required = MultiByteToWideChar(
        CP_UTF8,
        MB_ERR_INVALID_CHARS,
        value.data(),
        static_cast<int>(value.size()),
        nullptr,
        0);

    if (required <= 0)
        return std::wstring(value.begin(), value.end());

    std::wstring result(static_cast<std::size_t>(required), L'\0');
    MultiByteToWideChar(
        CP_UTF8,
        MB_ERR_INVALID_CHARS,
        value.data(),
        static_cast<int>(value.size()),
        result.data(),
        required);
    return result;
}

arc::mouse_button translate_mouse_button(UINT message, WPARAM wparam)
{
    switch (message)
    {
    case WM_LBUTTONDOWN:
    case WM_LBUTTONUP:
        return arc::mouse_button::left;
    case WM_RBUTTONDOWN:
    case WM_RBUTTONUP:
        return arc::mouse_button::right;
    case WM_MBUTTONDOWN:
    case WM_MBUTTONUP:
        return arc::mouse_button::middle;
    case WM_XBUTTONDOWN:
    case WM_XBUTTONUP:
        return HIWORD(wparam) == XBUTTON1 ? arc::mouse_button::x1 : arc::mouse_button::x2;
    default:
        return arc::mouse_button::unknown;
    }
}

arc::runtime* runtime_from_window(HWND window)
{
    return reinterpret_cast<arc::runtime*>(GetWindowLongPtrW(window, GWLP_USERDATA));
}

LRESULT CALLBACK window_proc(HWND window, UINT message, WPARAM wparam, LPARAM lparam)
{
    if (message == WM_NCCREATE)
    {
        const auto* create = reinterpret_cast<const CREATESTRUCTW*>(lparam);
        SetWindowLongPtrW(window, GWLP_USERDATA, reinterpret_cast<LONG_PTR>(create->lpCreateParams));
    }

    arc::runtime* runtime = runtime_from_window(window);

    switch (message)
    {
    case WM_CLOSE:
        if (runtime)
        {
            arc::event event{};
            event.type = arc::event_type::close_requested;
            runtime->dispatch(event);
        }
        DestroyWindow(window);
        return 0;

    case WM_DESTROY:
        PostQuitMessage(0);
        return 0;

    case WM_SIZE:
        if (runtime)
        {
            arc::event event{};
            event.type = arc::event_type::resized;
            event.width = static_cast<std::uint32_t>(LOWORD(lparam));
            event.height = static_cast<std::uint32_t>(HIWORD(lparam));
            runtime->dispatch(event);
        }
        return 0;

    case WM_SETFOCUS:
    case WM_KILLFOCUS:
        if (runtime)
        {
            arc::event event{};
            event.type = message == WM_SETFOCUS ? arc::event_type::focus_gained : arc::event_type::focus_lost;
            runtime->dispatch(event);
        }
        return 0;

    case WM_KEYDOWN:
    case WM_SYSKEYDOWN:
    case WM_KEYUP:
    case WM_SYSKEYUP:
        if (runtime)
        {
            arc::event event{};
            event.type = (message == WM_KEYDOWN || message == WM_SYSKEYDOWN) ? arc::event_type::key_down : arc::event_type::key_up;
            event.key_code = static_cast<int>(wparam);
            event.repeat = (lparam & (1 << 30)) != 0;
            runtime->dispatch(event);
        }
        return 0;

    case WM_MOUSEMOVE:
        if (runtime)
        {
            arc::event event{};
            event.type = arc::event_type::mouse_moved;
            event.x = GET_X_LPARAM(lparam);
            event.y = GET_Y_LPARAM(lparam);
            runtime->dispatch(event);
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
            arc::event event{};
            event.type =
                (message == WM_LBUTTONDOWN || message == WM_RBUTTONDOWN || message == WM_MBUTTONDOWN || message == WM_XBUTTONDOWN)
                    ? arc::event_type::mouse_button_down
                    : arc::event_type::mouse_button_up;
            event.button = translate_mouse_button(message, wparam);
            event.x = GET_X_LPARAM(lparam);
            event.y = GET_Y_LPARAM(lparam);
            runtime->dispatch(event);
        }
        return 0;

    case WM_MOUSEWHEEL:
        if (runtime)
        {
            arc::event event{};
            event.type = arc::event_type::mouse_wheel;
            event.wheel_delta = static_cast<float>(GET_WHEEL_DELTA_WPARAM(wparam)) / static_cast<float>(WHEEL_DELTA);
            event.x = GET_X_LPARAM(lparam);
            event.y = GET_Y_LPARAM(lparam);
            runtime->dispatch(event);
        }
        return 0;

    default:
        return DefWindowProcW(window, message, wparam, lparam);
    }
}

DWORD window_style(const arc::application_config& config)
{
    if (config.resizable)
        return WS_OVERLAPPEDWINDOW;

    return WS_OVERLAPPED | WS_CAPTION | WS_SYSMENU | WS_MINIMIZEBOX;
}

} // namespace

int WINAPI WinMain(HINSTANCE instance, HINSTANCE, LPSTR, int show_command)
{
    std::unique_ptr<arc::application> app = arc::create_application();
    if (!app)
        return -1;

    arc::runtime runtime(*app);
    const arc::application_config& config = runtime.config();
    const std::wstring class_name = L"ArcWindowsHost";
    const std::wstring title = widen(config.title);

    WNDCLASSEXW window_class{};
    window_class.cbSize = sizeof(window_class);
    window_class.style = CS_HREDRAW | CS_VREDRAW;
    window_class.lpfnWndProc = window_proc;
    window_class.hInstance = instance;
    window_class.hCursor = LoadCursorW(nullptr, MAKEINTRESOURCEW(32512));
    window_class.lpszClassName = class_name.c_str();

    if (!RegisterClassExW(&window_class))
        return -2;

    RECT rect{};
    rect.right = static_cast<LONG>(config.initial_width);
    rect.bottom = static_cast<LONG>(config.initial_height);
    const DWORD style = window_style(config);
    AdjustWindowRect(&rect, style, FALSE);

    HWND window = CreateWindowExW(
        0,
        class_name.c_str(),
        title.c_str(),
        style,
        CW_USEDEFAULT,
        CW_USEDEFAULT,
        rect.right - rect.left,
        rect.bottom - rect.top,
        nullptr,
        nullptr,
        instance,
        &runtime);

    if (!window)
        return -3;

    runtime.start();

    if (config.visible)
    {
        ShowWindow(window, show_command);
        UpdateWindow(window);
        if (config.start_focused)
            SetForegroundWindow(window);
    }

    MSG message{};
    while (runtime.running())
    {
        while (PeekMessageW(&message, nullptr, 0, 0, PM_REMOVE))
        {
            if (message.message == WM_QUIT)
            {
                runtime.request_stop();
                break;
            }

            TranslateMessage(&message);
            DispatchMessageW(&message);
        }

        if (runtime.running())
            runtime.tick();
    }

    runtime.shutdown();
    return 0;
}
